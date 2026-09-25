#!/usr/bin/env python3
"""Export the trained front-end to ONNX.

The exported graph covers the whole pipeline -- delay estimation, alignment,
the echo-estimate filter, the band features, the recurrent mask network and the
inverse STFT -- so a host program only has to feed in two waveforms.

The graph has static shapes. Inference therefore works on fixed-length segments
of ``--segment`` samples (3 s at 16 kHz by default); ``infer.py`` and the C++
example take care of cutting a longer recording into segments and cross-fading
the result. The segment length is a build-time choice: a longer segment gives
the recurrent network more context but costs more per call, and the internal
FFT size grows with it.

Usage:
    python export_onnx.py --ckpt weights/aec_lp.pt --out weights/aec_lp.onnx
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model import AecFrontend

SEGMENT = 48000          # samples per inference call
OPSET = 20               # ONNX DFT (used for the FFTs) needs >= 17


class ExportWrapper(nn.Module):
    """Returns the waveform and the estimated delay; drops the mask."""

    def __init__(self, model, segment):
        super().__init__()
        self.model = model
        self.segment = segment

    def forward(self, mic, ref):
        y, _, tau = self.model(mic, ref, length=self.segment)
        return y, tau


def load_checkpoint(path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = AecFrontend(**ck['config']).to(device)
    # strict=False: the sqrt-Hann window is a registered buffer that
    # every checkpoint recreates at construction
    model.load_state_dict(ck['model'], strict=False)
    for k, v in ck.get('knobs', {}).items():
        setattr(model, k, v)
    # A stateful per-frame delay tracker does not trace. The exported graph
    # therefore uses utterance-level alignment: one delay estimate for the
    # whole segment, applied as a single fractional shift. On a segment with a
    # stationary path the two modes agree to well under a sample of delay; a
    # drifting path is the one case where the Python/C++ runtime (stream mode)
    # and the ONNX graph differ, and the runtime is the reference.
    model.tde_mode = 'global'
    model.eval()
    return model, ck


def scrub_metadata(path):
    """Drop the debugging attributes the dynamo exporter attaches to nodes.

    ``stack_trace`` carries a Python traceback -- with the absolute paths of
    the machine the export ran on -- and ``nn_module_stack`` the module
    layout. Neither is needed to run the graph, and neither belongs in a
    shipped file.
    """
    import onnx
    m = onnx.load(path)
    dropped = 0

    def clean(obj):
        # metadata_props is a repeated StringStringEntryProto; the exporter
        # puts 'nn_module_stack' and 'stack_trace' there (a map, not the
        # classic op attributes)
        nonlocal dropped
        before = len(obj.metadata_props)
        if before:
            # the exporter's keys are prefixed ('pkg.torch.onnx.stack_trace',
            # 'pkg.torch.onnx.name_scopes', ...); drop the whole namespace,
            # it is debugging metadata with no effect on execution
            keep = [e for e in obj.metadata_props
                    if not e.key.startswith('pkg.torch.onnx')]
            del obj.metadata_props[:]
            obj.metadata_props.extend(keep)
            dropped += before - len(keep)
        if obj.doc_string:
            obj.doc_string = ''
            dropped += 1

    def walk(g):
        clean(g)
        for i, node in enumerate(g.node):
            clean(node)
            # the exporter names nodes after their op type with a counter
            # ('node_Sub_31', 'node_Add_7', ...). The names are pure labels,
            # and 'node_Sub_144' is indistinguishable at a glance from a
            # hexadecimal address, so relabel them neutrally.
            node.name = 'n%d' % i
        for sub in g.node:
            for a in sub.attribute:
                if a.HasField('g'):
                    walk(a.g)

    walk(m.graph)
    for f in m.functions:
        clean(f)
    onnx.save(m, path)
    print('scrubbed %d metadata attributes' % dropped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='weights/aec_lp.pt')
    ap.add_argument('--out', default='weights/aec_lp.onnx')
    ap.add_argument('--segment', type=int, default=SEGMENT)
    ap.add_argument('--opset', type=int, default=OPSET)
    ap.add_argument('--check', action='store_true', default=True,
                    help='verify the exported graph against PyTorch')
    args = ap.parse_args()

    model, ck = load_checkpoint(args.ckpt)
    print('config:', ck['config'], ' knobs:', ck.get('knobs', {}))

    wrapper = ExportWrapper(model, args.segment).eval()
    # a synthetic echo with a known delay, so the check below is meaningful:
    # on pure noise the delay estimate is undefined and any difference in it
    # shows up as a large waveform difference
    ref = torch.randn(1, args.segment)
    delay = 2528
    mic = 0.1 * torch.randn(1, args.segment)
    mic[:, delay:] += 0.5 * ref[:, :args.segment - delay]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # external_data=False keeps everything in one file: the dynamo exporter
    # otherwise splits the weights out into <name>.onnx.data
    torch.onnx.export(wrapper, (mic, ref), args.out, external_data=False,
                      opset_version=args.opset, dynamo=True,
                      input_names=['mic', 'ref'], output_names=['out', 'tau'],
                      report=False)
    scrub_metadata(args.out)
    size = Path(args.out).stat().st_size
    print('wrote %s (%.2f MB)' % (args.out, size / 1e6))

    if args.check:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=['CPUExecutionProvider'])
        got = sess.run(None, {'mic': mic.numpy(), 'ref': ref.numpy()})
        with torch.no_grad():
            want = wrapper(mic, ref)
        d = float(np.abs(got[0] - want[0].numpy()).max())
        rms = float(np.sqrt((want[0].numpy() ** 2).mean()))
        print('check: max |onnx - torch| = %.3e (output rms %.3e)   '
              'tau %.1f / %.1f  (true %d)'
              % (d, rms, float(got[1].reshape(-1)[0]),
                 float(want[1].reshape(-1)[0]), delay))


if __name__ == '__main__':
    main()

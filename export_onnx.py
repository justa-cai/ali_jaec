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
    model.load_state_dict(ck['model'])
    model.adapt_gain = ck.get('adapt_gain', 0)
    model.eval()
    return model, ck


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
    print('config:', ck['config'], ' adapt_gain:', model.adapt_gain)

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

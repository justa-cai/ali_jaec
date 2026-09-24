#!/usr/bin/env python3
"""Flatten the trained checkpoint into the raw array the C engine reads.

The C engine has no ONNX Runtime and no protobuf parser, so it cannot read
``aec_lp.onnx``. Instead this script writes one binary file:

    magic "AECLP01"  |  header (see struct Header below)  |  float32 arrays

The arrays follow each other in a fixed order and the header carries every
length, so the loader checks the counts it expects against the file and fails
loudly on a mismatch instead of reading garbage. Weights are reordered here,
once, into the layout the C inner loops want -- notably the echo filter, which
PyTorch stores as (bin, tap, re/im) and the C code wants as (tap, bin) with the
real and imaginary parts in separate arrays.

The derived quantities (the sqrt-Hann window, the lag axis of the delay
estimator) are *not* stored: they are deterministic functions of the design and
the C side recomputes them, in double precision and then rounded to float, the
same way the PyTorch code does.

Usage:
    python export_weights.py --ckpt weights/aec_lp.pt --out weights/aec_lp.bin
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import AecFrontend                                    # noqa: E402

MAGIC = b'AECLP01'
VERSION = 1

# Order matters: it is the order the C loader reads them in.
FIELDS = [
    'filt_re', 'filt_im',        # echo filter, (taps, freq)
    'bank_w', 'bank_b',          # band projection
    'reg0_w', 'reg0_b', 'reg2_w', 'reg2_b',   # delay regression head
    'gru_w_ih', 'gru_w_hh', 'gru_b_ih', 'gru_b_hh',
    'head_w', 'head_b',          # GRU -> band mask
    'mask_w', 'mask_b',          # band mask -> per-bin mask
]


def build(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = AecFrontend(**ck['config'])
    model.load_state_dict(ck['model'])
    sd = ck['model']

    filt = sd['filt'].numpy().astype(np.float32)            # (F, K, 2)
    filt_t = np.transpose(filt, (1, 0, 2))                  # (K, F, 2)

    arrays = {
        'filt_re': np.ascontiguousarray(filt_t[:, :, 0]).ravel(),
        'filt_im': np.ascontiguousarray(filt_t[:, :, 1]).ravel(),
        'bank_w': sd['bank.proj.weight'].numpy().astype(np.float32).ravel(),
        'bank_b': sd['bank.proj.bias'].numpy().astype(np.float32).ravel(),
        'reg0_w': sd['tde.reg.0.weight'].numpy().astype(np.float32).ravel(),
        'reg0_b': sd['tde.reg.0.bias'].numpy().astype(np.float32).ravel(),
        'reg2_w': sd['tde.reg.2.weight'].numpy().astype(np.float32).ravel(),
        'reg2_b': sd['tde.reg.2.bias'].numpy().astype(np.float32).ravel(),
        'gru_w_ih': sd['gru.weight_ih_l0'].numpy().astype(np.float32).ravel(),
        'gru_w_hh': sd['gru.weight_hh_l0'].numpy().astype(np.float32).ravel(),
        'gru_b_ih': sd['gru.bias_ih_l0'].numpy().astype(np.float32).ravel(),
        'gru_b_hh': sd['gru.bias_hh_l0'].numpy().astype(np.float32).ravel(),
        'head_w': sd['head.weight'].numpy().astype(np.float32).ravel(),
        'head_b': sd['head.bias'].numpy().astype(np.float32).ravel(),
        'mask_w': sd['mask_proj.weight'].numpy().astype(np.float32).ravel(),
        'mask_b': sd['mask_proj.bias'].numpy().astype(np.float32).ravel(),
    }

    cfg = ck['config']
    meta = dict(
        hid=cfg['hid'], bands=cfg['bands'],
        filt_taps=cfg['filt_taps'], max_delay=cfg['max_delay'],
        use_mic_feat=1 if cfg['use_mic_feat'] else 0,
        echo_gate=1 if cfg.get('echo_gate') else 0,
        adapt_gain=int(ck.get('adapt_gain', 0) or 0),
        mask_smooth=int(ck.get('mask_smooth', 0) or 0),
    )
    return arrays, meta


CONFIG_FIELDS = ('hid', 'bands', 'filt_taps', 'max_delay', 'use_mic_feat',
                 'echo_gate', 'adapt_gain', 'mask_smooth')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='weights/aec_lp.pt')
    ap.add_argument('--out', default='weights/aec_lp.bin')
    args = ap.parse_args()

    arrays, meta = build(args.ckpt)
    if meta['echo_gate']:
        raise SystemExit('echo_gate models are not supported by the C engine')

    counts = [len(arrays[f]) for f in FIELDS]
    # magic(8) + version + 16 lengths + 8 config = 8 + 4 + 64 + 32
    header = struct.pack('<8sI16I8I', MAGIC, VERSION, *counts,
                         *[meta[k] for k in CONFIG_FIELDS])
    assert len(header) == 108, len(header)

    blob = bytearray(header)
    for f in FIELDS:
        blob += arrays[f].tobytes()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)

    tot = sum(counts)
    print('%s -> %s' % (args.ckpt, out))
    print('  %d floats (%.1f kB), header 108 B, file %.1f kB'
          % (tot, tot * 4 / 1e3, len(blob) / 1e3))
    print('  config: ' + ' '.join('%s=%d' % (k, meta[k])
                                  for k in CONFIG_FIELDS))
    for f, n in zip(FIELDS, counts):
        print('    %-10s %7d' % (f, n))


if __name__ == '__main__':
    main()

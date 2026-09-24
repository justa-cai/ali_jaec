#!/usr/bin/env python3
"""Score a checkpoint on the held-out test split and print the acceptance table.

Usage:
    python evaluate.py --ckpt weights/aec_lp.pt --limit 500
"""
import argparse
import sys

import numpy as np
import torch

from dataset import load_arrays, load_meta, resolve_data, split_rows
from metrics import THRESHOLDS, score, summarise
from model import AecFrontend

SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='weights/aec_lp.pt')
    ap.add_argument('--data', help='packed dataset dir (or set AEC_DATA_DIR)')
    ap.add_argument('--limit', type=int, default=500)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    root = resolve_data(args.data)
    dev = args.device if torch.cuda.is_available() else 'cpu'
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model = AecFrontend(**ck['config']).to(dev).eval()
    model.load_state_dict(ck['model'])
    model.adapt_gain = ck.get('adapt_gain', 0)
    print('loaded %s  (val SI-SDR at save: %s)'
          % (args.ckpt, ck.get('val_si_sdr')))

    mic, ref, tgt = load_arrays(root)
    ids = split_rows(load_meta(root), 'test')[:args.limit]

    rows = []
    with torch.no_grad():
        for k, j in enumerate(ids):
            m = np.asarray(mic[j]); f = np.asarray(ref[j]); t = np.asarray(tgt[j])
            y, _, _ = model(torch.from_numpy(np.array(m))[None].to(dev),
                            torch.from_numpy(np.array(f))[None].to(dev),
                            length=len(m))
            rows.append(score(m, y[0].float().cpu().numpy().astype(np.float64),
                              f, t, delay=0))
            if (k + 1) % 100 == 0:
                print('  ...%d/%d' % (k + 1, len(ids)), flush=True)

    got = summarise(rows)
    print('\n=== acceptance (n=%d) ===' % len(ids))
    ok = True
    for key, (op, threshold) in THRESHOLDS.items():
        v = got[key]
        passed = v >= threshold if op == '>=' else v <= threshold
        ok = ok and passed
        print('  %-26s %8.3f   threshold %s %6.2f   %s'
              % (key, v, op, threshold, 'PASS' if passed else 'FAIL'))
    print('\n  overall: %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

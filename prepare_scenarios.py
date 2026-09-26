#!/usr/bin/env python3
"""Derive the scenario-coverage extension pack from a packed base corpus.

The AEC-Challenge synthetic corpus (what ``prepare_dataset.py`` packs) covers
one operating envelope: the far end is essentially always active and the echo
sits within +-10 dB of the near end. Four regimes a deployed canceller also
meets are absent, so this script derives them from existing TRAIN rows by
remixing ``mic' = tgt + g * (mic_src - tgt_src)`` -- the original echo path
character (EIR, reverb, noise) is kept and only its level moves:

  S1  far-end SILENT       ref' = 0; mic' = tgt + quiet uncorrelated noise.
      The reference channel is idle; the model must pass the mic through.
  S2  far ACTIVE, no echo  ref' = ref; mic' = tgt + a DIFFERENT row's echo.
      Nothing in mic' correlates with ref' (headphones, zero acoustic
      capture): the model must not suppress near-end speech just because
      the far end is loud.
  S3  weak echo            echo 12-35 dB BELOW the near end.
  S4  strong echo          echo 12-22 dB ABOVE the near end.

Rows are derived only from the base pack's train split, never its test split.
The output has the layout ``train.py --extra-data`` expects; its own 90/10
split measures scenario behaviour, not unseen-speaker generality (that stays
the base corpus's job).

Usage:
    python prepare_scenarios.py --data data --out data_scn
"""
import argparse
import json
from pathlib import Path

import numpy as np

from dataset import load_arrays, load_meta, resolve_data, split_rows

LEN = 160000           # 10 s at 16 kHz, one row
COUNTS = {'S1': 700, 'S2': 900, 'S3': 1300, 'S4': 900}
NOISE_ENR = (28.0, 38.0)      # S1/S2 room noise this far below the near end
S3_ENR = (12.0, 35.0)
S4_ENR = (-22.0, -12.0)


def db(x):
    return 10 * np.log10(np.mean(np.asarray(x, np.float64) ** 2) + 1e-20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', help='base packed dataset (or set AEC_DATA_DIR)')
    ap.add_argument('--out', default='data_scn')
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--counts', help='per-scenario row counts, e.g. '
                                     '"S1=700,S2=900,S3=1300,S4=900"')
    args = ap.parse_args()

    counts = dict(COUNTS)
    if args.counts:
        counts.update({k: int(v) for k, v in
                       (kv.split('=') for kv in args.counts.split(','))})

    root = resolve_data(args.data)
    mic, ref, tgt = load_arrays(root)
    meta = load_meta(root)
    ids = meta.get('ids')
    train = sorted(int(ids[fid]) if ids else int(fid)
                   for fid, s in meta['split'].items() if s == 'train')
    print('base corpus: %s (%d train rows)' % (root, len(train)))

    rng = np.random.default_rng(args.seed)
    n = sum(counts.values())
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    out = {k: np.lib.format.open_memmap(out_root / ('%s.npy' % k), mode='w+',
                                        dtype=np.float32, shape=(n, LEN))
           for k in ('mic', 'ref', 'tgt')}
    split, scale, nonlinear, scenario = {}, {}, {}, {}
    r = 0
    for sc, cnt in counts.items():
        lo, hi = {'S1': NOISE_ENR, 'S2': NOISE_ENR,
                  'S3': S3_ENR, 'S4': S4_ENR}[sc]
        for _ in range(cnt):
            i = train[rng.integers(0, len(train))]
            j = train[rng.integers(0, len(train))]
            t = np.asarray(tgt[i], np.float64)
            f = np.asarray(ref[i], np.float64)
            if sc in ('S3', 'S4'):
                echo = np.asarray(mic[i], np.float64) - t
                enr0 = db(t) - db(echo)
                g = 10 ** ((enr0 - rng.uniform(lo, hi)) / 20.0)
                m = t + g * echo
            else:
                noise = np.asarray(mic[j], np.float64) - np.asarray(tgt[j], np.float64)
                g = 10 ** ((db(t) - db(noise) - rng.uniform(lo, hi)) / 20.0)
                m = t + g * noise
                f = np.zeros(LEN) if sc == 'S1' else f
            out['mic'][r] = m.astype(np.float32)
            out['ref'][r] = f.astype(np.float32)
            out['tgt'][r] = t.astype(np.float32)
            split[str(r)] = 'train' if rng.random() < 0.9 else 'test'
            scale[str(r)] = 1.0
            nonlinear[str(r)] = 0 if sc in ('S1', 'S2') else 1
            scenario[str(r)] = sc
            r += 1
        print('  %s  %d rows' % (sc, cnt))
    for k in out:
        out[k].flush()
    (out_root / 'meta.json').write_text(json.dumps(
        {'split': split, 'scale': scale, 'nonlinear': nonlinear,
         'scenario': scenario}))
    print('wrote %s  (%d rows, %.1f GB)'
          % (out_root, n, 3 * n * LEN * 4 / 1e9))


if __name__ == '__main__':
    main()

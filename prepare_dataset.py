#!/usr/bin/env python3
"""Pack the AEC-Challenge corpus into the arrays the trainer reads.

Nothing here is specific to one machine: point ``--dataset`` at wherever you
unpacked the corpus, or export ``AEC_DATASET_DIR``. Download the corpus from
the AEC-Challenge repository (see README) and unpack it so that this layout
exists:

    <dataset>/
        meta.csv
        extract/
            synthetic_nearend_mic/nearend_mic_fileid_<N>.flac
            synthetic_farend/farend_speech_fileid_<N>.flac
            synthetic_nearend_speech/nearend_speech_fileid_<N>.flac

For every ``fileid`` listed in ``meta.csv`` the script writes three float32
arrays of shape ``(n_files, 160000)`` -- 10 s at 16 kHz:

    mic  the near-end microphone signal, i.e. near-end speech plus echo
    ref  the far-end reference (what the loudspeaker plays)
    tgt  the clean near-end speech, scaled by the corpus's ``nearend_scale``

plus ``meta.json`` holding the per-file scale and the train/test split. Reading
these memory-mapped is roughly 20x faster than decoding the FLACs on every
epoch, which matters because the packed set is ~19 GB.

Usage:
    python prepare_dataset.py --dataset /path/to/AEC-Challenge --out data
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
LEN = 160000           # 10 s


def resolve_dataset(arg):
    root = arg or os.environ.get('AEC_DATASET_DIR')
    if not root:
        raise SystemExit(
            'dataset path not given.\n'
            '  pass --dataset /path/to/AEC-Challenge, or set the '
            'AEC_DATASET_DIR environment variable.')
    root = Path(root).expanduser()
    if not (root / 'meta.csv').is_file():
        raise SystemExit('no meta.csv under %s -- is that the corpus root?' % root)
    return root


def read_meta(root):
    """-> list of (fileid, split, nearend_scale) sorted by fileid."""
    rows = []
    with open(root / 'meta.csv', newline='') as fh:
        for row in csv.DictReader(fh):
            rows.append((int(row['fileid']), row['split'],
                         float(row['nearend_scale'])))
    rows.sort()
    return rows


def load(path, dtype=np.float32):
    x, sr = sf.read(str(path), dtype=dtype)
    assert sr == SR, '%s is %d Hz, expected %d' % (path, sr, SR)
    if x.ndim > 1:
        x = x[:, 0]
    if len(x) < LEN:
        x = np.pad(x, (0, LEN - len(x)))
    return x[:LEN]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', help='corpus root (or set AEC_DATASET_DIR)')
    ap.add_argument('--out', default='data', help='where to write the arrays')
    ap.add_argument('--limit', type=int, default=0,
                    help='only pack the first N fileids (for a smoke test)')
    args = ap.parse_args()

    root = resolve_dataset(args.dataset)
    ex = root / 'extract'
    rows = read_meta(root)
    if args.limit:
        rows = rows[:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print('dataset : %s' % root)
    print('files   : %d' % len(rows))

    n = len(rows)
    mic = np.lib.format.open_memmap(out / 'mic.npy', mode='w+',
                                    dtype=np.float32, shape=(n, LEN))
    ref = np.lib.format.open_memmap(out / 'ref.npy', mode='w+',
                                    dtype=np.float32, shape=(n, LEN))
    tgt = np.lib.format.open_memmap(out / 'tgt.npy', mode='w+',
                                    dtype=np.float32, shape=(n, LEN))
    scale, split, ids = {}, {}, {}
    for k, (fid, sp, sc) in enumerate(rows):
        mic[k] = load(ex / 'synthetic_nearend_mic' / ('nearend_mic_fileid_%d.flac' % fid))
        ref[k] = load(ex / 'synthetic_farend' / ('farend_speech_fileid_%d.flac' % fid))
        near = load(ex / 'synthetic_nearend_speech' / ('nearend_speech_fileid_%d.flac' % fid))
        tgt[k] = near * sc
        scale[str(fid)] = sc
        split[str(fid)] = sp
        ids[str(fid)] = k
        if (k + 1) % 500 == 0:
            print('  ...%d/%d' % (k + 1, n), flush=True)
    for a in (mic, ref, tgt):
        a.flush()
    del mic, ref, tgt

    (out / 'meta.json').write_text(json.dumps(
        {'scale': scale, 'split': split, 'ids': ids}, indent=0))
    counts = {s: list(split.values()).count(s) for s in set(split.values())}
    print('wrote %s  (split %s)' % (out, counts))


if __name__ == '__main__':
    main()

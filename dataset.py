"""Reading the packed dataset produced by ``prepare_dataset.py``."""
import json
import os
from pathlib import Path


def resolve_data(arg):
    root = Path(arg or os.environ.get('AEC_DATA_DIR', 'data')).expanduser()
    if not (root / 'mic.npy').is_file():
        raise SystemExit(
            'no mic.npy in %s.\n'
            '  run prepare_dataset.py first, or pass --data / set '
            'AEC_DATA_DIR.' % root)
    return root


def load_meta(root):
    return json.loads((root / 'meta.json').read_text())


def split_rows(meta, which):
    """Row indices of the packed arrays belonging to one split, sorted.

    ``ids`` maps each corpus file id to its row; datasets packed before that
    field existed used the file id directly as the row index.
    """
    ids = meta.get('ids')
    out = [int(ids[fid]) if ids else int(fid)
           for fid, s in meta['split'].items() if s == which]
    return sorted(out)


def load_arrays(root):
    import numpy as np
    return (np.load(root / 'mic.npy', mmap_mode='r'),
            np.load(root / 'ref.npy', mmap_mode='r'),
            np.load(root / 'tgt.npy', mmap_mode='r'))

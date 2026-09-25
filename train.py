#!/usr/bin/env python3
"""Train the AEC front-end in two stages.

Stage 1 (--stage tde) trains only the delay estimator, on its own supervised
loss: synthetic paths carry an exact delay label, and the SI-SDR gradient is
deliberately not shared with this stage -- sharing it makes the two objectives
pull the delay in opposite directions and the run stops improving. Stage 2
(--stage lp) freezes the estimator and trains the mask network (and the
whitener and the synthesis weight) on the output objective.

The corpus's echo paths are single-tap, so two things are mixed into every
batch to widen the distribution:

  * **synthetic paths with a known delay** (``mic = near + g * ref shifted by
    d``), which carry the delay labels stage 1 needs. The fraction anneals to
    zero over ``--syn-anneal`` epochs so the net finishes on the real
    distribution.
  * **an optional second corpus** (``--extra-data``), mixed in per sample at
    ``--extra-frac``. Training the mask on one corpus alone teaches it that
    corpus's statistics; a second, differently-built corpus keeps it honest.

Usage:
    python train.py --stage tde --out weights/stage1.pt
    python train.py --stage lp --init weights/stage1.pt --out weights/aec_lp.pt
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from dataset import load_arrays, load_meta, resolve_data, split_rows
from model import AecFrontend, si_sdr
from stft import SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', help='packed dataset dir (or set AEC_DATA_DIR)')
    ap.add_argument('--extra-data',
                    help='optional second packed dataset, mixed in per sample')
    ap.add_argument('--extra-frac', type=float, default=0.3,
                    help='fraction of each batch drawn from --extra-data')
    ap.add_argument('--out', default='weights/aec_lp.pt')
    ap.add_argument('--stage', default='lp', choices=['tde', 'lp'],
                    help="'tde' trains only the delay estimator; 'lp' freezes "
                         "it and trains the mask network")
    ap.add_argument('--init', help='checkpoint to initialise from (the other '
                                   "stage's output)")
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--steps-per-epoch', type=int, default=500)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--crop', type=float, default=3.0, help='seconds per sample')
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--hid', type=int, default=96)
    ap.add_argument('--bands', type=int, default=16)
    ap.add_argument('--max-shift', type=int, default=10240,
                    help='largest synthetic echo delay, in samples')
    ap.add_argument('--syn-frac', type=float, default=0.5,
                    help='fraction of each batch that is a labelled synthetic path')
    ap.add_argument('--syn-anneal', type=int, default=40,
                    help='anneal --syn-frac to zero over this many epochs')
    ap.add_argument('--tde-weight', type=float, default=1.0,
                    help='weight of the delay loss, in units of 1000 samples')
    ap.add_argument('--tde-mode', default='stream', choices=['stream', 'global'],
                    help="delay handling at stage 2: 'stream' acquires once and "
                         "tracks per frame (the deployable mode, and what the "
                         "mask is trained against); 'global' uses one estimate "
                         "per utterance. Stage 1 always uses 'global' -- the "
                         "tracker has no trainable parameters and no gradient.")
    ap.add_argument('--farend-gate', type=float, default=0.005,
                    help='how strongly quiet reference bins force the mask to 1. '
                         'An inference-time knob, recorded in the checkpoint; '
                         'smaller values suppress more echo and attenuate more '
                         'near end.')
    ap.add_argument('--mic-feat', action=argparse.BooleanOptionalAction,
                    default=True,
                    help='feed the microphone band energies to the mask net')
    ap.add_argument('--val-files', type=int, default=60)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    root = resolve_data(args.data)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = args.device if torch.cuda.is_available() else 'cpu'
    print('device : %s' % dev, flush=True)
    print('data   : %s' % root, flush=True)

    mic, ref, tgt = load_arrays(root)
    meta = load_meta(root)
    ids = {sp: split_rows(meta, sp) for sp in ('train', 'test')}
    ex = None
    if args.extra_data:
        eroot = resolve_data(args.extra_data)
        emic, eref, etgt = load_arrays(eroot)
        eids = split_rows(load_meta(eroot), 'train')
        ex = (emic, eref, etgt, eids)
        print('extra  : %s (%d train rows) at %.2f'
              % (eroot, len(eids), args.extra_frac), flush=True)

    rng = np.random.default_rng(args.seed)
    tr_ids = ids['train']
    val_ids = tr_ids[:args.val_files]
    tr_ids = tr_ids[args.val_files:]
    crop = int(args.crop * SR)
    print('split  : %d train / %d val / %d test'
          % (len(tr_ids), len(val_ids), len(ids['test'])), flush=True)

    model = AecFrontend(hid=args.hid, bands=args.bands,
                        max_delay=args.max_shift,
                        use_mic_feat=args.mic_feat).to(dev)
    model.farend_gate = args.farend_gate
    model.tde_mode = 'global' if args.stage == 'tde' else args.tde_mode
    print('params : %.3f M' % (sum(p.numel() for p in model.parameters()) / 1e6),
          flush=True)

    if args.init:
        ck0 = torch.load(args.init, map_location='cpu', weights_only=False)
        cur = model.state_dict()
        # load only what matches: the point of the stage boundary is that the
        # two halves do not share weights, and a strict load would reject any
        # shape or presence difference
        keep = {k: v for k, v in ck0['model'].items()
                if k in cur and cur[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)
        print('init   : %d tensors from %s (%d skipped)'
              % (len(keep), args.init, len(ck0['model']) - len(keep)), flush=True)

    tde_params = [p for n, p in model.named_parameters() if n.startswith('tde.')]
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith('tde.')]
    if args.stage == 'tde':
        for p in other_params:
            p.requires_grad_(False)
        print('stage tde: %d delay tensors trainable, %d frozen elsewhere'
              % (len(tde_params), len(other_params)), flush=True)
    else:
        for p in tde_params:
            p.requires_grad_(False)
        print('stage lp: delay estimator frozen (%d tensors), %d trainable '
              'elsewhere' % (len(tde_params), len(other_params)), flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, 'nothing to train'
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * args.steps_per_epoch,
        pct_start=0.1)

    def batch(ids_, n, syn_frac):
        """Returns mic, ref, target, delay label, synthetic flag."""
        idx = rng.choice(ids_, size=n, replace=False)
        m = np.empty((n, crop), np.float32)
        f = np.empty((n, crop), np.float32)
        t = np.empty((n, crop), np.float32)
        label = np.zeros(n, np.float32)
        synth = np.zeros(n, np.float32)
        for k, j in enumerate(idx):
            if ex is not None and rng.random() < args.extra_frac:
                src_m, src_f, src_t, src_ids = ex
                j = src_ids[rng.integers(0, len(src_ids))]
            else:
                src_m, src_f, src_t = mic, ref, tgt
            s = rng.integers(0, src_m.shape[1] - crop)
            mc = src_m[j, s:s + crop]
            fc = src_f[j, s:s + crop]
            tc = src_t[j, s:s + crop]
            if syn_frac and rng.random() < syn_frac:
                d = int(rng.integers(0, args.max_shift + 1))
                g = rng.uniform(0.1, 0.7)
                mc = tc + np.concatenate(
                    [np.zeros(d, np.float32), (g * fc)[:crop - d]])
                label[k] = d
                synth[k] = 1.0
            m[k] = mc
            f[k] = fc
            t[k] = tc
        to = torch.from_numpy
        return (to(m).to(dev), to(f).to(dev), to(t).to(dev),
                to(label).to(dev), to(synth).to(dev))

    def silence_penalty(y, m, f, win=1600, hop=800, thresh_db=-45.0):
        """Penalise output energy removed where the far end is silent.

        There is no echo to cancel there, so the output must pass the
        microphone through; the far-end gate already forces this in bins the
        reference leaves quiet, and this term covers the frames it does not.
        """
        _, length = y.shape
        rms_m = m.pow(2).mean(-1, keepdim=True).sqrt() + 1e-12
        pen = y.new_zeros(())
        n = 0
        for i in range(0, length - win, hop):
            fs = f[:, i:i + win].pow(2).mean(-1) / rms_m[:, 0].pow(2)
            silent = fs < 10 ** (thresh_db / 10)
            if not silent.any():
                continue
            pm = m[:, i:i + win].pow(2).sum(-1)
            po = y[:, i:i + win].pow(2).sum(-1)
            good = silent & (pm > 1e-6)
            if good.any():
                ratio = (po[good] + 1e-12) / (pm[good] + 1e-12)
                pen = pen + torch.clamp(-torch.log(ratio + 1e-12), min=0).mean()
                n += 1
        return pen / max(n, 1)

    def evaluate(ids_):
        from metrics import score
        model.eval()
        rows = []
        with torch.no_grad():
            for j in ids_:
                m = np.asarray(mic[j]); f = np.asarray(ref[j]); t = np.asarray(tgt[j])
                y, _, _ = model(torch.from_numpy(np.array(m))[None].to(dev),
                                torch.from_numpy(np.array(f))[None].to(dev),
                                length=len(m))
                rows.append(score(m, y[0].float().cpu().numpy().astype(np.float64),
                                  f, t, delay=0))
        model.train()
        return rows

    best = -1e9
    t0 = time.time()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        total = tde_total = 0.0
        syn_now = (args.syn_frac * max(0.0, 1.0 - ep / args.syn_anneal)
                   if args.syn_anneal else args.syn_frac)
        for _ in range(args.steps_per_epoch):
            m, f, t, label, synth = batch(tr_ids, args.batch, syn_now)
            y, mask, tau = model(m, f, length=crop)
            if args.stage == 'tde':
                # the delay estimator is trained only by its own supervised
                # loss; sharing the SI-SDR gradient with it makes the two
                # objectives fight and the run stops improving
                loss = torch.zeros((), device=dev)
            else:
                loss = -si_sdr(y, t).mean()
                loss = loss + 3.0 * silence_penalty(y, m, f)

            if tau is not None and tau.requires_grad and synth.sum() > 0:
                sel = synth > 0.5
                tde_loss = (tau[sel] - label[sel].unsqueeze(1)).abs().mean() / 1000.0
                loss = loss + args.tde_weight * tde_loss
                tde_total += float(tde_loss)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            opt.step()
            sched.step()
            total += float(loss)

        msg = 'ep %3d  loss %8.3f' % (ep, total / args.steps_per_epoch)
        if tde_total:
            msg += '  tde %.4f' % (tde_total / args.steps_per_epoch)
        msg += '  %5.0fs' % (time.time() - t0)

        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            rows = evaluate(val_ids)
            s = np.array([r['si_sdr'] for r in rows])
            e = np.array([r['erle'] for r in rows]); e = e[np.isfinite(e)]
            d = np.array([r['far_silence_preservation'] for r in rows])
            d = d[np.isfinite(d)]
            msg += ('  | val SI-SDR %7.3f  ERLE %6.2f  sil-keep %6.2f'
                    % (s.mean(), e.mean() if len(e) else float('nan'),
                       d.mean() if len(d) else float('nan')))
            if s.mean() > best:
                best = float(s.mean())
                torch.save({'model': model.state_dict(),
                            'config': dict(hid=args.hid, bands=args.bands,
                                           max_delay=args.max_shift,
                                           use_mic_feat=args.mic_feat),
                            'knobs': dict(tde_mode=model.tde_mode,
                                          farend_gate=args.farend_gate,
                                          warmup=352),
                            'val_si_sdr': best,
                            'args': vars(args)}, args.out)
                msg += '  *saved*'
        print(msg, flush=True)

    print('\nbest val SI-SDR: %.3f dB -> %s' % (best, args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())

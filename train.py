#!/usr/bin/env python3
"""Train the AEC front-end.

The corpus itself is single-talk-ish: the echo path in every training file is a
single tap (``mic = near_end + g * ref delayed by d``) with a short delay, which
means the trained filter would never see a realistic room. Two things are mixed
into every batch to fix that:

  * **synthetic paths with a known delay.** ``mic = near_end + g * ref shifted
    by D`` for D drawn over the full range the estimator supports. These carry
    an exact label, which is what actually teaches the delay estimator -- the
    SI-SDR gradient through the fractional delay is orders of magnitude weaker
    than through the echo filter. The fraction anneals to zero over
    ``--syn-anneal`` epochs so the net finishes on the real distribution.
  * **the corpus as-is**, which keeps the in-domain distribution.

Usage:
    python train.py --data data --out weights/aec_lp.pt
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from dataset import load_arrays, load_meta, resolve_data, split_rows
from model import AecFrontend, si_sdr
from stft import SR, sqrt_hann, stft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', help='packed dataset dir (or set AEC_DATA_DIR)')
    ap.add_argument('--out', default='weights/aec_lp.pt')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--steps-per-epoch', type=int, default=500)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--crop', type=float, default=3.0, help='seconds per sample')
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--hid', type=int, default=96)
    ap.add_argument('--bands', type=int, default=16)
    ap.add_argument('--taps', type=int, default=64)
    ap.add_argument('--max-shift', type=int, default=10240,
                    help='largest synthetic echo delay, in samples')
    ap.add_argument('--syn-frac', type=float, default=0.5,
                    help='fraction of each batch that is a labelled synthetic path')
    ap.add_argument('--syn-anneal', type=int, default=40,
                    help='anneal --syn-frac to zero over this many epochs')
    ap.add_argument('--tde-weight', type=float, default=1.0,
                    help='weight of the delay loss, in units of 1000 samples')
    ap.add_argument('--echo-weight', type=float, default=1.0,
                    help='weight of the supervised echo-estimate loss')
    ap.add_argument('--adapt-gain', type=int, default=5,
                    help='frames of smoothing for the least-squares echo gain '
                         '(0 disables). Must be set at training time, not just '
                         'at inference, or the mask learns to compensate for it.')
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

    rng = np.random.default_rng(args.seed)
    tr_ids = ids['train']
    val_ids = tr_ids[:args.val_files]
    tr_ids = tr_ids[args.val_files:]
    crop = int(args.crop * SR)
    print('split  : %d train / %d val / %d test'
          % (len(tr_ids), len(val_ids), len(ids['test'])), flush=True)

    model = AecFrontend(hid=args.hid, bands=args.bands, filt_taps=args.taps,
                        max_delay=args.max_shift, use_mic_feat=args.mic_feat).to(dev)
    model.adapt_gain = args.adapt_gain
    print('params : %.3f M' % (sum(p.numel() for p in model.parameters()) / 1e6),
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * args.steps_per_epoch,
        pct_start=0.1)

    def batch(ids_, n, syn_frac):
        """Returns mic, ref, target, delay label, synthetic flag, true echo."""
        idx = rng.choice(ids_, size=n, replace=False)
        m = np.empty((n, crop), np.float32)
        f = np.empty((n, crop), np.float32)
        t = np.empty((n, crop), np.float32)
        label = np.zeros(n, np.float32)
        synth = np.zeros(n, np.float32)
        echo = np.zeros((n, crop), np.float32)
        for k, j in enumerate(idx):
            s = rng.integers(0, mic.shape[1] - crop)
            mc = mic[j, s:s + crop]
            fc = ref[j, s:s + crop]
            tc = tgt[j, s:s + crop]
            if syn_frac and rng.random() < syn_frac:
                d = int(rng.integers(0, args.max_shift + 1))
                g = rng.uniform(0.1, 0.7)
                e = np.zeros(crop, np.float32)
                e[d:] = (g * fc)[:crop - d]
                mc = tc + e
                echo[k] = e
                label[k] = d
                synth[k] = 1.0
            m[k] = mc
            f[k] = fc
            t[k] = tc
        to = torch.from_numpy
        return (to(m).to(dev), to(f).to(dev), to(t).to(dev),
                to(label).to(dev), to(synth).to(dev), to(echo).to(dev))

    def silence_penalty(y, m, f, win=1600, hop=800, thresh_db=-45.0):
        """Penalise output energy removed where the far end is silent.

        There is no echo to cancel there, so the output must pass the
        microphone through; without this the mask closes on those frames too
        and eats the near-end.
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
        total = tde_total = echo_total = 0.0
        syn_now = (args.syn_frac * max(0.0, 1.0 - ep / args.syn_anneal)
                   if args.syn_anneal else args.syn_frac)
        for _ in range(args.steps_per_epoch):
            m, f, t, label, synth, echo = batch(tr_ids, args.batch, syn_now)
            y, mask, tau = model(m, f, length=crop)
            loss = -si_sdr(y, t).mean()
            loss = loss + 3.0 * silence_penalty(y, m, f)

            # the delay estimator is trained only by its own supervised loss;
            # sharing the SI-SDR gradient with it makes the two objectives
            # fight and the run stops improving
            if tau is not None and synth.sum() > 0:
                sel = synth > 0.5
                tde_loss = (tau[sel] - label[sel].unsqueeze(1)).abs().mean() / 1000.0
                loss = loss + args.tde_weight * tde_loss
                tde_total += float(tde_loss)

            # the echo estimate is trained directly against the known echo, for
            # the same reason: on the SI-SDR gradient alone the filter is far
            # too slow to learn, because the mask above it can always undo it
            if args.echo_weight and synth.sum() > 0:
                sel = synth > 0.5
                target = stft(echo[sel], sqrt_hann().to(dev))
                num = (model.last_acc[sel] - target).abs().mean(dim=(1, 2))
                den = target.abs().mean(dim=(1, 2)) + 1e-6
                echo_loss = (num / den).mean()
                loss = loss + args.echo_weight * echo_loss
                echo_total += float(echo_loss)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            total += float(loss)

        msg = 'ep %3d  loss %8.3f' % (ep, total / args.steps_per_epoch)
        if tde_total:
            msg += '  tde %.4f' % (tde_total / args.steps_per_epoch)
        if echo_total:
            msg += '  echo %.4f' % (echo_total / args.steps_per_epoch)
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
                                           filt_taps=args.taps,
                                           max_delay=args.max_shift,
                                           use_mic_feat=args.mic_feat,
                                           echo_gate=False),
                            'adapt_gain': args.adapt_gain,
                            'val_si_sdr': best,
                            'args': vars(args)}, args.out)
                msg += '  *saved*'
        print(msg, flush=True)

    print('\nbest val SI-SDR: %.3f dB -> %s' % (best, args.out))


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Command-line inference for the AEC front-end.

Two ways to supply the input:

  * two separate files (near-end microphone and far-end reference):

        python infer.py --mic nearend_mic.wav --ref farend_speech.wav --out out.wav

  * one multi-channel file, where the channels are read in order
    (ch0 = near-end microphone, ch1 = far-end reference, any further channels
    are ignored -- so a 3-channel 近端/远端/算法后 file can be fed straight in):

        python infer.py --input demo_3ch.wav --out out.wav

Add ``--three-channel`` to write the same channel layout back out
(近端 mic / 远端 ref / 算法后 output) instead of a mono file. Input channels
must be 16 kHz, as the model is trained at that rate; other rates are resampled
linearly, which is convenient but not high quality.

The output is time-aligned with the microphone channel: the front-end has no
algorithmic delay.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

SR = 16000


# --------------------------------------------------------------------- audio
def read_wav(path):
    """-> (samples, rate) as float32 in [-1, 1], shape (channels, samples)."""
    import soundfile as sf
    x, sr = sf.read(str(path), dtype='float32', always_2d=True)
    return x.T, sr


def write_wav(path, data, sr=SR):
    """data: (channels, samples) float in [-1, 1]."""
    import soundfile as sf
    sf.write(str(path), np.clip(data.T, -1.0, 1.0), sr, subtype='PCM_16')


def resample(x, sr_in, sr_out=SR):
    if sr_in == sr_out:
        return x
    n_out = int(round(x.shape[-1] * sr_out / sr_in))
    src = np.arange(x.shape[-1], dtype=np.float64)
    dst = np.linspace(0, x.shape[-1] - 1, n_out)
    return np.stack([np.interp(dst, src, ch).astype(np.float32) for ch in x])


# ------------------------------------------------------------------- models
class TorchRunner:
    """Runs the PyTorch checkpoint."""

    def __init__(self, ckpt, device='cpu'):
        import torch
        from model import AecFrontend
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        self.model = AecFrontend(**ck['config']).to(device).eval()
        self.model.load_state_dict(ck['model'])
        self.model.adapt_gain = ck.get('adapt_gain', 0)
        self.torch = torch
        self.device = device

    def infer(self, mic, ref):
        t = self.torch
        with t.no_grad():
            y, _, tau = self.model(t.from_numpy(mic)[None].to(self.device),
                                   t.from_numpy(ref)[None].to(self.device),
                                   length=len(mic))
        return y[0].cpu().numpy(), float(tau.item())


class OnnxRunner:
    """Runs the exported ONNX graph. Needs a fixed segment length."""

    def __init__(self, path, segment):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        self.segment = segment

    def infer(self, mic, ref):
        o = self.sess.run(None, {'mic': mic[None], 'ref': ref[None]})
        return o[0][0], float(o[1].reshape(-1)[0])


# ------------------------------------------------------------------ segment
def crossfade_window(n):
    """Hann-like weights, strictly positive so the overlap-add never divides
    by zero at the very first and last sample."""
    i = np.arange(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2 * np.pi * (i + 1) / (n + 1))


def run_segmented(runner, mic, ref, segment, verbose=True):
    """Process a long recording in fixed-length segments with a 50 % overlap.

    The graph has static shapes, so longer input has to be cut up. The overlap
    and the raised-cosine weighting hide the fact that the recurrent network's
    context is reset at every boundary.
    """
    n = len(mic)
    if n <= segment:
        padded_m = np.pad(mic, (0, segment - n))
        padded_r = np.pad(ref, (0, segment - n))
        y, tau = runner.infer(padded_m, padded_r)
        if verbose:
            print('  one segment, estimated delay %.0f samples' % tau)
        return y[:n]

    hop = segment // 2
    w = crossfade_window(segment)
    num = np.zeros(n + segment, np.float64)
    den = np.zeros(n + segment, np.float64)
    starts = list(range(0, n - segment, hop)) + [n - segment]
    taus = []
    for s0 in starts:
        y, tau = runner.infer(np.ascontiguousarray(mic[s0:s0 + segment], np.float32),
                              np.ascontiguousarray(ref[s0:s0 + segment], np.float32))
        taus.append(tau)
        num[s0:s0 + segment] += w * y
        den[s0:s0 + segment] += w
    if verbose:
        print('  %d segments of %.1f s, estimated delay %.0f samples'
              % (len(starts), segment / SR, float(np.median(taus))))
    return (num[:n] / np.maximum(den[:n], 1e-9)).astype(np.float32)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description='Neural acoustic echo cancellation front-end (16 kHz).')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--input', help='multi-channel wav: ch0 = near-end mic, '
                                     'ch1 = far-end reference')
    src.add_argument('--mic', help='near-end microphone wav')
    ap.add_argument('--ref', help='far-end reference wav (required with --mic)')
    ap.add_argument('--out', default='aec_out.wav', help='output wav path')
    ap.add_argument('--three-channel', action='store_true',
                    help='write 3 channels (mic / ref / output) instead of mono')
    model = ap.add_mutually_exclusive_group()
    model.add_argument('--onnx', help='ONNX model (default)')
    model.add_argument('--ckpt', help='PyTorch checkpoint instead of ONNX')
    ap.add_argument('--device', default='cpu', help='torch device for --ckpt')
    ap.add_argument('--segment', type=int, default=48000,
                    help='samples per ONNX inference call (must match export)')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    if args.input:
        x, sr = read_wav(args.input)
        if x.shape[0] < 2:
            ap.error('--input needs at least 2 channels (mic, ref); got %d' % x.shape[0])
        x = resample(x, sr)
        mic, ref = x[0], x[1]
        print('input : %s  %d ch @ %d Hz -> %.2f s'
              % (args.input, x.shape[0], sr, x.shape[-1] / SR))
    else:
        if not args.ref:
            ap.error('--ref is required together with --mic')
        m, sr_m = read_wav(args.mic)
        r, sr_r = read_wav(args.ref)
        mic = resample(m, sr_m)[0]
        ref = resample(r, sr_r)[0]
        print('input : %s + %s  @ %d/%d Hz -> %.2f s'
              % (args.mic, args.ref, sr_m, sr_r, len(mic) / SR))

    if len(mic) != len(ref):
        n = min(len(mic), len(ref))
        print('  note: lengths differ, truncating to %d samples' % n)
        mic, ref = mic[:n], ref[:n]

    if args.ckpt:
        ckpt = args.ckpt
    else:
        here = Path(__file__).resolve().parent
        ckpt = str(here / 'weights' / 'aec_lp.onnx')
        if not args.onnx and not Path(ckpt).is_file():
            ap.error('no ONNX model found at %s; run export_onnx.py or pass '
                     '--onnx/--ckpt' % ckpt)

    if args.ckpt:
        runner = TorchRunner(args.ckpt, args.device)
        kind = 'torch %s' % args.ckpt
    else:
        runner = OnnxRunner(args.onnx or ckpt, args.segment)
        kind = 'onnx %s' % (args.onnx or ckpt)
    print('model : %s' % kind)

    out = run_segmented(runner, mic.astype(np.float32), ref.astype(np.float32),
                        args.segment, verbose=not args.quiet)

    data = np.stack([mic, ref, out]) if args.three_channel else out[None]
    write_wav(args.out, data, SR)
    rms = lambda v: 20 * np.log10(float(np.sqrt(np.mean(v.astype(np.float64) ** 2))) + 1e-20)
    print('output: %s  %d ch  %.2f s   mic %.1f dBFS -> out %.1f dBFS'
          % (args.out, data.shape[0], len(out) / SR, rms(mic), rms(out)))


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Acceptance test: the C engine must reproduce the ONNX graph.

Both paths are driven end to end on the same audio and compared sample by
sample. The default criterion is one 16-bit LSB, which is also the quantisation
step of the WAV the tool writes, so anything under it is inaudible by
construction. For reference, the two reference implementations already differ
by about that much on their own: ONNX Runtime and PyTorch agree only to ~9e-6
here, because a hand-written FFT and pocketfft round differently.

    python c/test_equiv.py                     # the bundled demo pair
    python c/test_equiv.py --mic a.wav --ref b.wav
    python c/test_equiv.py --tolerance 2       # in 16-bit LSBs

Note that `--segment` must match the segment length the ONNX model was exported
with -- the graph has static shapes, so a different length cannot be fed to it
at all. (The C engine has no such constraint.) To test another segment length,
export a matching model first:

    python export_onnx.py --segment 24000 --out /tmp/aec_lp_s24000.onnx
    python c/test_equiv.py --segment 24000 --onnx /tmp/aec_lp_s24000.onnx

Needs onnxruntime (in requirements.txt) and a built ./c/aec_infer.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

LSB = 1.0 / 32768.0          # the quantisation step of 16-bit PCM
SR = 16000


def run_c(binary, mic, ref, segment, out_f32, tau_out, model):
    cmd = [str(binary), '--mic', str(mic), '--ref', str(ref),
           '--out', str(out_f32) + '.wav', '--f32-out', str(out_f32),
           '--tau-out', str(tau_out), '--verbose', '--model', str(model)]
    if segment:
        cmd += ['--segment', str(segment)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise SystemExit('aec_infer failed (exit %d)' % r.returncode)
    return r.stdout


def erle(mic, out, ref, win=1600):
    """metrics.py's ERLE: the mean suppression over 100 ms blocks whose far end
    is active."""
    n = min(len(mic), len(out), len(ref))
    ref_db = 10 * np.log10(np.mean(ref[:n].astype(np.float64) ** 2) + 1e-20)
    acc, cnt = 0.0, 0
    for i in range(0, n - win, win // 2):
        rb = 10 * np.log10(np.mean(ref[i:i + win].astype(np.float64) ** 2) + 1e-20)
        mb = 10 * np.log10(np.mean(mic[i:i + win].astype(np.float64) ** 2) + 1e-20)
        if rb < ref_db - 35 or mb < -80:
            continue
        acc += mb - 10 * np.log10(np.mean(out[i:i + win].astype(np.float64) ** 2) + 1e-20)
        cnt += 1
    return acc / cnt if cnt else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--binary', default=str(HERE / 'aec_infer'))
    ap.add_argument('--model', default=str(ROOT / 'weights' / 'aec_lp.bin'))
    ap.add_argument('--onnx', default=str(ROOT / 'weights' / 'aec_lp.onnx'))
    ap.add_argument('--mic', default=str(ROOT / 'docs' / 'audio' / 'nearend_mic.wav'))
    ap.add_argument('--ref', default=str(ROOT / 'docs' / 'audio' / 'farend_speech.wav'))
    ap.add_argument('--segment', type=int, default=0,
                    help='0 = whatever the binary defaults to (48000)')
    ap.add_argument('--tolerance', type=float, default=1.0,
                    help='pass threshold, in 16-bit LSBs (default 1.0)')
    ap.add_argument('--out-f32', default=str(ROOT / 'tmp' / 'aec_c_out.f32'))
    args = ap.parse_args()

    import soundfile as sf
    from infer import OnnxRunner, run_segmented

    mic, sr = sf.read(args.mic, dtype='float32', always_2d=True)
    ref, _ = sf.read(args.ref, dtype='float32', always_2d=True)
    assert sr == SR and mic.shape[1] == 1 == ref.shape[1], \
        'need 16 kHz mono for this test'
    mic = mic[:, 0].copy()
    ref = ref[:len(mic), 0].copy()

    out_path = Path(args.out_f32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tau_path = out_path.with_suffix('.tau')
    log = run_c(Path(args.binary), Path(args.mic), Path(args.ref),
                args.segment, out_path, tau_path, Path(args.model))
    got = np.fromfile(out_path, dtype=np.float32)
    c_taus = np.fromfile(tau_path, dtype=np.float32)

    seg = args.segment or 48000
    runner = OnnxRunner(args.onnx, seg)
    o_taus = []
    orig_infer = runner.infer

    def spy(m, r):
        y, t = orig_infer(m, r)
        o_taus.append(t)
        return y, t
    runner.infer = spy
    want = run_segmented(runner, mic, ref, seg, verbose=False)

    print(log.rstrip())
    print('\ncomparing %d samples (%.2f s) against the ONNX graph'
          % (len(got), len(got) / SR))

    if len(got) != len(mic):
        raise SystemExit('C output has %d samples, input has %d'
                         % (len(got), len(mic)))
    diff = np.abs(got - want)
    c_erle = erle(mic, got, ref)
    o_erle = erle(mic, want, ref)

    print('\n  %-32s %s' % ('max |C - ONNX|', '%.4e' % diff.max()))
    print('  %-32s %.3f  (tolerance %.3f)' % ('in 16-bit LSBs',
                                              diff.max() / LSB, args.tolerance))
    print('  %-32s %.5f' % ('output rms', float(np.sqrt((want ** 2).mean()))))
    print('  %-32s %.1f dB' % ('difference vs output rms',
                               20 * np.log10(diff.max() /
                               float(np.sqrt((want ** 2).mean())) + 1e-30)))
    print('  %-32s %.2f dB   (ONNX %.2f dB)' % ('ERLE, C', c_erle, o_erle))

    dt = np.array(o_taus) - c_taus[:len(o_taus)]
    print('  %-32s max %.4f samples over %d segments'
          % ('delay estimate, C vs ONNX', np.abs(dt).max(), len(o_taus)))

    if diff.max() / LSB > args.tolerance:
        # The delay estimate is the one output of this model that is not
        # numerically robust. On a segment where it sits on a shallow optimum,
        # two implementations of the same arithmetic disagree in its third
        # decimal place, and 0.06 samples of delay is already ~1e-4 of
        # waveform. So say which segments moved, rather than leaving the
        # reader to guess.
        print('\n  segments where the delay estimate moved by more than 1e-3:')
        any_moved = False
        for i, d in enumerate(dt):
            if abs(d) > 1e-3:
                any_moved = True
                print('    segment %2d (samples %7d..%-7d): C %.4f  ONNX %.4f  '
                      'diff %+.4f' % (i, i * seg // 2, i * seg // 2 + seg,
                                      c_taus[i], o_taus[i], -d))
        if any_moved:
            print('\n  A different fractional delay shifts the aligned reference, so a\n'
                  '  sub-sample disagreement here explains the waveform difference:\n'
                  '  the estimator, not the engine, is the limit. Re-run with\n'
                  '  --tolerance to accept it, or use the default segment length.')
        else:
            print('    (none -- this looks like a real discrepancy in the engine)')

    ok = diff.max() / LSB <= args.tolerance
    print('\n%s: max difference %.3f LSB against a tolerance of %.3f LSB'
          % ('PASS' if ok else 'FAIL', diff.max() / LSB, args.tolerance))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

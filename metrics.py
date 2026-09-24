"""Evaluation metrics.

The primary number is SI-SDR against the clean near-end speech, because it is
the only single number that punishes both failure modes:

  * echo left in      -> the estimate drifts away from the near-end
  * near-end removed  -> the estimate also drifts away from the near-end

ERLE alone is gameable (outputting silence gives an infinite value), so it is
reported only as a diagnostic, and silence preservation is reported alongside
it to catch the case where the model simply attenuates everything.
"""
import numpy as np

SR = 16000
BLOCK = 1600          # 100 ms analysis block
HOP = 800


def db(x):
    x = np.asarray(x, np.float64)
    return 10 * np.log10(np.mean(x ** 2) + 1e-20)


def si_sdr(est, ref, eps=1e-8, floor_db=-60.0):
    """Scale-invariant SDR in dB.

    A silent estimate makes both the target and the residual vanish, which
    would score 0 dB -- better than an honest pass-through. That is an eps
    artefact rather than a result, so silence is pinned to ``floor_db``.
    """
    est = np.asarray(est, np.float64)
    ref = np.asarray(ref, np.float64)
    if np.sqrt(np.mean(est ** 2)) < 1e-5:
        return floor_db
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = float(np.dot(est, ref) / (np.dot(ref, ref) + eps))
    target = alpha * ref
    noise = est - target
    val = 10 * np.log10((np.sum(target ** 2) + eps) / (np.sum(noise ** 2) + eps))
    return float(max(val, floor_db))


def erle(mic, out, ref, win=BLOCK, hop=HOP, eps=1e-12, delay=0):
    """Echo return loss enhancement, averaged over far-end-active blocks.

    ``delay`` compensates a known algorithmic delay in ``out`` relative to
    ``mic``; this model has none, so the default is 0.
    """
    m = mic[delay:] if delay else np.asarray(mic, np.float64)
    o = out[:-delay] if delay else np.asarray(out, np.float64)
    r = ref[:len(m)]
    vals = []
    for i in range(0, len(m) - win, hop):
        if np.sum(r[i:i + win] ** 2) > 1e-6 and np.sum(m[i:i + win] ** 2) > 1e-6:
            vals.append(10 * np.log10(np.sum(m[i:i + win] ** 2) /
                                      max(np.sum(o[i:i + win] ** 2), eps)))
    return float(np.mean(vals)) if vals else float('nan')


def far_end_silence_preservation(mic, out, ref, win=BLOCK, hop=HOP,
                                 thresh_db=-45.0, eps=1e-12, delay=0):
    """How much of the microphone survives where there is no echo to cancel.

    With the far end silent the output should be a copy of the microphone, so
    the ratio is 0 dB when it is preserved and negative when the near-end is
    being eaten. Only blocks with an active near-end are counted, so the number
    is defined over a subset of files -- those with no such block return NaN.
    """
    m = mic[delay:] if delay else np.asarray(mic, np.float64)
    o = out[:-delay] if delay else np.asarray(out, np.float64)
    f = np.asarray(ref, np.float64)[:len(m)]
    rms_m = np.sqrt(np.mean(m ** 2)) + eps
    vals = []
    for i in range(0, len(m) - win, hop):
        if np.mean(f[i:i + win] ** 2) / rms_m ** 2 >= 10 ** (thresh_db / 10):
            continue                                   # far end is active here
        if np.sum(m[i:i + win] ** 2) < eps:
            continue                                   # nothing to preserve
        vals.append(10 * np.log10((np.sum(o[i:i + win] ** 2) + eps) /
                                  (np.sum(m[i:i + win] ** 2) + eps)))
    return float(np.mean(vals)) if vals else float('nan')


def score(mic, out, ref, clean=None, delay=0):
    """All metrics for one recording. ``clean`` enables SI-SDR."""
    r = {'erle': erle(mic, out, ref, delay=delay),
         'far_silence_preservation': far_end_silence_preservation(
             mic, out, ref, delay=delay)}
    if clean is not None:
        o = out[:-delay] if delay else out
        r['si_sdr'] = si_sdr(o, np.asarray(clean, np.float64)[:len(o)])
    return r


def summarise(rows):
    """Aggregate a list of ``score()`` dicts into the acceptance table."""
    def agg(key):
        v = np.array([r[key] for r in rows], np.float64)
        v = v[np.isfinite(v)]
        return float(np.mean(v)) if len(v) else float('nan')

    s = np.array([r['si_sdr'] for r in rows if 'si_sdr' in r], np.float64)
    return {'si_sdr_mean': float(s.mean()) if len(s) else float('nan'),
            'si_sdr_median': float(np.median(s)) if len(s) else float('nan'),
            'erle_mean': agg('erle'),
            'far_silence_preservation': agg('far_silence_preservation')}


# Acceptance thresholds, against the reference implementation's measured score.
THRESHOLDS = {
    'si_sdr_mean': ('>=', 6.0),
    'si_sdr_median': ('>=', 6.0),
    'erle_mean': ('>=', 6.0),
    'far_silence_preservation': ('>=', -3.0),
}

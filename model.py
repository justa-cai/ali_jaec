"""Neural acoustic-echo-cancellation front-end.

The model is a two-stage front-end operating on the microphone signal ``d``
and the far-end reference ``x``:

    1. TDE -- time-delay estimation. A GCC-PHAT cross-correlation is reduced to
       a single scalar bulk delay tau by a differentiable soft-argmax plus a
       small regression head, and the reference is then aligned into x_tau.
       Estimating the delay first is what lets the second stage be simple: it
       only ever sees the *residual* delay.

    2. LP -- linear processing. A per-frequency-bin causal FIR on the aligned
       reference produces an echo estimate y_hat, which is subtracted from the
       microphone. A per-bin least-squares gain validates the estimate against
       the actual microphone content, and a recurrent network over band
       energies predicts a spectral mask that shapes the residual.

    e(n) = mask( d(n) - alpha * y_hat(n) )

Everything runs on a 16 kHz mono signal through a 512/160 sqrt-Hann STFT
(``stft.py``). The output is time-aligned with the microphone: the analysis is
causal but the whole recording is available, so synthesis reconstructs it in
place rather than at a 352-sample offset. There is no nonlinear processing
stage -- only a linear function of the reference is ever removed.

All operations are chosen so the module exports to ONNX as-is; see
``export_onnx.py``. Keep it that way if you edit this file.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from stft import (FREQ, next_pow2, rfftfreq, sqrt_hann, stft, istft, to_full)


def time_average(x, k):
    """Moving average over the time axis of a (B, T, F) tensor.

    ``avg_pool1d`` pools the last axis, which here is frequency, so the tensor
    has to be transposed around it.
    """
    pad = k // 2
    y = F.pad(x.transpose(1, 2), (pad, pad), mode='replicate')     # (B, F, T)
    return F.avg_pool1d(y, k, stride=1).transpose(1, 2)


class SpectralBank(nn.Module):
    """Projection of a magnitude spectrum onto a small number of bands."""

    def __init__(self, freq=FREQ, bands=16):
        super().__init__()
        self.proj = nn.Linear(freq, bands)

    def forward(self, X):
        """(B, T, F) complex -> (B, T, bands) non-negative."""
        return F.relu(self.proj(X.abs()))


class DelayEstimator(nn.Module):
    """GCC-PHAT delay estimation over a bounded lag range.

    The cross-correlation is normalised by its magnitude (the PHAT weighting),
    which makes the peak independent of the loudspeaker and room magnitude
    response. The peak location is turned into a differentiable estimate in two
    steps: a soft-argmax over the lag axis gives a coarse, content-based value
    that transfers to unseen delays, and a small regression head on a few
    summary statistics of the correlation supplies a correction.

    The softmax temperature matters. The correlation curve is very flat -- the
    peak is around 1.0 while the standard deviation of the whole curve is about
    0.02 -- so a softmax over the raw values puts well under 0.1% of its mass
    on the peak and the soft-argmax collapses to zero. Standardising the curve
    first fixes it: a known delay is then recovered to within a sample over the
    whole range.
    """

    def __init__(self, max_delay=10240, temperature=20.0, hid=32):
        super().__init__()
        self.max_delay = max_delay
        self.temperature = temperature
        self.register_buffer('delays',
                             torch.arange(max_delay, -max_delay - 1, -1).float())
        self.reg = nn.Sequential(nn.Linear(6, hid), nn.Tanh(), nn.Linear(hid, 1))

    def forward(self, mic, ref):
        """(B, L) waveforms -> tau_hat (B, 1) in samples, plus the raw curve."""
        L = mic.shape[-1]
        cc = torch.fft.rfft(mic, dim=-1) * torch.conj(torch.fft.rfft(ref, dim=-1))
        cc = cc / (cc.abs() + 1e-6)
        r = torch.fft.ifft(to_full(cc), dim=-1).real          # circular correlation

        # negative lags wrap to L - k, which is exact for a circular correlation
        idx = (self.delays % L).long()
        R = r.index_select(1, idx)
        R = R / (R.std(dim=1, keepdim=True) + 1e-6)

        w = torch.softmax(self.temperature * R, dim=1)
        tau = (w * self.delays[None, :]).sum(1, keepdim=True)

        peak = R.amax(dim=1, keepdim=True)
        sharp = (w * (self.delays[None, :] - tau).pow(2)).sum(1, keepdim=True).sqrt()
        feats = torch.cat([tau / self.max_delay, peak, sharp / self.max_delay,
                           R.mean(dim=1, keepdim=True), R.std(dim=1, keepdim=True),
                           torch.ones_like(tau)], dim=1)
        return torch.clamp(tau + self.reg(feats), -self.max_delay, self.max_delay), R


class AecFrontend(nn.Module):
    """(mic, ref) waveforms -> echo-cancelled microphone waveform.

    Args:
        hid: recurrent width of the mask network.
        bands: number of frequency bands the mask network works on.
        filt_taps: length, in frames, of the per-bin FIR on the reference.
        max_delay: lag range of the delay estimator, in samples.
        echo_gate: if set, the network scales the echo estimate instead of
            masking the output (kept for experimentation).
        use_mic_feat: also feed the microphone's own band energies to the mask
            network. Without them the network cannot distinguish "echo present"
            from "the far end is loud but the microphone is silent", because the
            residual bands are then just the echo estimate itself.
    """

    def __init__(self, hid=96, bands=16, filt_taps=64, max_delay=10240,
                 echo_gate=False, use_mic_feat=True):
        super().__init__()
        self.register_buffer('window', sqrt_hann())
        self.bank = SpectralBank(FREQ, bands)
        self.tde = DelayEstimator(max_delay=max_delay)
        self.echo_gate = echo_gate
        self.use_mic_feat = use_mic_feat
        self.max_delay = max_delay
        self.bands = bands
        # inference-time knobs (not trained)
        self.adapt_gain = 0         # frames of smoothing for the LS echo gain
        self.mask_smooth = 0        # frames of averaging for the shaping mask

        # --- echo estimate: a causal per-bin FIR on the aligned reference.
        # TDE has already removed the bulk delay, so a short filter suffices.
        self.filt_taps = filt_taps
        self.filt = nn.Parameter(torch.zeros(FREQ, filt_taps, 2))
        with torch.no_grad():
            # Initialise as the identity, not as a no-op: the reference is
            # already aligned, so the echo estimate starts out as (a scaled
            # copy of) the reference itself.
            self.filt.data[:, 0, 0] = 1.0

        # --- mask network over band energies
        self.gru = nn.GRU((4 if use_mic_feat else 3) * bands, hid, batch_first=True)
        self.head = nn.Linear(hid, bands)
        if echo_gate:
            self.mask_proj = nn.Linear(bands, 2 * FREQ)
            nn.init.normal_(self.mask_proj.weight, std=1e-3)
            with torch.no_grad():
                self.mask_proj.bias[:FREQ].fill_(-1.0986)   # |g| = 0.5 initially
                self.mask_proj.bias[FREQ:].zero_()
        else:
            self.mask_proj = nn.Linear(bands, FREQ)
            nn.init.normal_(self.mask_proj.weight, std=1e-3)
            nn.init.constant_(self.mask_proj.bias, 4.0)     # sigmoid(4) ~ 0.98

    # ------------------------------------------------------------------ main
    def forward(self, mic, ref, length=None):
        tau, _ = self.tde(mic, ref)
        ref_aligned = fractional_delay(ref, tau, self.max_delay)

        Xm = stft(mic, self.window)                    # (B, T, F)
        Xr = stft(ref_aligned, self.window)

        acc = self.echo_estimate(Xr)                   # the echo estimate
        self.last_acc = acc                            # for echo supervision
        if self.adapt_gain:
            gain = self.adaptive_gain(Xm, acc, self.adapt_gain)
            if gain is not None:
                acc = gain * acc
        residual = Xm - acc

        bands = [self.bank(residual), self.bank(Xr),
                 self.bank(residual) * self.bank(Xr)]
        if self.use_mic_feat:
            bands.append(self.bank(Xm))
        h, _ = self.gru(torch.cat(bands, dim=-1))      # (B, T, hid)
        g = torch.sigmoid(self.head(h))                # (B, T, bands)

        if self.echo_gate:
            raw = self.mask_proj(g)
            phase = raw[..., FREQ:]
            mag = 2.0 * torch.sigmoid(raw[..., :FREQ])
            mask = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
            Y = Xm - self.smooth(mask, self.mask_smooth) * acc
        else:
            mask = torch.sigmoid(self.mask_proj(g))    # (B, T, F)
            Y = residual * self.smooth(mask, self.mask_smooth)

        return istft(Y, self.window, length=length), mask, tau

    # --------------------------------------------------------- components
    def echo_estimate(self, Xr):
        """sum_k filt[k] * ref shifted by k frames, per frequency bin.

        The complex product is expanded into real and imaginary parts and the
        complex tensor is only assembled at the end. Slicing, unsqueezing and
        concatenating complex tensors are not expressible in the ONNX op set,
        so the whole module keeps complex values at the outermost level only.
        """
        T = Xr.shape[1]
        xr, xi = Xr.real, Xr.imag
        wre, wim = self.filt[..., 0], self.filt[..., 1]            # (F, taps)
        acc_r = acc_i = None
        for k in range(self.filt_taps):
            if k == 0:
                sr, si = xr, xi
            else:
                shift = (0, 0, k, 0)
                sr = F.pad(xr, shift)[:, :T]
                si = F.pad(xi, shift)[:, :T]
            a = wre[:, k][None, None, :]
            b = wim[:, k][None, None, :]
            tr = sr * a - si * b
            ti = sr * b + si * a
            acc_r = tr if acc_r is None else acc_r + tr
            acc_i = ti if acc_i is None else acc_i + ti
        return torch.complex(acc_r, acc_i)

    def adaptive_gain(self, Xm, acc, k):
        """Closed-form per-bin least-squares gain on the echo estimate.

        The estimate depends only on the reference, so it is non-zero whenever
        the far end is playing -- even when the microphone contains no echo at
        all, in which case subtracting it injects the far end into the output.
        The least-squares gain

            alpha = <Xm, acc> / <acc, acc>

        measures how much of the microphone the estimate actually explains: it
        is close to one where the estimate is right and close to zero where
        there is nothing to cancel. The denominator is regularised relative to
        the bin's own power (an absolute epsilon lets the gain explode where the
        estimate vanishes) and the magnitude is clamped.

        Averages over ``k`` frames (odd, 1 disables).
        """
        if k is None or k <= 1:
            return None
        mr, mi = Xm.real, Xm.imag
        ar, ai = acc.real, acc.imag
        # <Xm, acc> and <acc, acc>, per bin
        cross_r = time_average(mr * ar + mi * ai, k)
        cross_i = time_average(mi * ar - mr * ai, k)
        power = time_average(ar * ar + ai * ai, k)
        alpha = torch.complex(cross_r, cross_i) / (power + 1e-3 * power.mean() + 1e-12)
        return alpha / (alpha.abs() + 1e-9) * alpha.abs().clamp(max=2.0)

    def smooth(self, m, k):
        """Average a mask over ``k`` frames (odd; 1 disables).

        The trained mask moves quickly -- the 99th percentile of its
        frame-to-frame change is 0.6 over 10 ms -- and that fast modulation is
        what makes suppression sound rough and punches spectral holes in the
        near-end. Light smoothing removes it at no cost in cancellation.
        """
        if k is None or k <= 1:
            return m
        if torch.is_complex(m):
            return torch.complex(time_average(m.real, k), time_average(m.imag, k))
        return time_average(m, k)


def fractional_delay(x, delay, max_delay=10240):
    """Delay ``x`` by ``delay`` samples, which may be fractional and negative.

    Implemented as a phase ramp on the zero-padded spectrum, which is exact for
    band-limited signals. The FFT size is a deterministic function of the input
    length and ``max_delay``, so the padded region is large enough for any delay
    the estimator can produce.
    """
    L = x.shape[-1]
    n = next_pow2(L + max_delay + 2)
    X = torch.fft.rfft(F.pad(x, (0, n - L)), dim=-1)
    theta = -2 * torch.pi * rfftfreq(n, device=x.device)[None, :] * delay
    shifted = X * torch.complex(torch.cos(theta), torch.sin(theta))
    return torch.fft.ifft(to_full(shifted), dim=-1).real[..., :L]


def si_sdr(pred, target, eps=1e-8):
    """Scale-invariant SDR, as a torch tensor (differentiable)."""
    t = target - target.mean(-1, keepdim=True)
    p = pred - pred.mean(-1, keepdim=True)
    a = (p * t).sum(-1, keepdim=True) / (t.pow(2).sum(-1, keepdim=True) + eps)
    num = (a * t).pow(2).sum(-1)
    den = (p - a * t).pow(2).sum(-1) + eps
    return 10 * torch.log10(num / den + eps)

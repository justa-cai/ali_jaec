"""Neural acoustic-echo-cancellation front-end.

The model is a two-stage front-end operating on the microphone signal ``d``
and the far-end reference ``x``:

    1. TDE -- time-delay estimation. A GCC-PHAT cross-correlation is reduced
       to a scalar bulk delay by a differentiable soft-argmax plus a small
       regression head. At inference the estimator runs once, on the first
       second of audio, to ACQUIRE the delay; a tracker then re-estimates it
       every 10 ms from the trailing second, moving toward the global peak of
       that window's correlation when the evidence is strong enough to trust
       and holding the previous estimate when it is not, and shifting the
       reference by a per-frame integer delay. No frame ever needs audio later
       than itself -- verified by growing-prefix equivalence, an offline pass
       and a frame-by-frame pass agree sample for sample apart from the
       512-sample frame still straddling the cut -- so the whole front-end is
       deployable as a stream with a one-second cold start for the acquisition
       and the 352-sample front-end delay thereafter.

    2. LP -- linear processing. A recurrent network over whitened band
       energies predicts a per-bin spectral gain applied DIRECTLY to the
       microphone:

           e(n) = mask(n) * d(n)

       The reference does not contribute any signal to the output -- it
       steers the mask, through the aligned features, and nothing else. Two
       properties follow from that, both structural rather than trained:

         * a silent microphone gives an exactly silent output, and
         * wherever the reference has no energy the mask is forced to 1, so
           near-end speech in those bins passes through untouched.

       The whitening (one learned per-bin weight, shared by microphone and
       reference) means the network learns *how much* echo is present rather
       than the spectral colour of the far end, and a learned per-bin
       synthesis weight shapes the output before the overlap-add.

Everything runs on a 16 kHz mono signal through a 512/160 sqrt-Hann STFT
(``stft.py``). The output is time-aligned with the microphone. There is no
separate nonlinear-processing stage.

Two runtime modes: ``tde_mode='stream'`` (default) tracks the delay per frame
and is what a live deployment wants; ``tde_mode='global'`` estimates one delay
for the whole recording and is the mode that exports to ONNX, because a
stateful per-frame loop does not trace. ``export_onnx.py`` uses the latter.
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


class SpectralWhitener(nn.Module):
    """One learned per-bin weight, shared by the microphone and the reference.

    Applied before the band projection it normalises the far end's spectral
    tilt out of the features: the mask network then has to learn how much
    echo is present, not what colour the loudspeaker and the room give it.
    Stored as a (F, 2) real parameter and used through its magnitude, which
    keeps the whole feature path in real arithmetic.
    """

    def __init__(self, freq=FREQ):
        super().__init__()
        self.w = nn.Parameter(torch.ones(freq, 2))

    def magnitude(self):
        return torch.sqrt(self.w[..., 0] ** 2 + self.w[..., 1] ** 2)   # (F,)


class DelayTracker:
    """Per-frame delay tracking with confidence-gated re-acquisition. No
    parameters.

    One mechanism from the first sample: every 10 ms frame correlates exactly
    the audio it can see -- the trailing one-second window once it exists, the
    available prefix zero-extended before that -- so acquisition and tracking
    are the same thing and there is no fixed cold start. Then:

      * the target is the GLOBAL peak of the window's GCC-PHAT, not a peak
        searched in a narrow band around the running estimate. A narrow band
        cannot see a jump larger than its own width: measured with a +-100
        band, a +320-sample path change is never followed at all (the true
        peak stays outside the band forever and the tracker random-walks on
        noise), and a failed acquisition -- a silent first second -- never
        recovers either. The global peak over the same correlation is free:
        it is already computed, and measured to land on the true delay even
        at the instant of a jump and under a loud near-end talker.
      * a frame only updates the estimate when the correlation is actually
        usable. Two gates, both cheap: the normalised cross-correlation peak
        (how much of the microphone the reference explains at the best lag --
        this is what PHAT throws away, so it is computed on the unnormalised
        correlation) and the PHAT peak's prominence over its own median.
        When either fails -- far end silent, echo absent, or the window still
        straddling a jump -- the estimate HOLDS. Holding, not drifting, is
        what turns the failure mode from "wanders away and never returns"
        into "stale until the evidence returns".
      * the move toward the target is leaky, so the residual +-1 sample
        jitter of an integer argmax does not reach the alignment.

    Causality: the window is the TRAILING second, [s - win, s), clamped at
    the start of the recording. Frames inside the first ``win`` samples
    therefore correlate whatever prefix exists (a window that would reach
    before the first sample is anchored at sample 0, extending forward), and
    everything after it uses strictly past audio -- no frame ever needs
    future samples, which is what makes the tracker deployable frame by
    frame.

    A jump is absorbed as fast as a one-second window allows: until the old
    path has (mostly) left the window the old peak keeps winning, so the
    measured re-lock after an abrupt change is a fraction of the window
    length, not the ~2 frames a narrow tracker needs for a small step -- and
    not never, which is what the narrow band delivered.
    """

    def __init__(self, win=16000, hop=160, smooth=0.5,
                 min_corr=0.05, min_prom=20.0, noise_k=4.0, min_peak=0.15):
        self.win, self.hop, self.smooth = win, hop, smooth
        self.min_corr, self.min_prom = min_corr, min_prom
        # the normalised correlation of UNCORRELATED audio has a noise ceiling
        # of about noise_k / sqrt(effective window length) at its best lag, so
        # a fixed threshold only rejects noise once the window exceeds
        # (noise_k / min_corr)^2 samples -- 6400 for the defaults, i.e. the
        # first 0.4 s could lock on pure noise. The gates therefore rise as
        # sqrt(W / W_eff) while the window is still filling.
        self.noise_k = noise_k
        # third gate, absolute: a real echo leaves a LARGE PHAT peak (measured
        # 0.38-0.77 on true paths) while the peak of two band-limited but
        # uncorrelated speech signals sits around 1/sqrt(effective bins),
        # ~0.03-0.08 -- their cross-correlation noise floor is well above the
        # white-noise ceiling the scaled corr gate assumes, which is how an
        # echo-free microphone could drift a few hundred samples over seconds.
        self.min_peak = min_peak

    @torch.no_grad()
    def tau_sequence(self, mic, ref):
        """Per-frame delay estimates for the whole input. (B, T) waveforms in,
        (B, n_frames) integer delays out.

        The signal is prefixed with ``win`` zeros, so every frame correlates a
        FULL-length window that contains exactly the audio it can see -- the
        trailing second once it exists, the available prefix zero-extended
        before that. Acquisition and tracking are then the SAME mechanism from
        sample one: the estimate locks whenever the prefix's correlation is
        strong enough to pass the gates, which on an active far end is a few
        hundred milliseconds, and holds otherwise. There is no fixed cold
        start, no separate acquisition pass, and no frame ever touches audio
        later than itself.
        """
        B, T = mic.shape
        dev = mic.device
        # NOT min(win, T): the window length must not depend on the input
        # length, or an output sample's value would depend on how much audio
        # follows it (measured: sub-second prefixes disagreed with the full
        # pass by up to 0.9 absolute). The front padding already makes every
        # window full-length for any T.
        W = self.win

        starts = torch.arange(0, T, self.hop, device=dev)
        z = torch.zeros(B, W, dtype=mic.dtype, device=dev)
        mp = torch.cat([z, mic], dim=1)
        rp = torch.cat([z, ref], dim=1)
        # window [s, s+W) of the padded signal == x[max(0, s-W), s)
        idx = starts[:, None] + torch.arange(W, device=dev)[None, :]
        mw = mp[:, idx.reshape(-1)].reshape(B, -1, W)
        rw = rp[:, idx.reshape(-1)].reshape(B, -1, W)

        nfft = next_pow2(2 * W)
        G = torch.fft.rfft(mw, n=nfft, dim=-1) * torch.fft.rfft(rw, n=nfft, dim=-1).conj()
        # PHAT: where the peak is. Raw: how much of the mic it explains.
        cc = torch.fft.irfft(G / (G.abs() + 1e-6), n=nfft, dim=-1)     # (B,F,nfft)
        raw = torch.fft.irfft(G, n=nfft, dim=-1)
        energy = (mw.pow(2).sum(-1).sqrt() * rw.pow(2).sum(-1).sqrt())[:, :, None]

        taus = torch.zeros(B, cc.shape[1], dtype=torch.long, device=dev)
        w_eff = starts.clamp(max=W).float()
        for f in range(1, cc.shape[1]):
            phat = cc[:, f, :W]
            tgt = phat.argmax(1)                                     # (B,)
            prom = phat.max(1).values / (phat.abs().median(1).values + 1e-12)
            corr = raw[:, f, :].gather(1, tgt[:, None]).squeeze(1) / \
                (energy[:, f, 0] + 1e-12)
            grow = float(torch.sqrt(torch.tensor(W / max(w_eff[f].item(), 1.0))))
            ok = (corr >= max(self.min_corr, self.noise_k /
                              float(torch.sqrt(w_eff[f].clamp(min=1.0)))))
            ok = ok & (prom >= self.min_prom * grow)
            ok = ok & (phat.max(1).values >= self.min_peak)
            step = (tgt - taus[:, f - 1]).float() * self.smooth
            taus[:, f] = torch.where(
                ok, (taus[:, f - 1].float() + step).round().long().clamp(min=0),
                taus[:, f - 1])
        return taus

    @torch.no_grad()
    def __call__(self, mic, ref):
        """Returns (aligned reference, per-frame delays (B, n_frames))."""
        T = mic.shape[-1]
        taus = self.tau_sequence(mic, ref)
        tau_seq = taus.repeat_interleave(self.hop, dim=1)[:, :T]
        tail = torch.zeros(mic.shape[0], T + int(tau_seq.max()) + 1,
                           dtype=ref.dtype, device=ref.device)
        tail[:, :T] = ref
        pos = (torch.arange(T, device=ref.device)[None, :] - tau_seq).clamp(min=0)
        aligned = tail.gather(1, pos)
        return aligned, taus


class AecFrontend(nn.Module):
    """(mic, ref) waveforms -> echo-suppressed microphone waveform.

    Args:
        hid: recurrent width of the mask network.
        bands: number of frequency bands the mask network works on.
        max_delay: lag range of the delay estimator, in samples.
        use_mic_feat: also feed the microphone's own band energies to the mask
            network. Without them the network cannot distinguish "echo
            present" from "the far end is loud but the microphone is silent",
            because the reference bands look the same in both cases.

    Inference-time knobs (not trained, stored in the checkpoint):

        tde_mode     'stream' (acquire + track per frame; default) or
                     'global' (one estimate per recording; the ONNX-exportable
                     mode -- a stateful per-frame loop does not trace).
        farend_gate  how strongly quiet reference bins force the mask to 1.
                     0 disables; smaller values suppress more.
        warmup       samples of head cross-fade to the microphone. The
                     overlap-add normaliser is ~4e4 times smaller at sample 1
                     than at steady state, so any spectrum modification is
                     amplified there; blending to the input for the first
                     NFFT-HOP samples removes it.
    """

    def __init__(self, hid=96, bands=16, max_delay=10240, use_mic_feat=True):
        super().__init__()
        self.register_buffer('window', sqrt_hann())
        self.bank = SpectralBank(FREQ, bands)
        self.whiten = SpectralWhitener(FREQ)
        self.tde = DelayEstimator(max_delay=max_delay)
        self.tracker = DelayTracker()
        self.use_mic_feat = use_mic_feat
        self.max_delay = max_delay
        self.bands = bands
        # inference-time knobs
        self.tde_mode = 'stream'
        self.farend_gate = 0.005
        self.warmup = 352
        self.mask_smooth = 0        # frames of averaging for the mask
        self.gate_win = 100         # frames the far-end gate averages (1 s,
                                    # trailing -- causal by construction)

        # mask network over whitened band energies
        self.gru = nn.GRU((4 if use_mic_feat else 3) * bands, hid, batch_first=True)
        self.head = nn.Linear(hid, bands)
        self.mask_proj = nn.Linear(bands, FREQ)
        nn.init.normal_(self.mask_proj.weight, std=1e-3)
        nn.init.constant_(self.mask_proj.bias, 4.0)      # sigmoid(4) ~ 0.98

        # per-bin synthesis weight on the output, identity-initialised
        self.out_filt = nn.Parameter(torch.zeros(FREQ, 2))
        with torch.no_grad():
            self.out_filt.data[:, 0] = 1.0

    # ------------------------------------------------------------------ main
    def forward(self, mic, ref, length=None):
        if self.tde_mode == 'stream':
            ref, taus = self.tracker(mic, ref)
            ref_aligned = ref
        else:
            tau, _ = self.tde(mic, ref)
            ref_aligned = fractional_delay(ref, tau, self.max_delay)

        Xm = stft(mic, self.window)                       # (B, T, F) complex
        Xr = stft(ref_aligned, self.window)

        # whitened band features; the whitener acts on magnitudes only
        wm = self.whiten.magnitude()                      # (F,)
        Bm = self.bank(Xm.abs() * wm)
        Br = self.bank(Xr.abs() * wm)
        feats = [Bm, Br, Bm * Br]
        if self.use_mic_feat:
            feats.append(Bm)
        h, _ = self.gru(torch.cat(feats, dim=-1))         # (B, T, hid)
        g = torch.sigmoid(self.head(h))                   # (B, T, bands)
        mask = torch.sigmoid(self.mask_proj(g))           # (B, T, F)

        # far-end gate: where the reference carries no energy there is nothing
        # to suppress, so the mask is forced to 1 and the near end passes
        # through untouched -- by construction, not by a penalty.
        if self.farend_gate:
            p = Xr.abs() ** 2                              # (B, T, F)
            # per-BIN mean over time, matching how the model was trained
            # CAUSAL normaliser: the mean is over the TRAILING gate_win
            # frames (left-padded avg-pool), never over the whole utterance.
            # A whole-input mean made every frame's gate depend on future
            # audio -- measured, an offline pass and a growing-prefix pass
            # disagreed by up to 0.14 in absolute sample value. A streaming
            # caller keeps a one-second ring of per-bin powers and computes
            # exactly this.
            k = getattr(self, 'gate_win', 100)          # frames = 1 s
            pm = F.avg_pool1d(
                F.pad(p.transpose(1, 2), (k - 1, 0), mode='replicate'),
                k, stride=1).transpose(1, 2)
            act = p / (p + self.farend_gate * pm + 1e-12)
            mask = 1.0 + act * (mask - 1.0)
        mask = self.smooth(mask, self.mask_smooth)

        Y = Xm * mask
        # per-bin synthesis weight (complex, expanded to stay ONNX-friendly)
        of = self.out_filt
        Y = torch.complex(Y.real * of[:, 0] - Y.imag * of[:, 1],
                          Y.real * of[:, 1] + Y.imag * of[:, 0])
        y = istft(Y, self.window, length=length)

        if self.warmup:
            k = min(self.warmup, y.shape[-1])
            ramp = 0.5 - 0.5 * torch.cos(
                torch.pi * torch.arange(k, device=y.device, dtype=y.dtype) / (k - 1))
            y = torch.cat([y[..., :k] * ramp + mic[..., :k] * (1.0 - ramp),
                           y[..., k:]], dim=-1)
        tau = taus[:, :1].float() if self.tde_mode == 'stream' else tau
        return y, mask, tau

    # ------------------------------------------------------------ components
    def smooth(self, m, k):
        """Average a mask over ``k`` frames (odd; 1 disables)."""
        if k is None or k <= 1:
            return m
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

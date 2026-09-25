"""Neural acoustic-echo-cancellation front-end.

The model is a two-stage front-end operating on the microphone signal ``d``
and the far-end reference ``x``:

    1. TDE -- time-delay estimation. A GCC-PHAT cross-correlation is reduced
       to a scalar bulk delay by a differentiable soft-argmax plus a small
       regression head. At inference the estimator runs once, on the first
       second of audio, to ACQUIRE the delay; a classical tracker then follows
       it frame by frame, searching only +-100 samples around the running
       estimate on a 10 ms grid and shifting the reference by a per-frame
       integer delay. A whole-recording FFT is therefore never needed, and the
       delay can drift within an utterance without the alignment being lost.

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
    """Classical per-frame tracker around a once-off acquisition. No parameters.

    The estimator (``DelayEstimator``) answers "where is the echo, anywhere in
    the supported range" -- which needs a long window and a wide search. This
    tracker answers "has it moved since the last frame", which needs neither:
    a GCC-PHAT over the trailing second, a peak search in
    ``[tau - track, tau + track]`` and a leaky update per 10 ms frame.

    A pure narrow tracker can never find a 2528-sample delay from zero, which
    is why the acquisition step exists; and a whole-recording estimate can
    never follow a delay that drifts mid-utterance, which is why the tracker
    exists. Frames inside the first ``win`` samples see a window that extends
    into the future -- unavoidable with a fixed-length window, and harmless in
    practice, since the tracker only steers features.
    """

    def __init__(self, win=16000, hop=160, track=100, smooth=0.25):
        self.win, self.hop, self.track, self.smooth = win, hop, track, smooth

    @torch.no_grad()
    def __call__(self, tde, mic, ref):
        B, T = mic.shape
        dev = mic.device
        W = min(self.win, T)
        tau0, _ = tde(mic[:, :W], ref[:, :W])
        tau0 = tau0.round().long().clamp(min=0, max=max(0, T - 1))

        starts = torch.arange(0, T, self.hop, device=dev)
        s_cl = starts.clamp(max=max(0, T - W))
        idx = s_cl[:, None] + torch.arange(W, device=dev)[None, :]
        mw = mic[:, idx.reshape(-1)].reshape(B, -1, W)
        rw = ref[:, idx.reshape(-1)].reshape(B, -1, W)

        nfft = next_pow2(2 * W)
        G = torch.fft.rfft(mw, n=nfft, dim=-1) * torch.fft.rfft(rw, n=nfft, dim=-1).conj()
        cc = torch.fft.irfft(G / (G.abs() + 1e-6), n=nfft, dim=-1)   # (B,F,nfft)
        Fq = cc.shape[1]

        offs = torch.arange(-self.track, self.track + 1, device=dev)
        taus = torch.zeros(B, Fq, dtype=torch.long, device=dev)
        taus[:, 0] = tau0[:, 0]
        for f in range(1, Fq):
            cur = taus[:, f - 1]
            cand = (cur[:, None] + offs[None, :]).clamp(min=0)        # (B,2t+1)
            best = cand.gather(1, cc[:, f, :].gather(1, cand % nfft)
                                .argmax(1, keepdim=True))
            # leaky update; clamped at 0 because an echo cannot lead the
            # reference (a negative delay would index past the buffer)
            taus[:, f] = (cur + (best[:, 0] - cur) * self.smooth) \
                .round().long().clamp(min=0)

        tau_seq = taus.repeat_interleave(self.hop, dim=1)[:, :T]
        tail = torch.zeros(B, T + int(tau_seq.max()) + 1,
                           dtype=ref.dtype, device=dev)
        tail[:, :T] = ref
        pos = (torch.arange(T, device=dev)[None, :] - tau_seq).clamp(min=0)
        aligned = tail.gather(1, pos)
        return aligned, tau_seq[:, :1].float()


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
            ref, tau = self.tracker(self.tde, mic, ref)
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
            act = p / (p + self.farend_gate * p.mean(dim=1, keepdim=True) + 1e-12)
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

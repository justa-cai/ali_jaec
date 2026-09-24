"""Analysis / synthesis front-end.

The transforms are written out explicitly (frame -> window -> FFT, and
inverse -> window -> overlap-add) rather than delegating to ``torch.stft`` /
``torch.istft``. Two reasons:

* ``torch.stft`` with ``center=False`` leaves the last ``(L - NFFT) % HOP``
  samples of the input uncovered; ``torch.istft`` can then only zero-fill them,
  which silently mutes up to 10 ms at the end of every utterance.
* the explicit form uses only operations that exist in the ONNX op set, so the
  same code can be exported and run by ONNX Runtime (see ``export_onnx.py``).

Layout is ``(batch, time, freq)`` throughout.

The parameters are fixed by the model's design:

    NFFT = 512, HOP = 160 (10 ms at 16 kHz), sqrt-Hann window

and the causal framing gives the well-known algorithmic delay

    ALG_DELAY = NFFT - HOP = 352 samples (22 ms)

relative to the microphone signal.
"""
import torch
import torch.nn.functional as F

NFFT = 512
HOP = 160
ALG_DELAY = NFFT - HOP
FREQ = NFFT // 2 + 1

SR = 16000


def sqrt_hann(n=NFFT, device=None, dtype=torch.float32):
    i = torch.arange(n, device=device, dtype=torch.float64)
    return torch.sqrt(0.5 * (1.0 - torch.cos(2 * torch.pi * i / n))).to(dtype)


def rfftfreq(n, device=None, dtype=torch.float32):
    return torch.arange(n // 2 + 1, device=device, dtype=torch.float64).to(dtype) / n


def next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _cover_tail(x, nfft=NFFT, hop=HOP):
    """Zero-pad so that every input sample falls inside at least one frame."""
    need = (-(x.shape[-1] - nfft)) % hop
    return F.pad(x, (0, need)) if need else x


def stft(x, window, hop=HOP):
    """(B, L) real -> (B, T, F) complex (one-sided)."""
    nfft = window.shape[0]
    frames = _cover_tail(x, nfft, hop).unfold(-1, nfft, hop)      # (B, T, NFFT)
    return torch.fft.rfft(frames * window, dim=-1)


def to_full(X):
    """(..., F) one-sided spectrum -> (..., NFFT) Hermitian spectrum."""
    return torch.cat([X, torch.conj(torch.flip(X[..., 1:-1], dims=[-1]))], dim=-1)


def _ola_kernel(nfft, weight, dtype, device):
    """Weight for the overlap-add transposed convolution.

    ``conv_transpose1d`` with a diagonal kernel places frame ``t`` at offset
    ``t*hop`` and sums, which is exactly overlap-add.
    """
    return torch.diag(weight).reshape(nfft, 1, nfft).to(dtype=dtype, device=device)


def istft(X, window, length=None, hop=HOP):
    """(B, T, F) complex -> (B, L) real, by explicit overlap-add.

    The synthesis normaliser is the overlap-add of the squared window, which is
    the same operation applied to a constant signal.
    """
    nfft = window.shape[0]
    frames = torch.fft.ifft(to_full(X), dim=-1).real * window     # (B, T, NFFT)
    b, t, _ = frames.shape

    y = F.conv_transpose1d(frames.transpose(1, 2),
                           _ola_kernel(nfft, torch.ones(nfft), frames.dtype, frames.device),
                           stride=hop)[:, 0]
    norm = F.conv_transpose1d(torch.ones(1, nfft, t, dtype=frames.dtype, device=frames.device),
                              _ola_kernel(nfft, window * window, frames.dtype, frames.device),
                              stride=hop)[0, 0]
    y = y / norm.clamp_min(1e-10)[None, :]

    if length is not None:
        if length <= y.shape[1]:
            y = y[:, :length]
        else:
            y = F.pad(y, (0, length - y.shape[1]))
    return y

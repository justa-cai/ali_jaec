# ali_jaec

A 16 kHz neural acoustic echo cancellation front-end: given the near-end
microphone signal `d(n)` and the far-end reference `x(n)` (what the loudspeaker
plays), it estimates and removes the echo, and returns the near-end speech.

* 0.058 M parameters (226 kB as fp32), in two networks trained in stages:
  the delay estimator first, on its own supervised loss; then, frozen, the
  mask network.
* 10 ms frames (`NFFT = 512`, `HOP = 160`, sqrt-Hann window).
* Runs offline or in fixed 3 s segments; the same graph is exported to ONNX
  and runs from Python or any ONNX Runtime host.
* Training code, evaluation code and a reproducible acceptance suite are all
  included.
* A browser demo runs the exported model client-side, with no backend:
  **https://justa-cai.github.io/ali_jaec/** — listen to the example pair, run
  the inference yourself, or try your own mic/reference files (nothing is
  uploaded). The same page also plays the *released* front-end's output on that
  clip for A/B comparison, and draws a four-row spectrogram of mic, reference,
  both outputs, computed in the page from the same WAVs the players use.

## Results

### Held-out test split (500 files, 10 s each, 16 kHz)

Scored against the *clean* near-end speech, on files never seen during
training. SI-SDR is the primary metric because it punishes both failure modes
at once -- echo left in and near-end removed.

| metric | our model | released reference | threshold |
|---|---|---|---|
| SI-SDR mean | **6.820 dB** | 5.963 dB | ≥ 6.00 |
| SI-SDR median | **7.118 dB** | — | ≥ 6.00 |
| ERLE mean | **6.747 dB** | 6.689 dB | ≥ 6.00 |
| far-end silence preservation | **+2.306 dB** | −0.525 dB | ≥ −3.00 |

The last row is the one this model is built around: +2.25 dB means the output
on far-end-silent frames is *at least* the microphone, i.e. the near end is
never attenuated there -- by construction, not by a penalty (see the far-end
gate below). Every previous version of this model scored between −0.7 and
−2.6 dB on that row.

"far-end silence preservation" is the ratio of output to microphone energy on
frames where the far end is silent: 0 dB means the microphone passes through
untouched, negative means the near-end is being eaten. It exists to catch a
model that simply attenuates everything.

### Real acoustic paths

The training corpus has single-tap echo paths, so it says little about real
rooms. These two measurements use real recordings.

Measured on the official demo pair (10 s, 2528-sample echo delay), with each
stream compared against the microphone at its own delay:

| test | our model | released reference |
|---|---|---|
| ERLE | 7.25 dB | **10.33 dB** |
| near-end preservation in bins the reference leaves quiet | **−3.2 dB** | −7.4 dB |

The trade is deliberate: the released reference removes more echo from this
pair and attenuates the near end more than twice as heavily while doing it.
The far-end gate (below) is what buys the near-end column.

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/architecture-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/architecture-light.svg">
  <img alt="Signal flow. The far-end reference x(n) and the microphone d(n) both
    pass a shared per-bin whitening weight, then a 16-band filterbank; a GRU over
    those band energies predicts a 257-bin mask, which multiplies the microphone
    spectrum directly. In parallel a GCC-PHAT delay estimator acquires the bulk
    delay once and a per-frame tracker follows it, shifting the reference so its
    bands line up with the echo. The output is e(n) = mask * d(n), shaped by a
    per-bin synthesis weight and overlap-added back to a waveform."
    src="docs/architecture-light.svg">
</picture>

The model is **two networks, trained in stages** -- the delay estimator first,
on its own supervised loss, then frozen while the mask network trains on the
output objective. Sharing the output gradient with the delay estimator makes
the two objectives pull the delay in opposite directions; separated, each
reaches its own optimum.

1. **Delay estimation, in two halves.** A GCC-PHAT cross-correlation between
   microphone and reference is reduced to a scalar delay by a soft-argmax over
   the lag axis plus a small regression head (the correlation curve is very
   flat, so it is standardised before the softmax -- without that the
   soft-argmax collapses to zero). This runs **once, on the first second**,
   anywhere in a +-1 s range, and tracks it with ONE mechanism from the first
   sample: every 10 ms the tracker correlates exactly the audio it can see --
   the trailing second once it exists, the available prefix zero-extended
   before that -- and moves toward that window's GLOBAL correlation peak when
   the evidence passes the confidence gates (normalised cross-correlation
   peak, scaled for the window's effective length; PHAT prominence; an
   absolute PHAT peak), holding the previous estimate when it does not. A
   narrow band around the running estimate cannot see a jump larger than the
   band -- measured, a +320-sample path change was never followed and a
   failed acquisition never recovered -- which is why the target is global.
   The estimate locks when the evidence arrives (measured 0.37 s on an active
   far end at a 2528-sample delay, scaling with the delay), not after a fixed
   window. The tracker has no trainable parameters, no frame needs audio
   later than itself, and no whole-recording FFT is ever needed: a stream
   with no cold start beyond the 352-sample front-end delay.

2. **Whitening.** One learned per-bin weight, shared by microphone and
   reference, normalises the far end's spectral tilt out of the features. The
   mask network then learns *how much* echo is present, not what colour the
   loudspeaker and the room give it.

3. **The mask, applied directly to the microphone.** A GRU over the whitened
   16-band energies (of the microphone, the aligned reference, their product,
   and the microphone again -- a duplicate the network was trained with and is
   kept as trained) predicts a 257-bin gain and the output is

       e(n) = mask(n) * d(n)

   The reference contributes no signal to the output -- it steers the mask and
   nothing else. Two properties follow structurally, not from training: a
   silent microphone gives an exactly silent output, and wherever the
   reference has no energy the gate below forces the mask to 1, so near-end
   speech in those bins passes through untouched whatever the weights say.

4. **The far-end gate.** Where a bin of the aligned reference carries no
   energy there is nothing to suppress, so the mask is blended toward 1 there
   (`1 + act * (mask - 1)`, with `act` the bin's share of the reference's own
   average power). This is the single biggest knob on the model's behaviour:
   opening it suppresses more echo and attenuates more near end, and it is an
   inference-time setting, recorded in the checkpoint.

5. **Synthesis.** A learned per-bin weight (identity-initialised) shapes the
   masked spectrum, and the explicit overlap-add reconstructs the waveform.
   The first `NFFT - HOP` samples are cross-faded to the microphone: the
   overlap-add normaliser is four orders of magnitude smaller at sample 1
   than at steady state, which would otherwise amplify any modification into
   an audible click there.

## Layout

```
model.py             the network
stft.py              analysis / synthesis front-end
metrics.py           SI-SDR, ERLE, silence preservation
dataset.py           reads the packed corpus
prepare_dataset.py   packs the corpus into memory-mappable arrays
train.py             training
evaluate.py          acceptance on the held-out split
export_onnx.py       ONNX export
infer.py             Python command-line inference
weights/aec_lp.pt    trained checkpoint
weights/aec_lp.onnx  the same graph, exported
docs/                browser demo on GitHub Pages, plus the architecture
                     figure (gen_architecture_svg.py regenerates it)
```

## Inference

```bash
pip install -r requirements.txt

# two files
python infer.py --mic nearend_mic.wav --ref farend_speech.wav --out out.wav

# one multi-channel file: ch0 = near-end mic, ch1 = far-end ref
python infer.py --input demo_3ch.wav --out out.wav --three-channel

# the PyTorch checkpoint instead of the ONNX graph
python infer.py --mic mic.wav --ref ref.wav --out out.wav --ckpt weights/aec_lp.pt
```

The output is time-aligned with the microphone channel: the front-end adds no
algorithmic delay. Recordings longer than 3 s are processed in overlapping
segments and cross-faded. Input must be 16 kHz; other rates are resampled
linearly, which is convenient but not high quality.


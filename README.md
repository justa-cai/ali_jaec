# ali_jaec

A 16 kHz neural acoustic echo cancellation front-end: given the near-end
microphone signal `d(n)` and the far-end reference `x(n)` (what the loudspeaker
plays), it estimates and removes the echo, and returns the near-end speech.

* 0.090 M parameters (351 kB as fp32).
* 10 ms frames (`NFFT = 512`, `HOP = 160`, sqrt-Hann window).
* Runs offline or in fixed 3 s segments; the same graph is exported to ONNX and
  driven from Python, C++ -- or from a hand-written C99 engine that needs
  neither ONNX Runtime nor a C++ compiler.
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
| SI-SDR mean | **8.830 dB** | 5.963 dB | ≥ 6.00 |
| SI-SDR median | **8.954 dB** | — | ≥ 6.00 |
| ERLE mean | **15.529 dB** | 6.689 dB | ≥ 6.00 |
| far-end silence preservation | −0.671 dB | −0.525 dB | ≥ −3.00 |

"far-end silence preservation" is the ratio of output to microphone energy on
frames where the far end is silent: 0 dB means the microphone passes through
untouched, negative means the near-end is being eaten. It exists to catch a
model that simply attenuates everything.

### Real acoustic paths

The training corpus has single-tap echo paths, so it says little about real
rooms. These two measurements use real recordings.

| test | our model | released reference |
|---|---|---|
| official demo pair (10 s, 2528-sample echo delay) — ERLE | **12.46 dB** | 10.24 dB |
| the same pair — worst case by which the output exceeds the microphone in any 100 ms window | **+0.38 dB** | −0.01 dB |
| 15 unseen real recordings, far-end single-talk — ERLE | **11.31 dB** | 6.909 dB |

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/architecture-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/architecture-light.svg">
  <img alt="Signal flow. The far-end reference x(n) runs along the top through four
    stages: a GCC-PHAT delay estimator that yields a single delay tau, an alignment
    by tau, a per-bin causal FIR echo estimate y-hat, and a least-squares gain
    alpha averaged over 50 ms. The resulting alpha*y-hat is carried down the
    right-hand spine into a subtractor fed by the near-end microphone d(n). The
    residual e0(n) taps off into a second branch -- 16 band features, a GRU
    (64 to 96), and a 257-bin spectral mask -- which returns to a multiplier,
    giving the output e(n) = mask * (d(n) - alpha*y-hat(n))."
    src="docs/architecture-light.svg">
</picture>

1. **Delay estimation.** A GCC-PHAT cross-correlation between microphone and
   reference is reduced to a single scalar delay by a soft-argmax over the lag
   axis plus a small regression head, and the reference is shifted by it. This
   step is what keeps the rest of the model simple: everything downstream only
   sees the *residual* delay, which is near zero.

   Two details matter. The correlation curve is very flat -- the peak is around
   1.0 while the standard deviation of the whole curve is about 0.02 -- so a
   softmax over the raw values puts well under 0.1 % of its mass on the peak
   and the soft-argmax collapses to zero; standardising the curve first fixes
   it. And the delay gradient is taken as a constant (`detach`): the estimator
   is trained purely by its own supervised delay loss, because sharing the
   SI-SDR gradient with it makes the two objectives fight.

2. **Echo estimate.** A causal FIR across frames, one complex weight per
   frequency bin per tap, applied to the aligned reference. It is trained not
   only through the output but directly against the known echo on synthetic
   paths -- on the SI-SDR gradient alone it barely learns, because the mask
   above it can always undo its contribution.

3. **Least-squares gain.** The estimate depends only on the reference, so it is
   non-zero whenever the far end is playing -- even when the microphone
   contains no echo at all, in which case subtracting it *injects* the far end
   into the output. The gain `alpha = <Xm, y_hat> / <y_hat, y_hat>`, averaged
   over 50 ms, is close to one where the estimate is right and close to zero
   where there is nothing to cancel.

4. **Mask.** A GRU over 16 band energies (of the residual, the reference, their
   product, and the microphone itself) predicts a per-bin spectral mask that
   shapes what is left. The microphone bands are needed to tell "echo present"
   from "far end loud but microphone silent"; without them the residual bands
   look identical in both cases.

There is no nonlinear processing stage: only a linear function of the reference
is ever removed, which is what the reference implementation does as well. Echo
paths that are nonlinear in the microphone (clipping before the reference sees
it) cannot be recovered by this class of model.

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
weights/aec_lp.bin   the same weights as a flat array, for the C engine
cpp/                 C++ / ONNX Runtime example (see cpp/README.md)
c/                   dependency-free C99 engine (see c/README.md)
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

For the C++ version see [`cpp/README.md`](cpp/README.md).

## A dependency-free C engine

ONNX Runtime is a heavy dependency for a model this small: the exported graph
is 6.75 MB on disk and **93 % of that is protobuf describing the graph**, not
weights — 7559 nodes, of which 2402 `Constant`, 1592 `Cast`, 1569 `Reshape` and
579 `Slice` exist only to spell out the FFTs and the shape juggling around them.
Profiling it on one 3 s segment puts **58 % of the time in the `DFT` operators
and 32 % in `Add`/`Slice`/`Pad`/`Mul`**, while the GRU — the only part that is
really a neural network — costs 0.7 %.

`c/` is the same computation written out by hand in C99, linking against
nothing but `libc` and `libm`. It reads a 359 kB flat weight blob instead of the
ONNX file, so there is no protobuf parser, no runtime and no C++ compiler
involved. On 10 s of audio, one core:

| implementation | wall time | RTF |
|---|---|---|
| C engine (`c/`) | **112.6 ms** | **0.0113** |
| ONNX Runtime (`cpp/`) | 537.7 ms | 0.0538 |
| PyTorch, CPU (`infer.py`) | 952.2 ms | 0.0952 |

`aec_infer` is a 47 kB binary. It is checked against the ONNX graph sample by
sample — 0.582 of a 16-bit LSB on the bundled demo pair, about the same as the
disagreement between ONNX Runtime and PyTorch themselves — and the FFT has its
own self-test against a naive DFT. See [`c/README.md`](c/README.md) for the
build, the `engine.h` API for embedding it, and the one caveat worth knowing
(the delay estimate is not numerically robust, so at very short segments the
two paths can differ by ~4 LSB).

## Training

The corpus is [AEC-Challenge](https://github.com/microsoft/AEC-Challenge)
(`nearend_mic`, `farend_speech` and the clean `nearend_speech` per file). It is
not redistributed here. Point the scripts at wherever you unpacked it, either
with `--dataset` or through the `AEC_DATASET_DIR` environment variable -- no
path is baked into the code:

```bash
python prepare_dataset.py --dataset /path/to/AEC-Challenge --out data   # ~19 GB
python train.py --data data --out weights/aec_lp.pt                     # ~1 h on one GPU
python evaluate.py --ckpt weights/aec_lp.pt --limit 500
python export_onnx.py --ckpt weights/aec_lp.pt --out weights/aec_lp.onnx
python c/export_weights.py --ckpt weights/aec_lp.pt --out weights/aec_lp.bin
```

`--data` / `AEC_DATA_DIR` selects the packed arrays for `train.py` and
`evaluate.py`.

Training mixes three things into every batch, and all three matter:

* **labelled synthetic paths** -- `mic = near_end + g * ref` delayed by a
  uniformly drawn amount up to 1 s. These give the delay estimator an exact
  target and cover delays the corpus never contains. The fraction anneals to
  zero over the first 40 epochs.
* **the corpus as-is**, which keeps the in-domain distribution. Delaying the
  *real* half of the batch as well was tried and is actively harmful: the delay
  estimator cannot be supervised there, it only makes the echo paths harder.
* **a silence penalty** on far-end-silent frames, without which the mask closes
  there too and eats the near-end.

The default configuration in `train.py` reproduces the checkpoint in
`weights/`.

## Limitations

* **Real reverberant rooms with double-talk.** On a real room response with the
  near-end talking over the far end, SI-SDR is 3.72 dB against the reference
  implementation's 8.53 dB. The gap is confined to double-talk: on
  far-end-silent frames the near-end distortion matches, and far-end leakage is
  lower in every frequency band. The cause is that the echo estimate comes from
  a single static filter for the whole recording, which can only represent an
  average room. Adding real room echoes to training was tried and made results
  worse, not better; closing this gap needs a time-varying filter.
* **Nonlinear echo paths.** Clipping that happens after the reference is
  captured cannot be modelled.
* **Delay range** is ±10240 samples (640 ms) at 16 kHz, and 16 kHz only.

## Reference

The architecture and the measured behaviour of the released JAEC front-end
(`iic/speech_jaec_aec_16k`) were used as the target for this reimplementation.
Numbers quoted above as "released reference" are that model's, measured on the
same data with the same metrics code in `metrics.py`. Both sides of the demo
pair's comparison now ship in `docs/audio/` — `aec_out.wav` from this model and
`aec_out_jaec.wav` from the released one, both aligned to `nearend_mic.wav` — so
that row can be listened to rather than taken on trust.

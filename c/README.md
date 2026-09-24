# C inference engine

The same model as `../cpp/`, but with the computation written out by hand
instead of being handed to ONNX Runtime. Plain C99, and the only libraries it
links against are `libc` and `libm`:

```
$ ldd aec_infer
        linux-vdso.so.1
        libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6
```

No ONNX Runtime, no protobuf, no C++ compiler, no build system to install.
This is the version to reach for when the target is an embedded or constrained
host, or when a 40 MB shared library is not worth a 0.09 M-parameter model.

## Why it is worth doing

The exported graph is a poor fit for a runtime built for large models:

| | `aec_lp.onnx` | `aec_lp.bin` |
|---|---|---|
| size on disk | 6.75 MB | 0.36 MB |
| of which weights | 0.44 MB | 0.36 MB |
| of which graph structure | 6.31 MB (7559 nodes) | — |

93 % of the ONNX file is protobuf describing the graph. The graph itself is
mostly bookkeeping: 2402 `Constant`, 1592 `Cast`, 1569 `Reshape` and 579
`Slice` nodes exist only to spell out the FFTs and the shape juggling around
them. Profiling ONNX Runtime on one 3 s segment puts **58 % of the time in the
`DFT` operators and 32 % in `Add`/`Slice`/`Pad`/`Mul`** — while the GRU, the
only part that is really a neural network, costs 0.7 %.

So an FFT and a GRU written directly in C, with the layout decisions made once
at export time instead of per inference, remove essentially all of the cost.

## Build

```bash
make            # builds ./aec_infer
make test       # builds and runs the FFT self-test ("354 checks, 0 failures")
make libaec.a   # the engine as a static library, for embedding
```

`CC`, `CFLAGS` and `LDLIBS` behave as usual, so cross-compiling is a matter of
`make CC=arm-none-eabi-gcc`. `-std=gnu99` also works and is worth using if the
target's libc exposes `clock_gettime` only under GNU extensions; `aec_infer`
falls back to `clock()` without it either way.

## Weights

The engine cannot read `aec_lp.onnx` — that would mean shipping a protobuf
parser. Instead, flatten the checkpoint once:

```bash
python export_weights.py --ckpt ../weights/aec_lp.pt --out ../weights/aec_lp.bin
```

This writes `magic | header | float32 arrays`: a 108-byte header carrying every
array length and the model geometry, followed by the weights in a fixed order.
Two things happen here rather than at run time:

* **Reordering.** The echo filter is stored in PyTorch as `(bin, tap, re/im)`
  and written as `(tap, bin)` with the real and imaginary parts in separate
  arrays, which is the layout the C inner loop wants. The GRU matrices are
  likewise transposed so the recurrence is a stride-1 `saxpy`.
* **Nothing derived is stored.** The sqrt-Hann window and the lag axis of the
  delay estimator are recomputed on the C side, in double precision then
  rounded to float, the same way the PyTorch code computes them.

The loader validates the magic, the version and every count against the
geometry it was compiled for, and refuses the file rather than reading garbage.

## Run

The command line is the same as `../cpp/`:

```bash
# two files: near-end microphone and far-end reference
./aec_infer --mic nearend_mic.wav --ref farend_speech.wav --out out.wav

# one multi-channel file: ch0 = near-end mic, ch1 = far-end ref
./aec_infer --input demo_3ch.wav --out out.wav --three-channel

./aec_infer --help
```

Plus three options the C++ example does not have, all for checking the engine
against the ONNX graph:

| option | effect |
|---|---|
| `--f32-out FILE` | write the raw float32 output, before 16-bit quantisation |
| `--tau-out FILE` | write one float32 per segment: its delay estimate |
| `--verbose` | print the delay estimate of every segment |
| `--bench N` | repeat the inference N times and report the real-time factor |

`--model` defaults to `weights/aec_lp.bin` relative to the working directory,
so run from the `ali_jaec` directory or pass it explicitly. Unlike the ONNX
graph, the engine's segment length is not baked in: `--segment` accepts any
value whose FFT sizes are 2/3/5-smooth.

## Embedding it

`engine.h` is the whole interface, and `make libaec.a` gives you the object
files. One engine per thread — it owns its scratch buffers, 9.2 MB of them.

```c
AecEngine* e = aec_engine_new("weights/aec_lp.bin", 0, err, sizeof err);
aec_engine_run(e, mic, ref, out, &tau);      /* exactly segment samples */
aec_engine_free(e);
```

`aec_engine_new` takes the segment length up front and precomputes the FFT
plans, the window and the OLA normalisation from it. `aec_engine_from_memory`
takes a blob you already hold, for a host that wants to link the weights into
its own binary. `aec_engine_process` handles a whole recording, cutting it into
overlapping segments and cross-fading the same way `../infer.py` does.

The output is time-aligned with the microphone: the front-end adds no
algorithmic delay.

## Verification

Two levels, both of them runnable.

**The FFT against a known answer.** `make test` checks the mixed-radix
transform against a naive O(n²) DFT at every 2/3/5-smooth size up to 500, then
checks round-trip identity and Parseval up to 250 and at the three sizes the
model actually uses (48000, 65536, 512). 354 checks, no test framework
involved.

**The whole pipeline against the ONNX graph.** `test_equiv.py` drives both
implementations over the same audio and compares them sample by sample:

```bash
python c/test_equiv.py                       # the bundled demo pair
python c/test_equiv.py --tolerance 2         # in 16-bit LSBs
```

The criterion is one 16-bit LSB, which is also the quantisation step of the WAV
the tool writes. On the bundled 10 s demo pair:

```
max |C - ONNX|                   1.7762e-05
in 16-bit LSBs                   0.582  (tolerance 1.000)
output rms                       0.09141
difference vs output rms         -74.2 dB
ERLE, C                          12.73 dB   (ONNX 12.73 dB)
delay estimate, C vs ONNX        max 0.0000 samples over 6 segments
```

For scale: the two *reference* implementations already disagree by about as
much. ONNX Runtime and PyTorch agree only to ~9e-6 on this input, because a
hand-written FFT and pocketfft round differently.

## Performance

10 s of audio (six 3 s segments) on one core of a loaded 32-core x86-64 machine,
best of five, same input for all three:

| implementation | wall time | RTF |
|---|---|---|
| this engine | **112.6 ms** | **0.0113** |
| ONNX Runtime | 537.7 ms | 0.0538 |
| PyTorch (CPU) | 952.2 ms | 0.0952 |

About 4.8× faster than ONNX Runtime and 8.5× faster than PyTorch here, and
roughly 85× faster than real time. `aec_infer` is 47 kB; the static library is
42 kB.

## Caveats

* **The delay estimate is the one output that is not numerically robust.** When
  a segment's estimate sits on a shallow optimum, two implementations of the
  same arithmetic can disagree in its third decimal place — and a 0.06-sample
  difference is already ~1e-4 of waveform. `test_equiv.py` prints which
  segments moved if the tolerance fails, because that diagnosis is otherwise
  invisible. At the default segment length the two paths agree exactly; at
  24000-sample segments they can differ by that much, which is a property of
  the estimator, not of the engine.
* **`echo_gate` models are rejected by `export_weights.py`.** The trained
  checkpoint does not use one.
* The weight blob is architecture-neutral but not byte-order neutral: it is
  written little-endian. Big-endian targets would need a byte swap.

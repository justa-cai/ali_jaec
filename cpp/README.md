# C++ inference example

A minimal ONNX Runtime command-line front-end for the exported model. No
dependency other than ONNX Runtime and a C++17 compiler; WAV reading and
writing are included (`wav_io.cpp`, PCM16 and float32).

## Build

Grab an ONNX Runtime release (1.17 or newer -- the exported graph uses the
`DFT` operator for its FFTs):

```bash
curl -LO https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
tar xzf onnxruntime-linux-x64-1.20.1.tgz
```

Then point CMake at it:

```bash
cmake -B build -DONNXRUNTIME_ROOT=$PWD/onnxruntime-linux-x64-1.20.1
cmake --build build -j
```

`ONNXRUNTIME_ROOT` can also come from the environment
(`ONNXRUNTIME_ROOT=... cmake -B build`), and if it is not set at all CMake
falls back to a system install.

## Run

```bash
# two files: near-end microphone and far-end reference
./build/aec_infer --mic nearend_mic.wav --ref farend_speech.wav --out out.wav

# one multi-channel file: ch0 = near-end mic, ch1 = far-end ref
./build/aec_infer --input demo_3ch.wav --out out.wav --three-channel
```

`--three-channel` writes mic / ref / output so the result can be compared
against the input in a single file. `--model` and `--segment` override the
model path and the segment length (which must match the value used at export
time).

The runtime library has to be findable at run time:

```bash
LD_LIBRARY_PATH=$PWD/onnxruntime-linux-x64-1.20.1/lib ./build/aec_infer --help
```

## How the segments work

The exported graph has static shapes, so a recording longer than `--segment`
(3 s by default) is cut into overlapping segments and raised-cosine weighted
back together. That hides the fact that the recurrent network's context
restarts at every boundary. `aec_infer` produces bit-identical output to
`../infer.py` on the same input, up to 16-bit quantisation.

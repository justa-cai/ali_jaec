/* Command-line acoustic echo canceller.
 *
 *   ./aec_infer --mic nearend_mic.wav --ref farend_speech.wav --out out.wav
 *   ./aec_infer --input demo_3ch.wav --out out.wav [--three-channel]
 *
 * Input channels are read in order: ch0 = near-end microphone, ch1 = far-end
 * reference. A 3-channel 近端/远端/算法后 file can therefore be fed straight in,
 * and --three-channel writes that same layout back out.
 *
 * The output is time-aligned with the microphone; the front-end adds no delay.
 *
 * There is no ONNX Runtime here: inference is the hand-written engine in
 * engine.c, so the only things this needs are a C99 compiler, libc and libm.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "engine.h"
#include "wav_io.h"

#define AEC_SR 16000
#define AEC_DEFAULT_WEIGHTS "weights/aec_lp.bin"

static void usage(void) {
  printf(
      "usage: aec_infer (--input FILE | --mic FILE --ref FILE) [options]\n"
      "\n"
      "  --input FILE     multi-channel wav: ch0 = near-end mic, ch1 = far-end ref\n"
      "  --mic FILE       near-end microphone wav\n"
      "  --ref FILE       far-end reference wav\n"
      "  --out FILE       output wav (default aec_out.wav)\n"
      "  --three-channel  write mic / ref / output instead of mono output\n"
      "  --model FILE     weight blob (default " AEC_DEFAULT_WEIGHTS ")\n"
      "  --segment N      samples per inference call (default 48000)\n"
      "  --f32-out FILE   also write the raw float32 output, for bit-level\n"
      "                   comparison against the ONNX graph\n"
      "  --tau-out FILE   write one float32 per segment: its delay estimate\n"
      "  --verbose        print the delay estimate of every segment\n"
      "  --bench N        repeat the inference N times and report the cost\n");
}

typedef struct {
  const char* input;
  const char* mic;
  const char* ref;
  const char* out;
  const char* model;
  const char* f32_out;
  const char* tau_out;
  int verbose;
  int three_channel;
  int segment;
  int bench;
} Options;

static int parse(int argc, char** argv, Options* o) {
  memset(o, 0, sizeof(*o));
  o->out = "aec_out.wav";
  o->model = AEC_DEFAULT_WEIGHTS;
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    const char* v = NULL;
#define NEED(name)                                 \
  do {                                             \
    if (i + 1 >= argc) {                           \
      fprintf(stderr, "%s needs a value\n", name); \
      return 0;                                    \
    }                                              \
    v = argv[++i];                                 \
  } while (0)
    if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
      usage();
      exit(0);
    } else if (!strcmp(a, "--input")) { NEED("--input"); o->input = v; }
    else if (!strcmp(a, "--mic")) { NEED("--mic"); o->mic = v; }
    else if (!strcmp(a, "--ref")) { NEED("--ref"); o->ref = v; }
    else if (!strcmp(a, "--out")) { NEED("--out"); o->out = v; }
    else if (!strcmp(a, "--model")) { NEED("--model"); o->model = v; }
    else if (!strcmp(a, "--f32-out")) { NEED("--f32-out"); o->f32_out = v; }
    else if (!strcmp(a, "--tau-out")) { NEED("--tau-out"); o->tau_out = v; }
    else if (!strcmp(a, "--verbose")) { o->verbose = 1; }
    else if (!strcmp(a, "--segment")) { NEED("--segment"); o->segment = atoi(v); }
    else if (!strcmp(a, "--bench")) { NEED("--bench"); o->bench = atoi(v); }
    else if (!strcmp(a, "--three-channel")) { o->three_channel = 1; }
    else {
      fprintf(stderr, "unknown option: %s\n", a);
      return 0;
    }
#undef NEED
  }
  if (!o->input && (!o->mic || !o->ref)) return 0;
  return 1;
}

/* Channel 0 of an interleaved signal. */
static float* first_channel(const float* x, size_t frames, int channels) {
  float* y = (float*)malloc(sizeof(float) * frames);
  if (!y) return NULL;
  for (size_t i = 0; i < frames; ++i) y[i] = x[i * (size_t)channels];
  return y;
}

/* Linear resampling. Fine for rounding a stray 44.1/48 kHz file up to the
 * model's rate; not a high-quality conversion. */
static float* resample(const float* x, size_t n, int sr_in, int sr_out,
                       size_t* n_out) {
  if (sr_in == sr_out || n == 0) {
    float* y = (float*)malloc(sizeof(float) * (n ? n : 1));
    if (y) memcpy(y, x, sizeof(float) * n);
    *n_out = n;
    return y;
  }
  const size_t m = (size_t)llround((double)n * sr_out / sr_in);
  float* y = (float*)malloc(sizeof(float) * (m ? m : 1));
  if (!y) return NULL;
  const double step = (double)(n - 1) / (double)(m > 1 ? m - 1 : 1);
  for (size_t i = 0; i < m; ++i) {
    const double p = (double)i * step;
    const size_t i0 = (size_t)p;
    const size_t i1 = i0 + 1 < n ? i0 + 1 : n - 1;
    const double t = p - (double)i0;
    y[i] = (float)(x[i0] * (1.0 - t) + x[i1] * t);
  }
  *n_out = m;
  return y;
}

static double rms_db(const float* v, size_t n) {
  double acc = 0.0;
  for (size_t i = 0; i < n; ++i) acc += (double)v[i] * v[i];
  return 10.0 * log10(acc / (double)(n ? n : 1) + 1e-20);
}

static double now_sec(void) {
#if defined(CLOCK_MONOTONIC)
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
#else
  return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

int main(int argc, char** argv) {
  Options opt;
  if (!parse(argc, argv, &opt)) {
    usage();
    return 2;
  }

  char err[512];
  float* mic = NULL;
  float* ref = NULL;
  size_t mic_n = 0, ref_n = 0;
  int sr_mic = AEC_SR, sr_ref = AEC_SR;

  if (opt.input) {
    float* x = NULL;
    size_t frames = 0;
    int ch = 0, sr = 0;
    if (wav_read(opt.input, &x, &frames, &ch, &sr, err, sizeof err)) {
      fprintf(stderr, "%s\n", err);
      return 1;
    }
    if (ch < 2) {
      fprintf(stderr, "%s needs at least 2 channels, has %d\n", opt.input, ch);
      free(x);
      return 1;
    }
    mic = (float*)malloc(sizeof(float) * frames);
    ref = (float*)malloc(sizeof(float) * frames);
    if (!mic || !ref) {
      fprintf(stderr, "out of memory\n");
      free(x); free(mic); free(ref);
      return 1;
    }
    for (size_t i = 0; i < frames; ++i) {
      mic[i] = x[i * (size_t)ch];
      ref[i] = x[i * (size_t)ch + 1];
    }
    free(x);
    mic_n = ref_n = frames;
    sr_mic = sr_ref = sr;
    printf("input : %s  %d ch @ %d Hz -> %.2f s\n", opt.input, ch, sr,
           (double)frames / AEC_SR);
  } else {
    float* x = NULL;
    size_t frames = 0;
    int ch = 0;
    if (wav_read(opt.mic, &x, &frames, &ch, &sr_mic, err, sizeof err)) {
      fprintf(stderr, "%s\n", err);
      return 1;
    }
    mic = ch > 1 ? first_channel(x, frames, ch) : x;
    if (ch > 1) free(x); else x = NULL;
    if (!mic) { fprintf(stderr, "out of memory\n"); return 1; }
    mic_n = frames;

    if (wav_read(opt.ref, &x, &frames, &ch, &sr_ref, err, sizeof err)) {
      fprintf(stderr, "%s\n", err);
      free(mic);
      return 1;
    }
    ref = ch > 1 ? first_channel(x, frames, ch) : x;
    if (ch > 1) free(x);
    if (!ref) { fprintf(stderr, "out of memory\n"); free(mic); return 1; }
    ref_n = frames;
    printf("input : %s + %s @ %d/%d Hz -> %.2f s\n", opt.mic, opt.ref,
           sr_mic, sr_ref,
           (double)(mic_n < ref_n ? mic_n : ref_n) / AEC_SR);
  }

  {
    size_t n = 0;
    float* p = resample(mic, mic_n, sr_mic, AEC_SR, &n);
    if (!p) { fprintf(stderr, "out of memory\n"); return 1; }
    free(mic);
    mic = p;
    mic_n = n;
  }
  {
    size_t n = 0;
    float* p = resample(ref, ref_n, sr_ref, AEC_SR, &n);
    if (!p) { fprintf(stderr, "out of memory\n"); return 1; }
    free(ref);
    ref = p;
    ref_n = n;
  }

  const size_t n = mic_n < ref_n ? mic_n : ref_n;
  if (n == 0) {
    fprintf(stderr, "empty input\n");
    free(mic); free(ref);
    return 1;
  }

  AecEngine* engine = aec_engine_new(opt.model, opt.segment, err, sizeof err);
  if (!engine) {
    fprintf(stderr,
            "%s\n"
            "  the default path is relative to the working directory; pass\n"
            "  --model, or run from the ali_jaec directory. Build the file with\n"
            "  c/export_weights.py if it does not exist yet.\n",
            err);
    free(mic); free(ref);
    return 1;
  }
  printf("model : %s (segment %d, %.1f MB of workspace)\n", opt.model,
         aec_engine_segment(engine), (double)aec_engine_memory(engine) / 1e6);

  float* out = (float*)calloc(n, sizeof(float));
  if (!out) {
    fprintf(stderr, "out of memory\n");
    aec_engine_free(engine); free(mic); free(ref);
    return 1;
  }

  /* one tau per segment; the number of segments is bounded by n/hop + 2 */
  const float seg_est = (float)aec_engine_segment(engine);
  const int max_taus = (int)(2.0 * (double)n / (double)seg_est) + 4;
  float* taus = (float*)calloc((size_t)max_taus, sizeof(float));
  if (!taus) { fprintf(stderr, "out of memory\n"); return 1; }

  float tau = 0.0f;
  const double t0 = now_sec();
  const int nseg = aec_engine_process_ex(engine, mic, ref, n, out, &tau,
                                         taus, max_taus);
  double elapsed = now_sec() - t0;
  if (opt.verbose && nseg) {
    printf("segments:");
    for (int i = 0; i < nseg; ++i) printf(" %.*f", 3, (double)taus[i]);
    printf("\n");
  }

  if (opt.bench > 1) {
    const double tb = now_sec();
    for (int i = 1; i < opt.bench; ++i)
      aec_engine_process_ex(engine, mic, ref, n, out, &tau, taus, max_taus);
    elapsed = (now_sec() - tb) / (double)(opt.bench - 1);
    printf("bench : %d runs, %.2f ms per pass, RTF %.4f "
           "(%.1f s of audio per second of CPU)\n",
           opt.bench, elapsed * 1e3, elapsed / ((double)n / AEC_SR),
           (double)n / AEC_SR / elapsed);
  } else {
    printf("time  : %.1f ms for %.2f s of audio, RTF %.4f\n", elapsed * 1e3,
           (double)n / AEC_SR, elapsed / ((double)n / AEC_SR));
  }
  printf("delay : %.0f samples (%.1f ms) estimated echo delay\n", tau,
         tau / AEC_SR * 1000.0);

  /* Report how much of the suppression is real, not just attenuation: the
   * ERLE over 100 ms blocks whose far end is active, the same definition
   * metrics.py uses. */
  {
    const int win = 1600;
    const double ref_db = rms_db(ref, n);
    double sum = 0.0;
    int count = 0;
    for (size_t i = 0; i + (size_t)win < n; i += win / 2) {
      const double rd = rms_db(ref + i, win);
      const double md = rms_db(mic + i, win);
      if (rd < ref_db - 35.0 || md < -80.0) continue;
      sum += md - rms_db(out + i, win);
      ++count;
    }
    if (count) printf("ERLE  : %.2f dB over %d far-end-active blocks\n",
                      sum / count, count);
  }

  if (opt.bench <= 1) {
    if (opt.three_channel) {
      float* data = (float*)malloc(sizeof(float) * n * 3);
      if (!data) {
        fprintf(stderr, "out of memory\n");
        aec_engine_free(engine); free(mic); free(ref); free(out);
        return 1;
      }
      for (size_t i = 0; i < n; ++i) {
        data[3 * i + 0] = mic[i];
        data[3 * i + 1] = ref[i];
        data[3 * i + 2] = out[i];
      }
      const int rc = wav_write(opt.out, data, n * 3, 3, AEC_SR, err, sizeof err);
      free(data);
      if (rc) { fprintf(stderr, "%s\n", err); return 1; }
    } else if (wav_write(opt.out, out, n, 1, AEC_SR, err, sizeof err)) {
      fprintf(stderr, "%s\n", err);
      return 1;
    }
    printf("output: %s  %d ch  %.2f s   mic %.1f dBFS -> out %.1f dBFS\n",
           opt.out, opt.three_channel ? 3 : 1, (double)n / AEC_SR,
           rms_db(mic, n), rms_db(out, n));

    if (opt.f32_out) {
      FILE* f = fopen(opt.f32_out, "wb");
      if (!f) {
        fprintf(stderr, "cannot write %s\n", opt.f32_out);
        return 1;
      }
      fwrite(out, sizeof(float), n, f);
      fclose(f);
      printf("raw   : %s  %zu float32 samples\n", opt.f32_out, n);
    }
    if (opt.tau_out) {
      FILE* f = fopen(opt.tau_out, "wb");
      if (!f) {
        fprintf(stderr, "cannot write %s\n", opt.tau_out);
        return 1;
      }
      fwrite(taus, sizeof(float), (size_t)nseg, f);
      fclose(f);
      printf("taus  : %s  %d segments\n", opt.tau_out, nseg);
    }
  }

  aec_engine_free(engine);
  free(mic);
  free(ref);
  free(out);
  free(taus);
  return 0;
}

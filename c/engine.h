/* AEC front-end inference engine: plain C99, no dependencies but libc and
 * libm.
 *
 * This is the same computation the exported ONNX graph performs -- the delay
 * estimator, the fractional-delay alignment, the per-bin causal FIR, the
 * least-squares echo gain, the band features, the GRU mask network, and the
 * inverse STFT -- written out by hand, so a host program does not need ONNX
 * Runtime, protobuf, or a C++ compiler.
 *
 * The weights come from a flat binary produced by c/export_weights.py; see
 * that script for the layout. There is no protobuf parsing here on purpose.
 *
 * The engine is not thread-safe: it owns scratch buffers, one of which
 * (`aec_engine_run`) is the whole per-segment working set. Use one engine per
 * thread.
 */
#ifndef AEC_ENGINE_H
#define AEC_ENGINE_H

#include <stddef.h>

/* Load / validate failures. */
enum {
  AEC_OK = 0,
  AEC_ERR_IO = 1,        /* the file could not be read */
  AEC_ERR_FORMAT = 2,    /* not a weight blob, or a truncated one */
  AEC_ERR_CONFIG = 3,    /* a count or geometry the engine cannot serve */
  AEC_ERR_MEMORY = 4,
  AEC_ERR_SEGMENT = 5    /* segment length the FFT plans cannot handle */
};

typedef struct AecEngine AecEngine;

/* `segment` is the number of samples per inference call; pass 0 for the
 * default of 48000 (3 s at 16 kHz). It has to match the value the weights were
 * exported for -- the model itself is delay-agnostic, but the FFT sizes and
 * the recurrent network's context are not. `segment/2` and
 * `next_pow2(segment + max_delay + 2)/2` must both be 2/3/5-smooth.
 *
 * On failure returns NULL and writes a message into `err` (if it is non-NULL
 * and `errlen` > 0). */
AecEngine* aec_engine_new(const char* weight_path, int segment,
                          char* err, int errlen);

/* Same, from a blob already in memory -- for a host that embeds the weights in
 * its own binary rather than shipping a .bin next to it. The blob is copied,
 * so the caller may free it. */
AecEngine* aec_engine_from_memory(const void* blob, size_t nbytes, int segment,
                                  char* err, int errlen);

void aec_engine_free(AecEngine* e);

/* Samples per inference call; 0 if `e` is NULL. */
int aec_engine_segment(const AecEngine* e);

/* Size of the working set, for a host that wants to know. */
size_t aec_engine_memory(const AecEngine* e);

/* One segment of exactly `aec_engine_segment(e)` samples.
 *
 * `mic`, `ref` and `out` may not overlap. `out` is time-aligned with `mic`:
 * the front-end adds no algorithmic delay. `*tau` (may be NULL) receives the
 * estimated echo delay in samples, the same value the graph returns. */
void aec_engine_run(AecEngine* e, const float* mic, const float* ref,
                    float* out, float* tau);

/* A whole recording of `n` samples, cut into overlapping segments and
 * cross-faded, which is what infer.py and the C++ example do. `out` holds `n`
 * samples; `*tau` (may be NULL) receives the median estimated delay. */
void aec_engine_process(AecEngine* e, const float* mic, const float* ref,
                        size_t n, float* out, float* tau);

/* As above, but also hands back the per-segment delay estimates, in the order
 * the segments were processed, and returns how many there were. At most
 * `max_taus` are written.
 *
 * Worth having because the delay estimate is the one output of this model that
 * is not numerically robust: on a segment where the estimate sits on a shallow
 * optimum, two implementations of the same arithmetic can disagree in its
 * third decimal place, and a 0.06-sample difference is already ~1e-4 of
 * waveform. Comparing these lists is how you tell that apart from a real
 * discrepancy. */
int aec_engine_process_ex(AecEngine* e, const float* mic, const float* ref,
                          size_t n, float* out, float* tau, float* seg_taus,
                          int max_taus);

#endif /* AEC_ENGINE_H */

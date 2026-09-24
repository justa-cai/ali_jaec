#include "engine.h"
#include "fft.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Fixed by the model design; see stft.py. */
#define AEC_NFFT 512
#define AEC_HOP 160
#define AEC_FREQ (AEC_NFFT / 2 + 1)          /* 257 one-sided bins */

#define AEC_DEFAULT_SEGMENT 48000            /* 3 s at 16 kHz */

/* The delay estimator's softmax temperature (model.py::DelayEstimator). */
#define AEC_TDE_TEMPERATURE 20.0f

/* PyTorch's nn.GRU packs the three gates in this order. */
#define AEC_GATE_R 0
#define AEC_GATE_Z 1
#define AEC_GATE_N 2

/* ------------------------------------------------------------ weight blob */

#define AEC_BLOB_MAGIC "AECLP01"
#define AEC_BLOB_VERSION 1

typedef struct {
  char magic[8];
  unsigned int version;
  unsigned int count[16];
  unsigned int hid;
  unsigned int bands;
  unsigned int filt_taps;
  unsigned int max_delay;
  unsigned int use_mic_feat;
  unsigned int echo_gate;
  unsigned int adapt_gain;
  unsigned int mask_smooth;
} BlobHeader;

/* Order must match FIELDS in export_weights.py. */
enum {
  B_FILT_RE = 0, B_FILT_IM, B_BANK_W, B_BANK_B,
  B_REG0_W, B_REG0_B, B_REG2_W, B_REG2_B,
  B_GRU_WIH, B_GRU_WHH, B_GRU_BIH, B_GRU_BHH,
  B_HEAD_W, B_HEAD_B, B_MASK_W, B_MASK_B,
  B_NFIELDS
};

struct AecEngine {
  int segment;                           /* L, samples per inference call */
  int taps, hid, in_dim, bands, max_delay, nlag;
  int use_mic_feat, adapt_gain, mask_smooth;

  unsigned int count[B_NFIELDS];
  float* blob;                           /* owned copy of the weight data */
  const float* w[B_NFIELDS];

  float window[AEC_NFFT];

  AecRfft* fft_seg;                      /* real transform of length L */
  AecRfft* fft_frac;                     /* real transform for the phase ramp */
  AecRfft* fft_win;                      /* the 512-point STFT */
  int n_frac;

  int lp;                                /* L after the tail-covering pad */
  int nframes;                           /* T */
  int lola;                              /* (T-1)*HOP + NFFT */

  /* delay estimate */
  float* seg_re; float* seg_im;          /* reused for both spectra */
  float* cc_re; float* cc_im;
  float* corr;                           /* L */
  float* lags;                           /* nlag */
  float* rs;                             /* standardised correlation, nlag */
  float* ws;                             /* softmax weights, nlag */
  float* reg_h;                          /* regression head hidden layer */
  int reg_hid;

  /* fractional delay */
  float* frac_re; float* frac_im;        /* n_frac/2+1 */
  float* frac_buf;                       /* n_frac */
  float* ref_al;                         /* L, the aligned reference */

  /* STFT domain */
  float* xm_re; float* xm_im;            /* microphone */
  float* xr_re; float* xr_im;            /* aligned reference */
  float* ac_re; float* ac_im;            /* echo estimate */
  float* rs_re; float* rs_im;            /* residual */
  float* ym_re; float* ym_im;            /* masked residual, into the ISTFT */
  float* band;                           /* in_dim band energies per frame */
  float* mag_rs; float* mag_xr; float* mag_xm;   /* magnitudes, T*FREQ each */
  float* wih_t; float* whh_t;            /* GRU weights, transposed for saxpy */
  float* gain_re; float* gain_im;        /* per-bin echo gain */
  float* hh;                             /* GRU hidden-to-hidden terms */
  float* gates; float* h0; float* h1;
  float* maskv;
  float* frame;                          /* NFFT */
  float* norm;                           /* overlap-add denominator, lola */
  float* seg_out;                        /* L */
  float* seg_mic; float* seg_ref;        /* L, for aec_engine_process */

  double* ola_num; double* ola_den;
  float* xfadew;
};

/* ------------------------------------------------------------------ utils */

static int next_pow2(int n) {
  int p = 1;
  while (p < n) p *= 2;
  return p;
}

static float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

static int max_i(int a, int b) { return a > b ? a : b; }
static int min_i(int a, int b) { return a < b ? a : b; }

static void seterr(char* err, int errlen, const char* fmt, ...) {
  if (!err || errlen <= 0) return;
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(err, (size_t)errlen, fmt, ap);
  va_end(ap);
}

/* Zero-pad so every input sample falls inside at least one frame
 * (stft.py::_cover_tail). */
static int cover_tail_need(int len) {
  int r = (len - AEC_NFFT) % AEC_HOP;
  if (r < 0) r += AEC_HOP;
  return (AEC_HOP - r) % AEC_HOP;
}

static void* xcalloc(size_t n, size_t sz) { return calloc(n ? n : 1, sz); }

static float* fvec(int n) {
  return (float*)xcalloc((size_t)max_i(n, 1), sizeof(float));
}

void aec_engine_free(AecEngine* e) {
  if (!e) return;
  aec_rfft_free(e->fft_seg);
  aec_rfft_free(e->fft_frac);
  aec_rfft_free(e->fft_win);
  free(e->blob);
  free(e->seg_re); free(e->seg_im);
  free(e->cc_re); free(e->cc_im);
  free(e->corr); free(e->lags); free(e->rs); free(e->ws); free(e->reg_h);
  free(e->frac_re); free(e->frac_im); free(e->frac_buf); free(e->ref_al);
  free(e->xm_re); free(e->xm_im); free(e->xr_re); free(e->xr_im);
  free(e->ac_re); free(e->ac_im); free(e->rs_re); free(e->rs_im);
  free(e->ym_re); free(e->ym_im);
  free(e->band); free(e->mag_rs); free(e->mag_xr); free(e->mag_xm);
  free(e->wih_t); free(e->whh_t);
  free(e->gain_re); free(e->gain_im); free(e->hh); free(e->gates); free(e->h0); free(e->h1);
  free(e->maskv); free(e->frame); free(e->norm); free(e->seg_out);
  free(e->seg_mic); free(e->seg_ref);
  free(e->ola_num); free(e->ola_den); free(e->xfadew);
  free(e);
}

/* --------------------------------------------------------------- loading */

static int parse_blob(AecEngine* e, const void* blob, size_t nbytes,
                      char* err, int errlen) {
  if (nbytes < sizeof(BlobHeader)) {
    seterr(err, errlen, "weight file is %zu bytes, smaller than the %zu-byte "
                        "header", nbytes, sizeof(BlobHeader));
    return AEC_ERR_FORMAT;
  }
  BlobHeader hd;
  memcpy(&hd, blob, sizeof(hd));
  if (memcmp(hd.magic, AEC_BLOB_MAGIC, 8) != 0) {
    seterr(err, errlen, "not an ali_jaec weight blob (bad magic) -- regenerate "
                        "it with c/export_weights.py");
    return AEC_ERR_FORMAT;
  }
  if (hd.version != AEC_BLOB_VERSION) {
    seterr(err, errlen, "weight blob version %u, this engine speaks %d",
           hd.version, AEC_BLOB_VERSION);
    return AEC_ERR_FORMAT;
  }
  if (hd.echo_gate) {
    seterr(err, errlen, "this blob came from an echo_gate model, which the C "
                        "engine does not implement");
    return AEC_ERR_CONFIG;
  }
  if (hd.hid < 1 || hd.bands < 1 || hd.filt_taps < 1 || hd.max_delay < 1) {
    seterr(err, errlen, "nonsensical geometry in the weight blob "
                        "(hid=%u bands=%u taps=%u max_delay=%u)",
           hd.hid, hd.bands, hd.filt_taps, hd.max_delay);
    return AEC_ERR_CONFIG;
  }

  e->hid = (int)hd.hid;
  e->bands = (int)hd.bands;
  e->taps = (int)hd.filt_taps;
  e->max_delay = (int)hd.max_delay;
  e->use_mic_feat = (int)hd.use_mic_feat;
  e->adapt_gain = (int)hd.adapt_gain;
  e->mask_smooth = (int)hd.mask_smooth;
  e->nlag = 2 * e->max_delay + 1;
  e->in_dim = (e->use_mic_feat ? 4 : 3) * e->bands;

  size_t total = 0;
  for (int i = 0; i < B_NFIELDS; ++i) {
    e->count[i] = hd.count[i];
    total += hd.count[i];
  }
  if (nbytes < sizeof(BlobHeader) + total * sizeof(float)) {
    seterr(err, errlen, "weight blob is truncated: the header promises %.1f kB "
                        "of weights, the file holds %.1f kB",
           total * sizeof(float) / 1e3, (nbytes - sizeof(BlobHeader)) / 1e3);
    return AEC_ERR_FORMAT;
  }

  /* The counts are the only description of the tensor shapes, so check the
   * relations the code relies on. */
  const int reg_hid = (int)(e->count[B_REG0_W] / 6);
  const int ok =
      e->count[B_FILT_RE] == (unsigned)(e->taps * AEC_FREQ) &&
      e->count[B_FILT_IM] == e->count[B_FILT_RE] &&
      e->count[B_BANK_W] == (unsigned)(e->bands * AEC_FREQ) &&
      e->count[B_BANK_B] == (unsigned)e->bands &&
      e->count[B_REG0_W] == (unsigned)(reg_hid * 6) &&
      e->count[B_REG0_B] == (unsigned)reg_hid &&
      e->count[B_REG2_W] == (unsigned)reg_hid &&
      e->count[B_REG2_B] == 1 &&
      e->count[B_GRU_WIH] == (unsigned)(3 * e->hid * e->in_dim) &&
      e->count[B_GRU_WHH] == (unsigned)(3 * e->hid * e->hid) &&
      e->count[B_GRU_BIH] == (unsigned)(3 * e->hid) &&
      e->count[B_GRU_BHH] == (unsigned)(3 * e->hid) &&
      e->count[B_HEAD_W] == (unsigned)(e->bands * e->hid) &&
      e->count[B_HEAD_B] == (unsigned)e->bands &&
      e->count[B_MASK_W] == (unsigned)(AEC_FREQ * e->bands) &&
      e->count[B_MASK_B] == (unsigned)AEC_FREQ &&
      (e->use_mic_feat == 0 || e->use_mic_feat == 1);
  if (!ok) {
    seterr(err, errlen, "weight blob shapes are inconsistent with a %d-band / "
                        "%d-hidden model (regression head %d wide)",
           e->bands, e->hid, reg_hid);
    return AEC_ERR_CONFIG;
  }
  e->reg_hid = reg_hid;

  e->blob = (float*)xcalloc(total, sizeof(float));
  if (!e->blob) {
    seterr(err, errlen, "out of memory (%.1f MB of weights)",
           total * sizeof(float) / 1e6);
    return AEC_ERR_MEMORY;
  }
  memcpy(e->blob, (const unsigned char*)blob + sizeof(BlobHeader),
         total * sizeof(float));
  size_t off = 0;
  for (int i = 0; i < B_NFIELDS; ++i) {
    e->w[i] = e->blob + off;
    off += e->count[i];
  }
  return AEC_OK;
}

static int alloc_buffers(AecEngine* e, char* err, int errlen) {
  const int L = e->segment;
  const int half = L / 2 + 1;
  const int nf = e->nframes;

  /* The delay estimate correlates at the exact segment length, so L/2 has to
   * be 2/3/5-smooth; the phase ramp uses the next power of two above
   * L + max_delay + 2, which always is. */
  if ((L & 1) || !aec_fft_smooth(L / 2)) {
    seterr(err, errlen, "segment length %d is not usable: segment/2 = %d must "
                        "be a product of 2s, 3s and 5s", L, L / 2);
    return AEC_ERR_SEGMENT;
  }
  e->n_frac = next_pow2(L + e->max_delay + 2);
  e->fft_seg = aec_rfft_plan(L);
  e->fft_frac = aec_rfft_plan(e->n_frac);
  e->fft_win = aec_rfft_plan(AEC_NFFT);
  if (!e->fft_seg || !e->fft_frac || !e->fft_win) {
    seterr(err, errlen, "could not build the FFT plans for a %d-sample segment",
           L);
    return AEC_ERR_MEMORY;
  }

  e->seg_re = fvec(half); e->seg_im = fvec(half);
  e->cc_re = fvec(half); e->cc_im = fvec(half);
  e->corr = fvec(L);
  e->lags = fvec(e->nlag); e->rs = fvec(e->nlag); e->ws = fvec(e->nlag);
  e->reg_h = fvec(e->reg_hid);
  e->frac_re = fvec(e->n_frac / 2 + 1); e->frac_im = fvec(e->n_frac / 2 + 1);
  e->frac_buf = fvec(e->n_frac);
  e->ref_al = fvec(L);
  e->xm_re = fvec(nf * AEC_FREQ); e->xm_im = fvec(nf * AEC_FREQ);
  e->xr_re = fvec(nf * AEC_FREQ); e->xr_im = fvec(nf * AEC_FREQ);
  e->ac_re = fvec(nf * AEC_FREQ); e->ac_im = fvec(nf * AEC_FREQ);
  e->rs_re = fvec(nf * AEC_FREQ); e->rs_im = fvec(nf * AEC_FREQ);
  e->ym_re = fvec(nf * AEC_FREQ); e->ym_im = fvec(nf * AEC_FREQ);
  e->band = fvec((size_t)nf * e->in_dim);
  e->mag_rs = fvec(nf * AEC_FREQ);
  e->mag_xr = fvec(nf * AEC_FREQ);
  e->mag_xm = fvec(nf * AEC_FREQ);
  e->wih_t = fvec(e->in_dim * 3 * e->hid);
  e->whh_t = fvec(e->hid * 3 * e->hid);
  e->gain_re = fvec(nf * AEC_FREQ); e->gain_im = fvec(nf * AEC_FREQ);
  e->hh = fvec(3 * e->hid);
  e->gates = fvec(3 * e->hid);
  e->h0 = fvec(e->hid); e->h1 = fvec(e->hid);
  e->maskv = fvec(nf * AEC_FREQ);
  e->frame = fvec(AEC_NFFT);
  e->norm = fvec(e->lola);
  e->seg_out = fvec(L);
  e->seg_mic = fvec(L); e->seg_ref = fvec(L);

  const int bad = !e->seg_re || !e->seg_im || !e->cc_re || !e->cc_im ||
                  !e->corr || !e->lags || !e->rs || !e->ws || !e->reg_h ||
                  !e->frac_re || !e->frac_im || !e->frac_buf || !e->ref_al ||
                  !e->xm_re || !e->xm_im || !e->xr_re || !e->xr_im ||
                  !e->ac_re || !e->ac_im || !e->rs_re || !e->rs_im ||
                  !e->ym_re || !e->ym_im || !e->band || !e->mag_rs ||
                  !e->mag_xr || !e->mag_xm || !e->wih_t || !e->whh_t ||
                  !e->gain_re ||
                  !e->gain_im || !e->hh || !e->gates ||
                  !e->h0 || !e->h1 || !e->maskv || !e->frame || !e->norm ||
                  !e->seg_out || !e->seg_mic || !e->seg_ref;
  if (bad) {
    seterr(err, errlen, "out of memory");
    return AEC_ERR_MEMORY;
  }

  /* The GRU weights are stored row-major as [gate*hid + unit][input]. The
   * inner loop wants the input index outermost so it can broadcast one input
   * value across all outputs -- a stride-1 saxpy instead of a strided
   * reduction, which is the difference between ~4 and ~25 GFLOP/s. The
   * accumulation order per output is unchanged, so the result is identical. */
  /* The GRU weights arrive row-major as [gate*hid + unit][input]. The inner
   * loop wants the input index outermost, so that one input value is broadcast
   * across all outputs -- a stride-1 saxpy instead of a strided reduction,
   * which is the difference between roughly 4 and 25 GFLOP/s here. The
   * accumulation order for each output is unchanged (still ascending in the
   * input index), so the arithmetic result is identical, not merely close. */
  {
    const int H = e->hid, I = e->in_dim, G = 3 * H;
    const float* wih = e->w[B_GRU_WIH];
    const float* whh = e->w[B_GRU_WHH];
    for (int g = 0; g < 3; ++g)
      for (int j = 0; j < H; ++j)
        for (int i = 0; i < I; ++i)
          e->wih_t[(size_t)i * G + g * H + j] = wih[(size_t)(g * H + j) * I + i];
    for (int g = 0; g < 3; ++g)
      for (int j = 0; j < H; ++j)
        for (int i = 0; i < H; ++i)
          e->whh_t[(size_t)i * G + g * H + j] = whh[(size_t)(g * H + j) * H + i];
  }
  for (int i = 0; i < e->nlag; ++i) e->lags[i] = (float)(e->max_delay - i);

  /* sqrt-Hann, computed in double and rounded, the way stft.py does it */
  for (int i = 0; i < AEC_NFFT; ++i)
    e->window[i] = (float)sqrt(0.5 * (1.0 - cos(2.0 * M_PI * i / AEC_NFFT)));

  /* The overlap-add denominator does not depend on the signal, so it is the
   * same for every segment and every call. */
  for (int t = 0; t < nf; ++t)
    for (int i = 0; i < AEC_NFFT; ++i)
      e->norm[t * AEC_HOP + i] += e->window[i] * e->window[i];

  return AEC_OK;
}

AecEngine* aec_engine_from_memory(const void* blob, size_t nbytes, int segment,
                                  char* err, int errlen) {
  if (!blob) {
    seterr(err, errlen, "no weight blob given");
    return NULL;
  }
  AecEngine* e = (AecEngine*)xcalloc(1, sizeof(AecEngine));
  if (!e) {
    seterr(err, errlen, "out of memory");
    return NULL;
  }
  e->segment = segment > 0 ? segment : AEC_DEFAULT_SEGMENT;
  e->lp = e->segment + cover_tail_need(e->segment);
  e->nframes = (e->lp - AEC_NFFT) / AEC_HOP + 1;
  e->lola = (e->nframes - 1) * AEC_HOP + AEC_NFFT;

  int rc = parse_blob(e, blob, nbytes, err, errlen);
  if (rc == AEC_OK) rc = alloc_buffers(e, err, errlen);
  if (rc != AEC_OK) {
    aec_engine_free(e);
    return NULL;
  }
  return e;
}

AecEngine* aec_engine_new(const char* weight_path, int segment,
                          char* err, int errlen) {
  FILE* f = fopen(weight_path, "rb");
  if (!f) {
    seterr(err, errlen, "cannot open '%s'", weight_path);
    return NULL;
  }
  if (fseek(f, 0, SEEK_END) != 0) {
    seterr(err, errlen, "cannot seek in '%s'", weight_path);
    fclose(f);
    return NULL;
  }
  const long size = ftell(f);
  if (size <= 0) {
    seterr(err, errlen, "'%s' is empty", weight_path);
    fclose(f);
    return NULL;
  }
  rewind(f);
  unsigned char* buf = (unsigned char*)malloc((size_t)size);
  if (!buf) {
    seterr(err, errlen, "out of memory reading '%s'", weight_path);
    fclose(f);
    return NULL;
  }
  const size_t got = fread(buf, 1, (size_t)size, f);
  fclose(f);
  if (got != (size_t)size) {
    seterr(err, errlen, "short read on '%s'", weight_path);
    free(buf);
    return NULL;
  }
  AecEngine* e = aec_engine_from_memory(buf, (size_t)size, segment, err, errlen);
  free(buf);
  return e;
}

int aec_engine_segment(const AecEngine* e) { return e ? e->segment : 0; }

size_t aec_engine_memory(const AecEngine* e) {
  if (!e) return 0;
  size_t c = sizeof(AecEngine);
  c += sizeof(float) * (size_t)(2 * (e->segment / 2 + 1) +       /* seg spectra */
       e->segment + 3 * e->nlag + e->reg_hid +
       2 * (e->n_frac / 2 + 1) + e->n_frac + e->segment +
       6 * e->nframes * AEC_FREQ +
       (size_t)e->nframes * (e->in_dim + 2 * AEC_FREQ) +
       6 * e->hid + e->nframes * AEC_FREQ + AEC_NFFT + e->lola +
       2 * e->segment);
  /* each real plan holds a half-length complex plan: twiddles plus scratch,
   * which is on the order of 8 complex values per point */
  c += sizeof(float) * 8 * 2 * (size_t)(e->segment / 2 + e->n_frac / 2 +
                                        AEC_NFFT / 2);
  c += (size_t)e->segment * 2 * sizeof(double);   /* ola accumulators, when used */
  return c;
}

/* ------------------------------------------------- 1. delay estimation */

/* GCC-PHAT cross-correlation between microphone and reference, reduced to one
 * scalar lag by a soft-argmax plus a small regression correction.
 *
 * Two details in model.py matter and are easy to get wrong here:
 *   - `torch.std` defaults to the *unbiased* estimator, so the variance
 *     divides by n-1, not by n;
 *   - the correlation is circular at the exact segment length (no
 *     zero-padding), so a negative lag wraps to L + lag. */
static float estimate_delay(AecEngine* e, const float* mic, const float* ref) {
  const int L = e->segment, half = L / 2 + 1, nlag = e->nlag;

  aec_rfft_exec(e->fft_seg, mic, e->seg_re, e->seg_im);
  aec_rfft_exec(e->fft_seg, ref, e->cc_re, e->cc_im);
  for (int k = 0; k < half; ++k) {
    /* (a+bi) * conj(c+di) = (ac+bd) + (bc-ad)i */
    const float a = e->seg_re[k], b = e->seg_im[k];
    const float c = e->cc_re[k], d = e->cc_im[k];
    e->seg_re[k] = a * c + b * d;
    e->seg_im[k] = b * c - a * d;
  }
  /* PHAT weighting: unit magnitude, so the peak does not depend on the
   * loudspeaker or the room magnitude response */
  for (int k = 0; k < half; ++k) {
    const float m = sqrtf(e->seg_re[k] * e->seg_re[k] +
                          e->seg_im[k] * e->seg_im[k]) + 1e-6f;
    e->seg_re[k] /= m;
    e->seg_im[k] /= m;
  }
  aec_irfft_exec(e->fft_seg, e->seg_re, e->seg_im, e->corr);

  for (int i = 0; i < nlag; ++i) {
    long idx = (long)(e->max_delay - i) % L;
    if (idx < 0) idx += L;
    e->rs[i] = e->corr[idx];
  }

  /* Standardise first. The curve is very flat -- the peak sits near 1 while
   * the standard deviation of the whole curve is around 0.02 -- so a softmax
   * over the raw values would put essentially no mass on the peak and the
   * soft-argmax would collapse to zero. */
  double acc = 0.0;
  for (int i = 0; i < nlag; ++i) acc += e->rs[i];
  const float mean = (float)(acc / nlag);
  float sd = 0.0f;
  if (nlag > 1) {
    double v = 0.0;
    for (int i = 0; i < nlag; ++i) {
      const double d = e->rs[i] - mean;
      v += d * d;
    }
    sd = (float)sqrt(v / (nlag - 1));            /* unbiased */
  }
  {
    const float den = sd + 1e-6f;
    for (int i = 0; i < nlag; ++i) e->rs[i] /= den;
  }

  /* softmax(temperature * R), then the first two moments of the lag axis */
  float mx = e->rs[0] * AEC_TDE_TEMPERATURE;
  for (int i = 1; i < nlag; ++i)
    mx = fmaxf(mx, e->rs[i] * AEC_TDE_TEMPERATURE);
  double ssum = 0.0;
  for (int i = 0; i < nlag; ++i) {
    e->ws[i] = expf(e->rs[i] * AEC_TDE_TEMPERATURE - mx);
    ssum += e->ws[i];
  }
  double t1 = 0.0;
  for (int i = 0; i < nlag; ++i) {
    e->ws[i] = (float)(e->ws[i] / ssum);
    t1 += (double)e->ws[i] * e->lags[i];
  }
  const float tau0 = (float)t1;

  float peak = e->rs[0];
  for (int i = 1; i < nlag; ++i) peak = fmaxf(peak, e->rs[i]);
  double t2 = 0.0;
  for (int i = 0; i < nlag; ++i) {
    const double d = (double)e->lags[i] - tau0;
    t2 += (double)e->ws[i] * d * d;
  }
  const float sharp = (float)sqrt(t2);

  /* R.mean() / R.std() of the standardised curve, for the regression head */
  double m2 = 0.0;
  for (int i = 0; i < nlag; ++i) m2 += e->rs[i];
  const float rmean = (float)(m2 / nlag);
  float rstd = 0.0f;
  if (nlag > 1) {
    double v = 0.0;
    for (int i = 0; i < nlag; ++i) {
      const double d = e->rs[i] - rmean;
      v += d * d;
    }
    rstd = (float)sqrt(v / (nlag - 1));
  }

  const float inv_md = 1.0f / (float)e->max_delay;
  float feat[6];
  feat[0] = tau0 * inv_md;
  feat[1] = peak;
  feat[2] = sharp * inv_md;
  feat[3] = rmean;
  feat[4] = rstd;
  feat[5] = 1.0f;

  /* 6 -> reg_hid -> 1 with a tanh in between, correcting the content-based
   * soft-argmax */
  const float* rw0 = e->w[B_REG0_W];
  const float* rb0 = e->w[B_REG0_B];
  const float* rw2 = e->w[B_REG2_W];
  for (int j = 0; j < e->reg_hid; ++j) {
    float s = rb0[j];
    for (int i = 0; i < 6; ++i) s += rw0[j * 6 + i] * feat[i];
    e->reg_h[j] = tanhf(s);
  }
  float corr = e->w[B_REG2_B][0];
  for (int j = 0; j < e->reg_hid; ++j) corr += rw2[j] * e->reg_h[j];

  float tau = tau0 + corr;
  const float lim = (float)e->max_delay;
  if (tau > lim) tau = lim;
  if (tau < -lim) tau = -lim;
  return tau;
}

/* -------------------------------------------------- 2. fractional delay */

/* Delay the reference by a possibly fractional, possibly negative amount, as
 * a phase ramp on the zero-padded spectrum. Exact for band-limited signals,
 * which is the regime the delay estimator leaves us in. */
static void align_reference(AecEngine* e, const float* ref, float tau) {
  const int L = e->segment, n = e->n_frac, half = n / 2 + 1;
  memset(e->frac_buf, 0, sizeof(float) * (size_t)n);
  memcpy(e->frac_buf, ref, sizeof(float) * (size_t)L);
  aec_rfft_exec(e->fft_frac, e->frac_buf, e->frac_re, e->frac_im);
  /* The phase is evaluated in single precision, and in this order, to match
   * the graph. Theta reaches ~8000 radians at the top bin, where a float32
   * ulp is already 1e-3 -- so computing it in double instead would put us
   * ~5e-4 rad away from what ONNX Runtime produces, which shows up as a
   * 1e-4-level difference in the aligned reference. Reproducing the graph is
   * the point here, not being more accurate than it. */
  const float c2pi = (float)(-2.0 * M_PI);
  for (int k = 0; k < half; ++k) {
    const float theta = c2pi * ((float)k / (float)n) * tau;
    const float wr = cosf(theta), wi = sinf(theta);
    const float a = e->frac_re[k], b = e->frac_im[k];
    e->frac_re[k] = a * wr - b * wi;
    e->frac_im[k] = a * wi + b * wr;
  }
  aec_irfft_exec(e->fft_frac, e->frac_re, e->frac_im, e->frac_buf);
  memcpy(e->ref_al, e->frac_buf, sizeof(float) * (size_t)L);
}

/* --------------------------------------------------------------- 3. STFT */

static void stft_(AecEngine* e, const float* x, float* re, float* im) {
  const int L = e->segment;
  for (int t = 0; t < e->nframes; ++t) {
    const int off = t * AEC_HOP;
    const int n_copy = min_i(AEC_NFFT, max_i(L - off, 0));
    for (int i = 0; i < n_copy; ++i) e->frame[i] = x[off + i] * e->window[i];
    for (int i = n_copy; i < AEC_NFFT; ++i) e->frame[i] = 0.0f;
    aec_rfft_exec(e->fft_win, e->frame, re + (size_t)t * AEC_FREQ,
                  im + (size_t)t * AEC_FREQ);
  }
}

/* ------------------------------------------------- 4. echo estimate (FIR) */

/* A causal FIR across frames, one complex weight per bin per tap, applied to
 * the aligned reference: the y_hat of the model's second stage. The ONNX graph
 * expresses the same sum with a Pad and a Slice per tap, which is a large part
 * of why it is slow. */
static void echo_estimate(AecEngine* e) {
  const int T = e->nframes, F = AEC_FREQ, K = e->taps;
  const float* wr = e->w[B_FILT_RE];
  const float* wi = e->w[B_FILT_IM];
  memset(e->ac_re, 0, sizeof(float) * (size_t)T * F);
  memset(e->ac_im, 0, sizeof(float) * (size_t)T * F);
  for (int k = 0; k < K; ++k) {
    const float* cr = wr + (size_t)k * F;
    const float* ci = wi + (size_t)k * F;
    for (int t = k; t < T; ++t) {
      const float* sr = e->xr_re + (size_t)(t - k) * F;
      const float* si = e->xr_im + (size_t)(t - k) * F;
      float* dr = e->ac_re + (size_t)t * F;
      float* di = e->ac_im + (size_t)t * F;
      for (int f = 0; f < F; ++f) {
        const float a = sr[f], b = si[f], c = cr[f], d = ci[f];
        dr[f] += a * c - b * d;
        di[f] += a * d + b * c;
      }
    }
  }
}

/* ------------------------------------------- 5. least-squares echo gain */

/* alpha = <Xm, acc> / <acc, acc>, averaged over `k` frames and clamped.
 *
 * The estimate depends only on the reference, so it is non-zero whenever the
 * far end is playing -- even when the microphone holds no echo at all, in
 * which case subtracting it *injects* the far end. The gain measures how much
 * of the microphone the estimate actually explains: near one where it is
 * right, near zero where there is nothing to cancel. */
static void apply_adaptive_gain(AecEngine* e) {
  const int T = e->nframes, F = AEC_FREQ;
  const int k = e->adapt_gain;
  if (k <= 1) return;
  const int half = k / 2;
  const float scale = 1.0f / (float)k;

  double psum = 0.0;
  for (size_t i = 0; i < (size_t)T * F; ++i) {
    const float ar = e->ac_re[i], ai = e->ac_im[i];
    psum += (double)ar * ar + (double)ai * ai;
  }
  const float eps = 1e-3f * (float)(psum / ((double)T * F)) + 1e-12f;

  /* The gain is a moving average, so it reads frames the loop has not written
   * yet only if it is applied in place -- which would feed already-gained
   * frames back into the average. Compute it for the whole segment first. */
  for (int t = 0; t < T; ++t) {
    for (int f = 0; f < F; ++f) {
      float cr = 0.0f, ci = 0.0f, pw = 0.0f;
      for (int j = -half; j <= half; ++j) {
        int s = t + j;
        if (s < 0) s = 0;
        if (s > T - 1) s = T - 1;
        const size_t i = (size_t)s * F + f;
        const float mr = e->xm_re[i], mi = e->xm_im[i];
        const float ar = e->ac_re[i], ai = e->ac_im[i];
        cr += mr * ar + mi * ai;
        ci += mi * ar - mr * ai;
        pw += ar * ar + ai * ai;
      }
      cr *= scale; ci *= scale;
      const float den = pw * scale + eps;
      const size_t i = (size_t)t * F + f;
      e->gain_re[i] = cr / den;
      e->gain_im[i] = ci / den;
    }
  }
  for (size_t i = 0; i < (size_t)T * F; ++i) {
    const float gr = e->gain_re[i], gi = e->gain_im[i];
    const float mag = sqrtf(gr * gr + gi * gi);
    const float lim = mag < 2.0f ? mag : 2.0f;
    /* alpha / (|alpha| + 1e-9) * |alpha|.clamp(max=2): keep the phase, cap the
     * magnitude */
    const float s = lim / (mag + 1e-9f);
    const float ar = e->ac_re[i], ai = e->ac_im[i];
    e->ac_re[i] = (gr * s) * ar - (gi * s) * ai;
    e->ac_im[i] = (gr * s) * ai + (gi * s) * ar;
  }
}

/* --------------------------------------------------- 6. spectral bands */

/* |X| for a whole spectrogram. Computed once per spectrum rather than inside
 * each projection: the magnitude needs a square root, and the three (four,
 * with the microphone) projections would otherwise repeat it. */
static void magnitudes(const float* re, const float* im, size_t n, float* mag) {
  for (size_t i = 0; i < n; ++i) mag[i] = sqrtf(re[i] * re[i] + im[i] * im[i]);
}

/* bank(X) = relu(W |X| + b): a magnitude spectrum projected onto a few bands,
 * non-negative by construction. `out` is written with an explicit row stride
 * because the blocks of the feature vector are built one at a time. */
static void project_bands(const float* mag, int T, int bands, int stride,
                          int offset, const float* W, const float* b,
                          float* out) {
  for (int t = 0; t < T; ++t) {
    const float* m = mag + (size_t)t * AEC_FREQ;
    float* dst = out + (size_t)t * stride + offset;
    for (int q = 0; q < bands; ++q) {
      const float* wq = W + (size_t)q * AEC_FREQ;
      float s = b[q];
      for (int f = 0; f < AEC_FREQ; ++f) s += wq[f] * m[f];
      dst[q] = s > 0.0f ? s : 0.0f;
    }
  }
}

/* --------------------------------------------------------------- 7. GRU */

/* One step of PyTorch's nn.GRU with linear_before_reset=1:
 *
 *   r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
 *   z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
 *   n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
 *   h = (1 - z) n + z h
 *
 * The packed weights hold the gates in the order r, z, n. Getting that order
 * wrong is a silent, very hard-to-spot bug, so the gate index is spelled out
 * rather than folded into a loop. */
static void gru_step(AecEngine* e, const float* x, const float* hprev,
                     float* hnext) {
  const int H = e->hid, I = e->in_dim, G = 3 * H;
  const float* bih = e->w[B_GRU_BIH];
  const float* bhh = e->w[B_GRU_BHH];
  float* g = e->gates;
  float* hh = e->hh;

  for (int k = 0; k < G; ++k) {
    g[k] = bih[k];
    hh[k] = bhh[k];
  }
  for (int i = 0; i < I; ++i) {
    const float xi = x[i];
    const float* row = e->wih_t + (size_t)i * G;
    for (int k = 0; k < G; ++k) g[k] += row[k] * xi;
  }
  /* Every hh term is read from the *previous* hidden state, so they are all
   * accumulated before any component of h is updated. */
  for (int i = 0; i < H; ++i) {
    const float hi = hprev[i];
    const float* row = e->whh_t + (size_t)i * G;
    for (int k = 0; k < G; ++k) hh[k] += row[k] * hi;
  }
  for (int j = 0; j < H; ++j) {
    const float r = sigmoidf_(g[AEC_GATE_R * H + j] + hh[AEC_GATE_R * H + j]);
    const float z = sigmoidf_(g[AEC_GATE_Z * H + j] + hh[AEC_GATE_Z * H + j]);
    const float n = tanhf(g[AEC_GATE_N * H + j] + r * hh[AEC_GATE_N * H + j]);
    hnext[j] = (1.0f - z) * n + z * hprev[j];
  }
}

/* A moving average over time with replicated edges (model.py::time_average,
 * which pads then average-pools). Applied to the mask when the checkpoint asks
 * for it: the trained mask moves fast enough (a 99th-percentile frame-to-frame
 * change of 0.6 over 10 ms) that the modulation is audible as roughness. */
static void smooth_frames(float* v, int T, int F, int k) {
  if (k <= 1 || T < 1) return;
  const int half = k / 2;
  const float scale = 1.0f / (float)k;
  float* tmp = (float*)xcalloc((size_t)T * F, sizeof(float));
  if (!tmp) return;
  for (int t = 0; t < T; ++t)
    for (int f = 0; f < F; ++f) {
      float s = 0.0f;
      for (int j = -half; j <= half; ++j) {
        int s_idx = t + j;
        if (s_idx < 0) s_idx = 0;
        if (s_idx > T - 1) s_idx = T - 1;
        s += v[(size_t)s_idx * F + f];
      }
      tmp[(size_t)t * F + f] = s * scale;
    }
  memcpy(v, tmp, sizeof(float) * (size_t)T * F);
  free(tmp);
}

/* ------------------------------------------------- 8. mask and synthesis */

/* sigmoid(head(h)) -> 16 band gains, then sigmoid(mask_proj(.)) -> one gain
 * per frequency bin. */
static void mask_for_frame(AecEngine* e, const float* h, float* mask_out) {
  const int B = e->bands, H = e->hid;
  const float* hw = e->w[B_HEAD_W];
  const float* hb = e->w[B_HEAD_B];
  const float* mw = e->w[B_MASK_W];
  const float* mb = e->w[B_MASK_B];
  float gt[64];
  if (B > 64) return;
  for (int q = 0; q < B; ++q) {
    const float* wq = hw + (size_t)q * H;
    float s = hb[q];
    for (int j = 0; j < H; ++j) s += wq[j] * h[j];
    gt[q] = sigmoidf_(s);
  }
  for (int f = 0; f < AEC_FREQ; ++f) {
    const float* wf = mw + (size_t)f * B;
    float s = mb[f];
    for (int q = 0; q < B; ++q) s += wf[q] * gt[q];
    mask_out[f] = sigmoidf_(s);
  }
}

/* --------------------------------------------------------------- driver */

void aec_engine_run(AecEngine* e, const float* mic, const float* ref,
                    float* out, float* tau) {
  if (!e || !mic || !ref || !out) {
    if (tau) *tau = 0.0f;
    return;
  }
  const int T = e->nframes, F = AEC_FREQ, B = e->bands, nb = e->in_dim;
  const float* bank_w = e->w[B_BANK_W];
  const float* bank_b = e->w[B_BANK_B];

  /* 1. delay estimate, then align the reference by it */
  const float t_hat = estimate_delay(e, mic, ref);
  align_reference(e, ref, t_hat);
  if (tau) *tau = t_hat;

  /* 2. analysis */
  stft_(e, mic, e->xm_re, e->xm_im);
  stft_(e, e->ref_al, e->xr_re, e->xr_im);

  /* 3. echo estimate, then the residual */
  echo_estimate(e);
  if (e->adapt_gain > 1) apply_adaptive_gain(e);
  for (size_t i = 0; i < (size_t)T * F; ++i) {
    e->rs_re[i] = e->xm_re[i] - e->ac_re[i];
    e->rs_im[i] = e->xm_im[i] - e->ac_im[i];
  }

  /* 4. band features: residual, aligned reference, their product, and the
   * microphone itself. The last one matters -- without it the network cannot
   * tell "echo present" from "far end loud but the microphone is silent",
   * because the residual bands look the same in both cases. */
  magnitudes(e->rs_re, e->rs_im, (size_t)T * F, e->mag_rs);
  magnitudes(e->xr_re, e->xr_im, (size_t)T * F, e->mag_xr);
  if (e->use_mic_feat) magnitudes(e->xm_re, e->xm_im, (size_t)T * F, e->mag_xm);
  project_bands(e->mag_rs, T, B, nb, 0, bank_w, bank_b, e->band);
  project_bands(e->mag_xr, T, B, nb, B, bank_w, bank_b, e->band);
  for (int t = 0; t < T; ++t) {
    float* row = e->band + (size_t)t * nb;
    for (int q = 0; q < B; ++q) row[2 * B + q] = row[q] * row[B + q];
  }
  if (e->use_mic_feat)
    project_bands(e->mag_xm, T, B, nb, 3 * B, bank_w, bank_b, e->band);

  /* 5. the recurrent mask network over those band energies */
  memset(e->h0, 0, sizeof(float) * (size_t)e->hid);
  for (int t = 0; t < T; ++t) {
    gru_step(e, e->band + (size_t)t * nb, e->h0, e->h1);
    memcpy(e->h0, e->h1, sizeof(float) * (size_t)e->hid);
    mask_for_frame(e, e->h0, e->maskv + (size_t)t * F);
  }
  smooth_frames(e->maskv, T, F, e->mask_smooth);

  /* 6. shape the residual, then overlap-add back to the time domain */
  for (size_t i = 0; i < (size_t)T * F; ++i) {
    const float m = e->maskv[i];
    e->ym_re[i] = e->rs_re[i] * m;
    e->ym_im[i] = e->rs_im[i] * m;
  }
  memset(e->seg_out, 0, sizeof(float) * (size_t)e->segment);
  for (int t = 0; t < T; ++t) {
    aec_irfft_exec(e->fft_win, e->ym_re + (size_t)t * F,
                   e->ym_im + (size_t)t * F, e->frame);
    const int off = t * AEC_HOP;
    const int n_add = min_i(AEC_NFFT, e->segment - off);
    for (int i = 0; i < n_add; ++i)
      e->seg_out[off + i] += e->frame[i] * e->window[i];
  }
  for (int p = 0; p < e->segment; ++p) {
    const float d = e->norm[p];
    out[p] = e->seg_out[p] / (d > 1e-10f ? d : 1e-10f);
  }
}

/* ---------------------------------------------------------- whole file */

void aec_engine_process(AecEngine* e, const float* mic, const float* ref,
                        size_t n, float* out, float* tau) {
  aec_engine_process_ex(e, mic, ref, n, out, tau, NULL, 0);
}

int aec_engine_process_ex(AecEngine* e, const float* mic, const float* ref,
                          size_t n, float* out, float* tau, float* seg_taus,
                          int max_taus) {
  if (!e || !mic || !ref || !out) {
    if (tau) *tau = 0.0f;
    return 0;
  }
  const int seg = e->segment;

  if ((int)n <= seg) {
    for (int i = 0; i < seg; ++i) {
      const int have = i < (int)n;
      e->seg_mic[i] = have ? mic[i] : 0.0f;
      e->seg_ref[i] = have ? ref[i] : 0.0f;
    }
    float t = 0.0f;
    aec_engine_run(e, e->seg_mic, e->seg_ref, e->seg_out, &t);
    for (size_t i = 0; i < n; ++i) out[i] = e->seg_out[i];
    if (tau) *tau = t;
    if (seg_taus && max_taus > 0) seg_taus[0] = t;
    return 1;
  }

  const int hop = seg / 2;
  if (!e->ola_num) {
    e->ola_num = (double*)xcalloc((size_t)n + (size_t)seg, sizeof(double));
    e->ola_den = (double*)xcalloc((size_t)n + (size_t)seg, sizeof(double));
    e->xfadew = fvec(seg);
    for (int i = 0; i < seg; ++i)
      e->xfadew[i] = (float)(0.5 - 0.5 * cos(2.0 * M_PI * (i + 1) / (seg + 1)));
    if (!e->ola_num || !e->ola_den || !e->xfadew) return 0;
  }
  const size_t ntotal = n + (size_t)seg;
  memset(e->ola_num, 0, sizeof(double) * ntotal);
  memset(e->ola_den, 0, sizeof(double) * ntotal);

  /* hop-aligned starts, then one more right-aligned to the end -- the same
   * schedule as infer.py::run_segmented */
  const size_t nstarts = ((n - (size_t)seg - 1) / (size_t)hop + 1) + 1;
  float* taus = (float*)xcalloc(nstarts, sizeof(float));
  if (!taus) return 0;

  size_t si = 0;
  for (size_t s = 0; s + (size_t)seg < n; s += (size_t)hop) {
    aec_engine_run(e, mic + s, ref + s, e->seg_out, &taus[si++]);
    for (int i = 0; i < seg; ++i) {
      e->ola_num[s + i] += (double)e->xfadew[i] * e->seg_out[i];
      e->ola_den[s + i] += e->xfadew[i];
    }
  }
  {
    const size_t s = n - (size_t)seg;
    aec_engine_run(e, mic + s, ref + s, e->seg_out, &taus[si++]);
    for (int i = 0; i < seg; ++i) {
      e->ola_num[s + i] += (double)e->xfadew[i] * e->seg_out[i];
      e->ola_den[s + i] += e->xfadew[i];
    }
  }
  for (size_t i = 0; i < n; ++i) {
    const double d = e->ola_den[i];
    out[i] = (float)(e->ola_num[i] / (d > 1e-9 ? d : 1e-9));
  }

  if (seg_taus)
    for (size_t i = 0; i < si && (int)i < max_taus; ++i) seg_taus[i] = taus[i];

  if (si) {
    for (size_t a = 1; a < si; ++a) {   /* insertion sort: a handful of items */
      const float v = taus[a];
      size_t b = a;
      while (b > 0 && taus[b - 1] > v) {
        taus[b] = taus[b - 1];
        --b;
      }
      taus[b] = v;
    }
    if (tau)
      *tau = (si & 1) ? taus[si / 2]
                      : 0.5f * (taus[si / 2 - 1] + taus[si / 2]);
  }
  free(taus);
  return (int)si;
}

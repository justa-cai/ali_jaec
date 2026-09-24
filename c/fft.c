#include "fft.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int aec_fft_smooth(int n) {
  if (n < 1) return 0;
  static const int primes[3] = {2, 3, 5};
  for (int i = 0; i < 3; ++i)
    while (n % primes[i] == 0) n /= primes[i];
  return n == 1;
}

static int smallest_factor(int n) {
  if (n % 2 == 0) return 2;
  if (n % 3 == 0) return 3;
  if (n % 5 == 0) return 5;
  return n;                     /* only reachable for a prime n > 5 */
}

/* ------------------------------------------------------------ complex FFT */

typedef struct {
  int n;      /* size of the sub-transform at this level */
  int p;      /* radix applied here; stage.n / stage.p is the next level */
} Stage;

struct AecCfft {
  int n;
  int nstages;
  Stage* stage;
  float* tw_re;                 /* exp(-2*pi*i*k/n), k = 0..n-1 */
  float* tw_im;
  float wp_re[6][5];            /* exp(-2*pi*i*e/p), p = 2..5, e = 0..p-1 */
  float wp_im[6][5];
  float* in_re;                 /* copy of the caller's input, n */
  float* in_im;
  float* out_re;                /* the transform, n */
  float* out_im;
  float* sc_re;                 /* recursion scratch, 2n */
  float* sc_im;
};

/* out[0..n-1] = sum in[j] * exp(sign*2*pi*i*j*k/n), unnormalised, where the
 * stride arguments let the caller place a sub-transform inside a larger
 * buffer. `scratch` has room for 2n points: this level's small-DFT results in
 * scratch[0..n) and everything the recursion needs after that.
 *
 * The split is n = p*m with the input index a*m+b (a < p, b < m) and the
 * output index c+p*d (c < p, d < m):
 *
 *     X[c + p*d] = sum_b w_n^{b*c} w_m^{b*d} * ( sum_a in[a*m+b] * w_p^{a*c} )
 *
 * so the body is: a p-point DFT down each column, a twiddle, then p
 * sub-transforms read with stride p. */
static void cfft_rec(const AecCfft* p, int lvl,
                     float* or_, float* oi, int so,
                     const float* ir, const float* ii, int si,
                     float* sr, float* si_) {
  const int n = p->stage[lvl].n;
  if (n == 1) {
    or_[0] = ir[0];
    oi[0] = ii[0];
    return;
  }
  const int radix = p->stage[lvl].p;
  const int m = n / radix;
  const long tstep = (long)p->n / n;   /* w_n^k = tw[k * tstep] */

  if (radix == 2) {
    for (int b = 0; b < m; ++b) {
      const float x0r = ir[(long)b * si], x0i = ii[(long)b * si];
      const float x1r = ir[(long)(m + b) * si], x1i = ii[(long)(m + b) * si];
      /* c = 0: twiddle is 1 */
      sr[b] = x0r + x1r;
      si_[b] = x0i + x1i;
      /* c = 1 */
      const long k = (long)b * tstep;
      const float wr = p->tw_re[k], wi = p->tw_im[k];
      const float dr = x0r - x1r, di = x0i - x1i;
      sr[m + b] = dr * wr - di * wi;
      si_[m + b] = dr * wi + di * wr;
    }
  } else {
    for (int b = 0; b < m; ++b) {
      float xr[5], xi[5];
      for (int a = 0; a < radix; ++a) {
        const long idx = (long)(a * m + b) * si;
        xr[a] = ir[idx];
        xi[a] = ii[idx];
      }
      for (int c = 0; c < radix; ++c) {
        float ar = 0.0f, ai = 0.0f;
        for (int a = 0; a < radix; ++a) {
          const int e = (a * c) % radix;
          const float wr = p->wp_re[radix][e], wi = p->wp_im[radix][e];
          ar += xr[a] * wr - xi[a] * wi;
          ai += xr[a] * wi + xi[a] * wr;
        }
        if (c == 0) {
          sr[b] = ar;
          si_[b] = ai;
        } else {
          const long k = (long)c * b * tstep;
          const float wr = p->tw_re[k], wi = p->tw_im[k];
          sr[(long)c * m + b] = ar * wr - ai * wi;
          si_[(long)c * m + b] = ar * wi + ai * wr;
        }
      }
    }
  }

  for (int c = 0; c < radix; ++c)
    cfft_rec(p, lvl + 1, or_ + (long)c * so, oi + (long)c * so, so * radix,
             sr + (long)c * m, si_ + (long)c * m, 1, sr + n, si_ + n);
}

AecCfft* aec_cfft_plan(int n) {
  if (n < 1 || !aec_fft_smooth(n)) return NULL;

  AecCfft* p = (AecCfft*)calloc(1, sizeof(AecCfft));
  if (!p) return NULL;
  p->n = n;

  int stages = 1, k = n;
  while (k > 1) {
    k /= smallest_factor(k);
    ++stages;
  }
  p->nstages = stages;
  p->stage = (Stage*)calloc((size_t)stages, sizeof(Stage));
  if (!p->stage) goto fail;
  k = n;
  for (int i = 0; i < stages; ++i) {
    p->stage[i].n = k;
    p->stage[i].p = (k > 1) ? smallest_factor(k) : 1;
    k = (k > 1) ? k / p->stage[i].p : 1;
  }

  p->tw_re = (float*)malloc(sizeof(float) * (size_t)n);
  p->tw_im = (float*)malloc(sizeof(float) * (size_t)n);
  p->in_re = (float*)malloc(sizeof(float) * (size_t)n);
  p->in_im = (float*)malloc(sizeof(float) * (size_t)n);
  p->out_re = (float*)malloc(sizeof(float) * (size_t)n);
  p->out_im = (float*)malloc(sizeof(float) * (size_t)n);
  p->sc_re = (float*)malloc(sizeof(float) * (size_t)(2 * n + 8));
  p->sc_im = (float*)malloc(sizeof(float) * (size_t)(2 * n + 8));
  if (!p->tw_re || !p->tw_im || !p->in_re || !p->in_im || !p->out_re ||
      !p->out_im || !p->sc_re || !p->sc_im)
    goto fail;

  for (int i = 0; i < n; ++i) {
    const double t = -2.0 * M_PI * (double)i / (double)n;
    p->tw_re[i] = (float)cos(t);
    p->tw_im[i] = (float)sin(t);
  }
  for (int q = 2; q <= 5; ++q)
    for (int e = 0; e < q; ++e) {
      const double t = -2.0 * M_PI * (double)e / (double)q;
      p->wp_re[q][e] = (float)cos(t);
      p->wp_im[q][e] = (float)sin(t);
    }
  return p;

fail:
  aec_cfft_free(p);
  return NULL;
}

void aec_cfft_free(AecCfft* p) {
  if (!p) return;
  free(p->stage);
  free(p->tw_re);
  free(p->tw_im);
  free(p->in_re);
  free(p->in_im);
  free(p->out_re);
  free(p->out_im);
  free(p->sc_re);
  free(p->sc_im);
  free(p);
}

int aec_cfft_size(const AecCfft* p) { return p ? p->n : 0; }

void aec_cfft_exec(AecCfft* p, float* re, float* im, int inverse) {
  const int n = p->n;
  if (inverse) {
    /* conj -> forward -> conj -> scale: one code path for both directions */
    for (int i = 0; i < n; ++i) {
      p->in_re[i] = re[i];
      p->in_im[i] = -im[i];
    }
  } else {
    for (int i = 0; i < n; ++i) {
      p->in_re[i] = re[i];
      p->in_im[i] = im[i];
    }
  }
  cfft_rec(p, 0, p->out_re, p->out_im, 1, p->in_re, p->in_im, 1,
           p->sc_re, p->sc_im);
  if (inverse) {
    const float s = 1.0f / (float)n;
    for (int i = 0; i < n; ++i) {
      re[i] = p->out_re[i] * s;
      im[i] = -p->out_im[i] * s;
    }
  } else {
    memcpy(re, p->out_re, sizeof(float) * (size_t)n);
    memcpy(im, p->out_im, sizeof(float) * (size_t)n);
  }
}

/* --------------------------------------------------------------- real FFT */

struct AecRfft {
  int n;                        /* real length */
  int m;                        /* n / 2 */
  AecCfft* half;
  float* pre_re;                /* exp(-2*pi*i*k/n), k = 0..m */
  float* pre_im;
  float* z_re;                  /* the packed half-length sequence, m points */
  float* z_im;
};

AecRfft* aec_rfft_plan(int n) {
  if (n < 2 || (n & 1) || !aec_fft_smooth(n / 2)) return NULL;

  AecRfft* p = (AecRfft*)calloc(1, sizeof(AecRfft));
  if (!p) return NULL;
  p->n = n;
  p->m = n / 2;
  p->half = aec_cfft_plan(p->m);
  p->pre_re = (float*)malloc(sizeof(float) * (size_t)(p->m + 1));
  p->pre_im = (float*)malloc(sizeof(float) * (size_t)(p->m + 1));
  p->z_re = (float*)malloc(sizeof(float) * (size_t)p->m);
  p->z_im = (float*)malloc(sizeof(float) * (size_t)p->m);
  if (!p->half || !p->pre_re || !p->pre_im || !p->z_re || !p->z_im) {
    aec_rfft_free(p);
    return NULL;
  }
  for (int k = 0; k <= p->m; ++k) {
    const double t = -2.0 * M_PI * (double)k / (double)n;
    p->pre_re[k] = (float)cos(t);
    p->pre_im[k] = (float)sin(t);
  }
  return p;
}

void aec_rfft_free(AecRfft* p) {
  if (!p) return;
  aec_cfft_free(p->half);
  free(p->pre_re);
  free(p->pre_im);
  free(p->z_re);
  free(p->z_im);
  free(p);
}

int aec_rfft_size(const AecRfft* p) { return p ? p->n : 0; }

/* One-sided transform of a real sequence, via the half-length complex
 * transform of the even/odd interleave. With M = n/2 and
 * z[i] = x[2i] + i*x[2i+1],
 *
 *     E[k] = (Z[k] + conj(Z[M-k])) / 2      DFT of the even samples
 *     O[k] = (Z[k] - conj(Z[M-k])) / (2i)   DFT of the odd samples
 *     X[k] = E[k] + exp(-2*pi*i*k/n) * O[k]
 *
 * for k = 0..M, with Z[M] read as Z[0]. */
void aec_rfft_exec(AecRfft* p, const float* x, float* re, float* im) {
  const int m = p->m;
  for (int i = 0; i < m; ++i) {
    p->z_re[i] = x[2 * i];
    p->z_im[i] = x[2 * i + 1];
  }
  aec_cfft_exec(p->half, p->z_re, p->z_im, 0);

  for (int k = 0; k <= m; ++k) {
    const int j = k % m, jm = (m - k) % m;
    const float ar = p->z_re[j], ai = p->z_im[j];
    const float br = p->z_re[jm], bi = p->z_im[jm];
    const float er = 0.5f * (ar + br), ei = 0.5f * (ai - bi);
    const float tr = 0.5f * (ai + bi), ti = -0.5f * (ar - br);
    const float wr = p->pre_re[k], wi = p->pre_im[k];
    re[k] = er + (tr * wr - ti * wi);
    im[k] = ei + (tr * wi + ti * wr);
  }
}

void aec_irfft_exec(AecRfft* p, const float* re, const float* im, float* x) {
  const int m = p->m;
  for (int k = 0; k < m; ++k) {
    const float xr = re[k], xi = im[k];
    const float yr = re[m - k], yi = im[m - k];
    const float er = 0.5f * (xr + yr), ei = 0.5f * (xi - yi);
    const float dr = xr - yr, di = xi + yi;
    const float wr = p->pre_re[k], wi = p->pre_im[k];
    /* O = (X[k] - conj(X[M-k])) * exp(+2*pi*i*k/n) / 2 */
    const float tr = 0.5f * (dr * wr + di * wi);
    const float ti = 0.5f * (di * wr - dr * wi);
    p->z_re[k] = er - ti;
    p->z_im[k] = ei + tr;
  }
  aec_cfft_exec(p->half, p->z_re, p->z_im, 1);
  for (int i = 0; i < m; ++i) {
    x[2 * i] = p->z_re[i];
    x[2 * i + 1] = p->z_im[i];
  }
}

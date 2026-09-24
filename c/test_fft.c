/* Self-test for the FFT. No build system, no test framework:
 *
 *     cc -O2 -std=c99 -o test_fft test_fft.c fft.c -lm && ./test_fft
 *
 * Two kinds of check:
 *
 *   - against a naive O(n^2) DFT, which is what pins the twiddle signs, the
 *     radix decomposition and the overall scale. Afforded up to n = 500.
 *   - round-trip identity and Parseval, which are cheap and are run at every
 *     2/3/5-smooth size up to 250 and at the three sizes the model uses
 *     (48000, 65536, 512).
 *
 * Exits non-zero on any failure.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NAIVE_LIMIT 500      /* largest size compared against a naive DFT */
#define SMALL_LIMIT 250      /* largest size given the full treatment */

static int failures = 0;
static int checks = 0;

/* A tiny deterministic generator, so a failure is reproducible without
 * shipping test vectors. */
static unsigned rng_state = 123456789u;
static double rnd(void) {
  rng_state = rng_state * 1103515245u + 12345u;
  return (double)((rng_state >> 8) & 0xffffffu) / 8388608.0 - 1.0;
}

static double max_abs_diff(const float* a, const float* b, int n) {
  double m = 0.0;
  for (int i = 0; i < n; ++i) {
    const double d = fabs((double)a[i] - (double)b[i]);
    if (d > m) m = d;
  }
  return m;
}

static double rms(const float* a, int n) {
  double s = 0.0;
  for (int i = 0; i < n; ++i) s += (double)a[i] * a[i];
  return sqrt(s / n);
}

/* X[k] = sum_j x[j] exp(sign*2*pi*i*k*j/N), with 1/N applied when inverse. */
static void naive_dft(const double* x, int n, double* re, double* im) {
  for (int k = 0; k < n; ++k) {
    double sr = 0.0, si = 0.0;
    for (int j = 0; j < n; ++j) {
      const double t = 2.0 * M_PI * (double)k * (double)j / (double)n;
      sr += x[j] * cos(t);
      si -= x[j] * sin(t);
    }
    re[k] = sr;
    im[k] = si;
  }
}

static void check(int cond, const char* what, int n, const char* detail) {
  ++checks;
  if (!cond) {
    printf("  FAIL  %-32s n=%-6d %s\n", what, n, detail);
    ++failures;
  }
}

static double rel(double err, double scale) { return err / (scale > 0 ? scale : 1); }

/* The complex transform, checked against a naive DFT of the same complex
 * input. The naive DFT only takes real input, so the imaginary part is
 * transformed separately and combined by linearity:
 * DFT(a + i b) = DFT(a) + i DFT(b). */
static void test_complex_naive(int n) {
  double* ar = malloc(sizeof(double) * n);
  double* ai = calloc(n, sizeof(double));
  double* br = malloc(sizeof(double) * n);
  double* bi = calloc(n, sizeof(double));
  double* gr = malloc(sizeof(double) * n);
  double* gi = malloc(sizeof(double) * n);
  float* xr = malloc(sizeof(float) * n);
  float* xi = malloc(sizeof(float) * n);
  if (!ar || !ai || !br || !bi || !gr || !gi || !xr || !xi) exit(1);
  for (int i = 0; i < n; ++i) {
    ar[i] = rnd();
    br[i] = rnd();
    xr[i] = (float)ar[i];
    xi[i] = (float)br[i];
  }
  naive_dft(ar, n, gr, gi);
  naive_dft(br, n, ar, ai);
  for (int k = 0; k < n; ++k) {          /* (Gr - Ai) + i (Gi + Ar) */
    const double r = gr[k] - ai[k];
    const double m = gi[k] + ar[k];
    gr[k] = r;
    gi[k] = m;
  }

  AecCfft* p = aec_cfft_plan(n);
  if (!p) { printf("  FAIL  aec_cfft_plan(%d)\n", n); ++failures; return; }
  aec_cfft_exec(p, xr, xi, 0);

  double scale = 0.0;
  for (int k = 0; k < n; ++k) {
    const double m = fabs(gr[k]);
    if (m > scale) scale = m;
  }
  double er = 0.0, ei = 0.0;
  for (int k = 0; k < n; ++k) {
    const double dr = fabs((double)xr[k] - gr[k]);
    const double di = fabs((double)xi[k] - gi[k]);
    if (dr > er) er = dr;
    if (di > ei) ei = di;
  }
  char buf[128];
  snprintf(buf, sizeof buf, "re rel %.2e im rel %.2e", rel(er, scale), rel(ei, scale));
  check(rel(er, scale) < 1e-5 && rel(ei, scale) < 1e-5, "cfft vs naive DFT", n, buf);

  aec_cfft_free(p);
  free(ar); free(ai); free(br); free(bi); free(gr); free(gi); free(xr); free(xi);
}

/* Round-trip and Parseval for the complex transform. */
static void test_complex_roundtrip(int n) {
  float* xr = malloc(sizeof(float) * n);
  float* xi = malloc(sizeof(float) * n);
  float* rr = malloc(sizeof(float) * n);
  float* ri = malloc(sizeof(float) * n);
  if (!xr || !xi || !rr || !ri) exit(1);
  for (int i = 0; i < n; ++i) {
    xr[i] = (float)rnd();
    xi[i] = (float)rnd();
  }
  AecCfft* p = aec_cfft_plan(n);
  if (!p) { printf("  FAIL  aec_cfft_plan(%d)\n", n); ++failures; return; }
  memcpy(rr, xr, sizeof(float) * n);
  memcpy(ri, xi, sizeof(float) * n);
  aec_cfft_exec(p, rr, ri, 0);
  aec_cfft_exec(p, rr, ri, 1);
  char buf[128];
  const double e = fmax(max_abs_diff(rr, xr, n), max_abs_diff(ri, xi, n));
  const double s = rms(xr, n) + rms(xi, n);
  snprintf(buf, sizeof buf, "abs %.2e (rms %.2e)", e, s);
  check(e < 1e-4 * s, "cfft round trip", n, buf);
  aec_cfft_free(p);
  free(xr); free(xi); free(rr); free(ri);
}

/* The real transform: against a naive DFT when the size allows, and always
 * round-trip + Parseval. */
static void test_real(int n, int with_naive) {
  float* x = malloc(sizeof(float) * n);
  float* y = malloc(sizeof(float) * n);
  float* re = malloc(sizeof(float) * (n / 2 + 1));
  float* im = malloc(sizeof(float) * (n / 2 + 1));
  if (!x || !y || !re || !im) exit(1);
  for (int i = 0; i < n; ++i) x[i] = (float)rnd();

  AecRfft* p = aec_rfft_plan(n);
  if (!p) { printf("  FAIL  aec_rfft_plan(%d)\n", n); ++failures; return; }
  aec_rfft_exec(p, x, re, im);

  if (with_naive) {
    double* xd = calloc(n, sizeof(double));
    double* gr = malloc(sizeof(double) * n);
    double* gi = malloc(sizeof(double) * n);
    if (!xd || !gr || !gi) exit(1);
    for (int i = 0; i < n; ++i) xd[i] = x[i];
    naive_dft(xd, n, gr, gi);
    double scale = 0.0;
    for (int k = 0; k <= n / 2; ++k) {
      const double m = fabs(gr[k]);
      if (m > scale) scale = m;
    }
    double er = 0.0, ei = 0.0;
    for (int k = 0; k <= n / 2; ++k) {
      const double dr = fabs((double)re[k] - gr[k]);
      const double di = fabs((double)im[k] - gi[k]);
      if (dr > er) er = dr;
      if (di > ei) ei = di;
    }
    char buf[128];
    snprintf(buf, sizeof buf, "re rel %.2e im rel %.2e", rel(er, scale), rel(ei, scale));
    check(rel(er, scale) < 1e-5 && rel(ei, scale) < 1e-5, "rfft vs naive DFT", n, buf);
    free(xd); free(gr); free(gi);
  }

  aec_irfft_exec(p, re, im, y);
  {
    char buf[128];
    const double e = max_abs_diff(y, x, n);
    snprintf(buf, sizeof buf, "abs %.2e (rms %.2e)", e, rms(x, n));
    check(e < 1e-4 * rms(x, n), "irfft(rfft(x)) == x", n, buf);
  }

  /* Parseval: the energy of a real signal is the sum of |X[k]|^2 over the
   * one-sided spectrum, counting the interior bins twice and the two ends
   * once. */
  {
    double lhs = 0.0, rhs = 0.0;
    for (int i = 0; i < n; ++i) lhs += (double)x[i] * x[i];
    for (int k = 0; k <= n / 2; ++k) {
      const double e = (double)re[k] * re[k] + (double)im[k] * im[k];
      rhs += (k == 0 || k == n / 2) ? e : 2.0 * e;
    }
    rhs /= n;
    char buf[128];
    snprintf(buf, sizeof buf, "%.6f vs %.6f", lhs, rhs);
    check(fabs(lhs - rhs) < 1e-4 * lhs, "Parseval", n, buf);
  }

  aec_rfft_free(p);
  free(x); free(y); free(re); free(im);
}

int main(void) {
  printf("mixed-radix FFT self-test\n\n");

  printf("against a naive DFT, every 2/3/5-smooth size <= %d:\n", NAIVE_LIMIT);
  int n_naive = 0;
  for (int n = 1; n <= NAIVE_LIMIT; ++n) {
    if (!aec_fft_smooth(n)) continue;
    test_complex_naive(n);
    ++n_naive;
  }
  for (int n = 2; n <= NAIVE_LIMIT; n += 2) {
    if (!aec_fft_smooth(n / 2)) continue;
    test_real(n, 1);
  }
  printf("  complex: %d sizes, real: even sizes with a smooth half\n", n_naive);
  printf("  %d checks so far, %d failures\n", checks, failures);

  printf("\nround trip and Parseval, every 2/3/5-smooth size <= %d:\n", SMALL_LIMIT);
  const int before = failures;
  int n_rt = 0;
  for (int n = 1; n <= SMALL_LIMIT; ++n) {
    if (!aec_fft_smooth(n)) continue;
    test_complex_roundtrip(n);
    ++n_rt;
  }
  for (int n = 2; n <= SMALL_LIMIT; n += 2) {
    if (!aec_fft_smooth(n / 2)) continue;
    test_real(n, 0);
  }
  printf("  %d complex sizes, %d failures\n", n_rt, failures - before);

  printf("\nthe three sizes the model uses:\n");
  const int before2 = failures;
  const int big[3] = {48000, 65536, 512};
  for (int i = 0; i < 3; ++i) {
    test_real(big[i], 0);
    if (aec_fft_smooth(big[i] / 2)) test_complex_roundtrip(big[i] / 2);
  }
  printf("  48000 / 65536 / 512, plus the half-length complex transforms: "
         "%d failures\n", failures - before2);

  printf("\n%d checks, %d failures -- %s\n", checks, failures,
         failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}

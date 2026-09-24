/* Mixed-radix FFT, standard C only.
 *
 * The model works at three transform sizes and all of them are 2/3/5-smooth:
 *
 *     real length   why                                  complex length
 *     48000         the delay estimator's circular        24000 = 2^6*3*5^3
 *                   correlation (exact, not zero-padded)
 *     65536         the fractional-delay phase ramp       32768 = 2^15
 *     512           the analysis / synthesis STFT         256 = 2^8
 *
 * so a radix-2/3/5 Cooley-Tukey is enough -- no Bluestein, no prime sizes.
 *
 * The real transforms are built on a half-length complex transform (the usual
 * even/odd split), which halves the work and needs one extra twiddle table of
 * exp(-2*pi*i*k/n) for the unpacking step.
 *
 * A plan owns its scratch memory, so it is not thread-safe: build one plan per
 * size and drive it from one thread.
 */
#ifndef AEC_FFT_H
#define AEC_FFT_H

/* True if n factors completely into 2, 3 and 5. */
int aec_fft_smooth(int n);

/* ------------------------------------------------------------ complex FFT */
typedef struct AecCfft AecCfft;

/* n must be 2/3/5-smooth. Returns NULL if it is not, or on allocation
 * failure. */
AecCfft* aec_cfft_plan(int n);
void aec_cfft_free(AecCfft* p);
int aec_cfft_size(const AecCfft* p);

/* In-place transform of n complex points held as separate real and imaginary
 * arrays. `inverse` applies the 1/n scaling, so a forward followed by an
 * inverse is the identity. */
void aec_cfft_exec(AecCfft* p, float* re, float* im, int inverse);

/* --------------------------------------------------------------- real FFT */
typedef struct AecRfft AecRfft;

/* n must be even and n/2 must be 2/3/5-smooth. */
AecRfft* aec_rfft_plan(int n);
void aec_rfft_free(AecRfft* p);
int aec_rfft_size(const AecRfft* p);

/* x[0..n-1] -> re/im[0..n/2], the one-sided spectrum. No scaling. */
void aec_rfft_exec(AecRfft* p, const float* x, float* re, float* im);

/* re/im[0..n/2], the one-sided spectrum of a real signal, -> x[0..n-1].
 * Includes the 1/n scaling, so it inverts aec_rfft_exec.
 *
 * PyTorch reaches the same result with `ifft(to_full(X)).real`; padding the
 * one-sided spectrum out to a Hermitian full spectrum and taking the ordinary
 * inverse transform is by definition the real inverse transform, so both
 * routes agree. */
void aec_irfft_exec(AecRfft* p, const float* re, const float* im, float* x);

#endif /* AEC_FFT_H */

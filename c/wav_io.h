/* Minimal RIFF/WAVE reader and writer -- enough for the formats this needs
 * (16-bit PCM and 32-bit float), with no dependency beyond libc. */
#ifndef AEC_WAV_IO_H
#define AEC_WAV_IO_H

#include <stddef.h>

/* Reads a WAVE file into interleaved float samples in [-1, 1]. On success
 * `*samples` points at a malloc'd buffer of `*frames * *channels` values that
 * the caller owns. Returns 0 on success, non-zero on failure with a message in
 * `err`. */
int wav_read(const char* path, float** samples, size_t* frames, int* channels,
             int* sample_rate, char* err, int errlen);

/* Writes interleaved float samples in [-1, 1] as 16-bit PCM. */
int wav_write(const char* path, const float* samples, size_t count,
              int channels, int sample_rate, char* err, int errlen);

#endif /* AEC_WAV_IO_H */

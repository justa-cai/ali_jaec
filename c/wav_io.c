#include "wav_io.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void seterr(char* err, int errlen, const char* fmt, ...) {
  if (!err || errlen <= 0) return;
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(err, (size_t)errlen, fmt, ap);
  va_end(ap);
}

static uint32_t rd32(const unsigned char* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

static uint16_t rd16(const unsigned char* p) {
  return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static void wr16(unsigned char* p, uint16_t v) {
  p[0] = (unsigned char)(v & 0xff);
  p[1] = (unsigned char)((v >> 8) & 0xff);
}

static void wr32(unsigned char* p, uint32_t v) {
  for (int i = 0; i < 4; ++i) p[i] = (unsigned char)((v >> (8 * i)) & 0xff);
}

/* Reads the whole file. Returns a malloc'd buffer, or NULL. */
static unsigned char* slurp(const char* path, size_t* size, char* err,
                            int errlen) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    seterr(err, errlen, "cannot open %s", path);
    return NULL;
  }
  if (fseek(f, 0, SEEK_END) != 0) {
    seterr(err, errlen, "cannot seek in %s", path);
    fclose(f);
    return NULL;
  }
  const long n = ftell(f);
  if (n <= 0) {
    seterr(err, errlen, "empty file: %s", path);
    fclose(f);
    return NULL;
  }
  rewind(f);
  unsigned char* buf = (unsigned char*)malloc((size_t)n);
  if (!buf) {
    seterr(err, errlen, "out of memory reading %s", path);
    fclose(f);
    return NULL;
  }
  if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
    seterr(err, errlen, "short read: %s", path);
    free(buf);
    fclose(f);
    return NULL;
  }
  fclose(f);
  *size = (size_t)n;
  return buf;
}

int wav_read(const char* path, float** samples, size_t* frames_out,
             int* channels, int* sample_rate, char* err, int errlen) {
  size_t size = 0;
  unsigned char* buf = slurp(path, &size, err, errlen);
  if (!buf) return 1;

  if (size < 12 || memcmp(buf, "RIFF", 4) != 0 ||
      memcmp(buf + 8, "WAVE", 4) != 0) {
    seterr(err, errlen, "not a RIFF/WAVE file: %s", path);
    free(buf);
    return 1;
  }

  unsigned format = 0, bits = 0, nch = 0;
  uint32_t rate = 0;
  size_t data_off = 0, data_len = 0;

  size_t pos = 12;
  while (pos + 8 <= size) {
    const unsigned char* ck = buf + pos;
    const uint32_t ck_len = rd32(ck + 4);
    const size_t body = pos + 8;
    if (memcmp(ck, "fmt ", 4) == 0 && ck_len >= 16 && body + 16 <= size) {
      format = rd16(buf + body);
      nch = rd16(buf + body + 2);
      rate = rd32(buf + body + 4);
      bits = rd16(buf + body + 14);
    } else if (memcmp(ck, "data", 4) == 0) {
      data_off = body;
      data_len = (body + ck_len <= size) ? ck_len : (size - body);
    }
    pos = body + ck_len + (ck_len & 1u);      /* chunks are word aligned */
  }

  if (!data_off || !nch || !rate) {
    seterr(err, errlen, "missing fmt/data chunk in %s", path);
    free(buf);
    return 1;
  }
  if (format == 0xFFFE) format = 1;           /* WAVE_FORMAT_EXTENSIBLE: PCM */
  if (!((format == 1 && bits == 16) || (format == 3 && bits == 32))) {
    seterr(err, errlen, "unsupported WAV format in %s (need PCM16 or float32)",
           path);
    free(buf);
    return 1;
  }

  const size_t frame_bytes = (size_t)bits / 8 * nch;
  const size_t frames = data_len / frame_bytes;
  float* out = (float*)malloc(sizeof(float) * frames * nch);
  if (!out) {
    seterr(err, errlen, "out of memory (%zu frames from %s)", frames, path);
    free(buf);
    return 1;
  }
  const unsigned char* p = buf + data_off;
  if (format == 1) {
    for (size_t i = 0; i < frames * nch; ++i)
      out[i] = (float)(int16_t)rd16(p + i * 2) / 32768.0f;
  } else {
    for (size_t i = 0; i < frames * nch; ++i) {
      const uint32_t u = rd32(p + i * 4);
      memcpy(&out[i], &u, 4);
    }
  }
  free(buf);
  *samples = out;
  *frames_out = frames;
  *channels = (int)nch;
  *sample_rate = (int)rate;
  return 0;
}

int wav_write(const char* path, const float* samples, size_t count,
              int channels, int sample_rate, char* err, int errlen) {
  const uint32_t data_bytes = (uint32_t)(count * 2);
  const size_t total = 44 + data_bytes;
  unsigned char* out = (unsigned char*)malloc(total);
  if (!out) {
    seterr(err, errlen, "out of memory writing %s", path);
    return 1;
  }
  const uint32_t byte_rate = (uint32_t)(sample_rate * channels * 2);
  memcpy(out, "RIFF", 4);
  wr32(out + 4, 36 + data_bytes);
  memcpy(out + 8, "WAVE", 4);
  memcpy(out + 12, "fmt ", 4);
  wr32(out + 16, 16);                          /* fmt chunk size */
  wr16(out + 20, 1);                           /* PCM */
  wr16(out + 22, (uint16_t)channels);
  wr32(out + 24, (uint32_t)sample_rate);
  wr32(out + 28, byte_rate);
  wr16(out + 32, (uint16_t)(channels * 2));    /* block align */
  wr16(out + 34, 16);                          /* bits per sample */
  memcpy(out + 36, "data", 4);
  wr32(out + 40, data_bytes);

  for (size_t i = 0; i < count; ++i) {
    float v = samples[i];
    if (v < -1.0f) v = -1.0f;
    if (v > 1.0f) v = 1.0f;
    int s = (int)lroundf(v * 32767.0f);
    if (s > 32767) s = 32767;
    if (s < -32768) s = -32768;
    wr16(out + 44 + 2 * i, (uint16_t)(int16_t)s);
  }

  FILE* f = fopen(path, "wb");
  if (!f) {
    seterr(err, errlen, "cannot write %s", path);
    free(out);
    return 1;
  }
  const size_t wrote = fwrite(out, 1, total, f);
  fclose(f);
  free(out);
  if (wrote != total) {
    seterr(err, errlen, "short write: %s", path);
    return 1;
  }
  return 0;
}

#include "wav_io.h"

#include <cstdio>
#include <cstring>
#include <cmath>

namespace wav {
namespace {

uint32_t rd32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}
uint16_t rd16(const uint8_t* p) {
  return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}
void wr32(std::vector<uint8_t>& v, uint32_t x) {
  v.push_back((uint8_t)(x & 0xff));
  v.push_back((uint8_t)((x >> 8) & 0xff));
  v.push_back((uint8_t)((x >> 16) & 0xff));
  v.push_back((uint8_t)((x >> 24) & 0xff));
}
void wr16(std::vector<uint8_t>& v, uint16_t x) {
  v.push_back((uint8_t)(x & 0xff));
  v.push_back((uint8_t)((x >> 8) & 0xff));
}

}  // namespace

bool read(const std::string& path, std::vector<float>& samples, int& channels,
          int& sample_rate, std::string& error) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    error = "cannot open " + path;
    return false;
  }
  std::vector<uint8_t> buf;
  std::fseek(f, 0, SEEK_END);
  long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (size <= 0) {
    std::fclose(f);
    error = "empty file: " + path;
    return false;
  }
  buf.resize((size_t)size);
  if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
    std::fclose(f);
    error = "short read: " + path;
    return false;
  }
  std::fclose(f);

  if (buf.size() < 12 || std::memcmp(buf.data(), "RIFF", 4) != 0 ||
      std::memcmp(buf.data() + 8, "WAVE", 4) != 0) {
    error = "not a RIFF/WAVE file: " + path;
    return false;
  }

  uint16_t format = 0, bits = 0, nch = 0;
  uint32_t rate = 0;
  size_t data_off = 0, data_len = 0;

  size_t pos = 12;
  while (pos + 8 <= buf.size()) {
    const uint8_t* ck = buf.data() + pos;
    uint32_t ck_len = rd32(ck + 4);
    size_t body = pos + 8;
    if (std::memcmp(ck, "fmt ", 4) == 0 && ck_len >= 16 && body + 16 <= buf.size()) {
      format = rd16(buf.data() + body);
      nch = rd16(buf.data() + body + 2);
      rate = rd32(buf.data() + body + 4);
      bits = rd16(buf.data() + body + 14);
    } else if (std::memcmp(ck, "data", 4) == 0) {
      data_off = body;
      data_len = (body + ck_len <= buf.size()) ? ck_len : (buf.size() - body);
    }
    pos = body + ck_len + (ck_len & 1u);  // chunks are word aligned
  }

  if (!data_off || !nch || !rate) {
    error = "missing fmt/data chunk in " + path;
    return false;
  }
  if (format == 0xFFFE) format = 1;  // WAVE_FORMAT_EXTENSIBLE: assume PCM
  if (!((format == 1 && bits == 16) || (format == 3 && bits == 32))) {
    error = "unsupported WAV format (need PCM16 or float32)";
    return false;
  }

  const size_t frame_bytes = (size_t)bits / 8 * nch;
  const size_t frames = data_len / frame_bytes;
  samples.resize(frames * nch);
  const uint8_t* p = buf.data() + data_off;
  if (format == 1) {
    for (size_t i = 0; i < frames * nch; ++i)
      samples[i] = (float)(int16_t)rd16(p + i * 2) / 32768.0f;
  } else {
    for (size_t i = 0; i < frames * nch; ++i) {
      uint32_t u = rd32(p + i * 4);
      float v;
      std::memcpy(&v, &u, 4);
      samples[i] = v;
    }
  }
  channels = nch;
  sample_rate = (int)rate;
  return true;
}

bool write(const std::string& path, const std::vector<float>& samples,
           int channels, int sample_rate, std::string& error) {
  std::vector<uint8_t> out;
  const uint32_t data_bytes = (uint32_t)(samples.size() * 2);
  out.reserve(data_bytes + 44);
  out.insert(out.end(), {'R', 'I', 'F', 'F'});
  wr32(out, 36 + data_bytes);
  out.insert(out.end(), {'W', 'A', 'V', 'E'});
  out.insert(out.end(), {'f', 'm', 't', ' '});
  wr32(out, 16);
  wr16(out, 1);                                     // PCM
  wr16(out, (uint16_t)channels);
  wr32(out, (uint32_t)sample_rate);
  wr32(out, (uint32_t)(sample_rate * channels * 2));  // byte rate
  wr16(out, (uint16_t)(channels * 2));               // block align
  wr16(out, 16);                                     // bits per sample
  out.insert(out.end(), {'d', 'a', 't', 'a'});
  wr32(out, data_bytes);
  for (float v : samples) {
    float c = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
    int s = (int)std::lround(c * 32767.0f);
    if (s > 32767) s = 32767;
    if (s < -32768) s = -32768;
    wr16(out, (uint16_t)(int16_t)s);
  }

  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) {
    error = "cannot write " + path;
    return false;
  }
  size_t wrote = std::fwrite(out.data(), 1, out.size(), f);
  std::fclose(f);
  if (wrote != out.size()) {
    error = "short write: " + path;
    return false;
  }
  return true;
}

}  // namespace wav

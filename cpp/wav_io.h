// Minimal RIFF/WAVE reader and writer -- enough for the formats this demo
// needs (16-bit PCM and 32-bit float), with no dependency beyond the standard
// library.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace wav {

// Reads a WAVE file into interleaved float samples in [-1, 1].
// Returns false and fills `error` on failure.
bool read(const std::string& path, std::vector<float>& samples,
          int& channels, int& sample_rate, std::string& error);

// Writes interleaved float samples in [-1, 1] as 16-bit PCM.
bool write(const std::string& path, const std::vector<float>& samples,
           int channels, int sample_rate, std::string& error);

}  // namespace wav

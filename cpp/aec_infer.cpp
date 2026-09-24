// Command-line acoustic echo canceller driven by the exported ONNX graph.
//
//   ./aec_infer --mic nearend_mic.wav --ref farend_speech.wav --out out.wav
//   ./aec_infer --input demo_3ch.wav --out out.wav [--three-channel]
//
// Input channels are read in order: ch0 = near-end microphone, ch1 = far-end
// reference. A 3-channel 近端/远端/算法后 file can therefore be fed straight in.
// The output is time-aligned with the microphone; the front-end adds no delay.
//
// The graph has static shapes, so the recording is cut into fixed-length
// segments with a 50 % overlap and raised-cosine weighted back together, which
// hides the fact that the recurrent network's context restarts at a boundary.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "wav_io.h"

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kSampleRate = 16000;
constexpr int kSegment = 48000;   // must match --segment used at export time

struct Options {
  std::string input, mic, ref, out = "aec_out.wav", model;
  bool three_channel = false;
  int segment = kSegment;
};

void usage() {
  std::printf(
      "usage: aec_infer (--input FILE | --mic FILE --ref FILE) [options]\n"
      "\n"
      "  --input FILE    multi-channel wav: ch0 = near-end mic, ch1 = far-end ref\n"
      "  --mic FILE      near-end microphone wav\n"
      "  --ref FILE      far-end reference wav\n"
      "  --out FILE      output wav (default aec_out.wav)\n"
      "  --three-channel write mic / ref / output instead of mono output\n"
      "  --model FILE    ONNX model (default weights/aec_lp.onnx)\n"
      "  --segment N     samples per inference call (default %d)\n",
      kSegment);
}

bool parse(int argc, char** argv, Options& o) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](std::string& dst) {
      if (i + 1 >= argc) return false;
      dst = argv[++i];
      return true;
    };
    if (a == "-h" || a == "--help") { usage(); std::exit(0); }
    else if (a == "--input") { if (!next(o.input)) return false; }
    else if (a == "--mic") { if (!next(o.mic)) return false; }
    else if (a == "--ref") { if (!next(o.ref)) return false; }
    else if (a == "--out") { if (!next(o.out)) return false; }
    else if (a == "--model") { if (!next(o.model)) return false; }
    else if (a == "--segment") { if (!next(a)) return false; o.segment = std::atoi(a.c_str()); }
    else if (a == "--three-channel") { o.three_channel = true; }
    else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); return false; }
  }
  if (o.input.empty() && (o.mic.empty() || o.ref.empty())) return false;
  if (o.segment <= 0) return false;
  return true;
}

// Keeps channel 0 of an interleaved signal.
std::vector<float> first_channel(const std::vector<float>& x, int channels) {
  std::vector<float> y(x.size() / channels);
  for (size_t i = 0; i < y.size(); ++i) y[i] = x[i * channels];
  return y;
}

// Linear resampling. Fine for rounding a stray 44.1/48 kHz file up to the
// model's rate; not a high-quality conversion.
std::vector<float> resample(const std::vector<float>& x, int sr_in, int sr_out) {
  if (sr_in == sr_out || x.empty()) return x;
  const size_t n_out = (size_t)std::llround((double)x.size() * sr_out / sr_in);
  std::vector<float> y(n_out);
  const double step = (double)(x.size() - 1) / (double)std::max<size_t>(n_out - 1, 1);
  for (size_t i = 0; i < n_out; ++i) {
    double p = i * step;
    size_t i0 = (size_t)p;
    size_t i1 = std::min(i0 + 1, x.size() - 1);
    double t = p - (double)i0;
    y[i] = (float)(x[i0] * (1.0 - t) + x[i1] * t);
  }
  return y;
}

// Hann-like weights, strictly positive so the overlap-add denominator is never
// zero at the first and last sample.
std::vector<float> crossfade_window(int n) {
  std::vector<float> w(n);
  for (int i = 0; i < n; ++i)
    w[i] = (float)(0.5 - 0.5 * std::cos(2.0 * kPi * (i + 1) / (n + 1)));
  return w;
}

class Engine {
 public:
  Engine(const std::string& model_path, int segment)
      : env_(ORT_LOGGING_LEVEL_WARNING, "aec"), segment_(segment) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(0);           // 0 = let ORT pick
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts);
    if (session_->GetInputCount() != 2 || session_->GetOutputCount() != 2)
      throw std::runtime_error("expected 2 inputs and 2 outputs in " + model_path);
  }

  // Returns the enhanced segment; `tau` receives the estimated delay.
  std::vector<float> run(const float* mic, const float* ref, float& tau) {
    const int64_t shape[2] = {1, (int64_t)segment_};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in[2] = {
        Ort::Value::CreateTensor<float>(mem, const_cast<float*>(mic), segment_, shape, 2),
        Ort::Value::CreateTensor<float>(mem, const_cast<float*>(ref), segment_, shape, 2)};

    const char* in_names[] = {"mic", "ref"};
    const char* out_names[] = {"out", "tau"};
    auto outs = session_->Run(Ort::RunOptions{nullptr}, in_names, in, 2, out_names, 2);

    const float* y = outs[0].GetTensorData<float>();
    std::vector<float> result(y, y + segment_);
    tau = outs[1].GetTensorData<float>()[0];
    return result;
  }

  int segment() const { return segment_; }

 private:
  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  int segment_;
};

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  if (!parse(argc, argv, opt)) {
    usage();
    return 2;
  }

  std::string err;
  std::vector<float> mic, ref;
  int sr_mic = kSampleRate, sr_ref = kSampleRate;

  if (!opt.input.empty()) {
    std::vector<float> x;
    int ch = 0, sr = 0;
    if (!wav::read(opt.input, x, ch, sr, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    if (ch < 2) { std::fprintf(stderr, "%s needs at least 2 channels, has %d\n", opt.input.c_str(), ch); return 1; }
    const size_t frames = x.size() / ch;
    mic.resize(frames); ref.resize(frames);
    for (size_t i = 0; i < frames; ++i) { mic[i] = x[i * ch]; ref[i] = x[i * ch + 1]; }
    sr_mic = sr_ref = sr;
    std::printf("input : %s  %d ch @ %d Hz -> %.2f s\n", opt.input.c_str(), ch, sr,
                (double)frames / kSampleRate);
  } else {
    int ch = 0;
    if (!wav::read(opt.mic, mic, ch, sr_mic, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    if (ch > 1) mic = first_channel(mic, ch);
    if (!wav::read(opt.ref, ref, ch, sr_ref, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    if (ch > 1) ref = first_channel(ref, ch);
    std::printf("input : %s + %s @ %d/%d Hz -> %.2f s\n", opt.mic.c_str(), opt.ref.c_str(),
                sr_mic, sr_ref, (double)std::min(mic.size(), ref.size()) / kSampleRate);
  }

  mic = resample(mic, sr_mic, kSampleRate);
  ref = resample(ref, sr_ref, kSampleRate);
  const size_t n = std::min(mic.size(), ref.size());
  mic.resize(n); ref.resize(n);
  if (n == 0) { std::fprintf(stderr, "empty input\n"); return 1; }

  std::string model_path = opt.model;
  if (model_path.empty()) model_path = "weights/aec_lp.onnx";

  Engine engine(model_path, opt.segment);
  std::printf("model : %s (segment %d)\n", model_path.c_str(), engine.segment());

  const int seg = engine.segment();
  std::vector<float> out(n, 0.0f);

  if ((int)n <= seg) {
    std::vector<float> pm(seg, 0.0f), pr(seg, 0.0f);
    std::copy(mic.begin(), mic.end(), pm.begin());
    std::copy(ref.begin(), ref.end(), pr.begin());
    float tau = 0.0f;
    auto y = engine.run(pm.data(), pr.data(), tau);
    std::copy(y.begin(), y.begin() + n, out.begin());
    std::printf("  one segment, estimated delay %.0f samples\n", tau);
  } else {
    const int hop = seg / 2;
    const std::vector<float> w = crossfade_window(seg);
    std::vector<double> num(n + seg, 0.0), den(n + seg, 0.0);
    std::vector<float> pm(seg), pr(seg);
    std::vector<float> taus;
    // hop-aligned starts, with the final segment right-aligned to the end
    std::vector<size_t> starts;
    for (size_t s = 0; s + seg < n; s += hop) starts.push_back(s);
    starts.push_back(n - seg);
    for (size_t start : starts) {
      std::copy(mic.begin() + start, mic.begin() + start + seg, pm.begin());
      std::copy(ref.begin() + start, ref.begin() + start + seg, pr.begin());
      float tau = 0.0f;
      auto y = engine.run(pm.data(), pr.data(), tau);
      taus.push_back(tau);
      for (int i = 0; i < seg; ++i) {
        num[start + i] += (double)w[i] * y[i];
        den[start + i] += w[i];
      }
    }
    std::sort(taus.begin(), taus.end());
    std::printf("  %zu segments of %.1f s, estimated delay %.0f samples\n", starts.size(),
                (double)seg / kSampleRate, (double)taus[taus.size() / 2]);
    for (size_t i = 0; i < n; ++i) out[i] = (float)(num[i] / std::max(den[i], 1e-9));
  }

  auto rms_db = [](const std::vector<float>& v) {
    double acc = 0;
    for (float s : v) acc += (double)s * s;
    return 10.0 * std::log10(acc / std::max<size_t>(v.size(), 1) + 1e-20);
  };

  std::vector<float> data;
  if (opt.three_channel) {
    data.reserve(3 * n);
    data.insert(data.end(), mic.begin(), mic.end());
    data.insert(data.end(), ref.begin(), ref.end());
    data.insert(data.end(), out.begin(), out.end());
  } else {
    data = out;
  }
  if (!wav::write(opt.out, data, opt.three_channel ? 3 : 1, kSampleRate, err)) {
    std::fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  std::printf("output: %s  %d ch  %.2f s   mic %.1f dBFS -> out %.1f dBFS\n",
              opt.out.c_str(), opt.three_channel ? 3 : 1, (double)n / kSampleRate,
              rms_db(mic), rms_db(out));
  return 0;
}

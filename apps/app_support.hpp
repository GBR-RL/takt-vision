#pragma once

#include <memory>
#include <string>
#include <vector>

#include "takt/infer/backend.hpp"
#include "takt/pipeline/source_sink.hpp"

namespace takt::app {

struct BackendSettings {
  std::string kind;  // "onnx" or "fake"; empty = onnx if a model is given, else fake
  std::string model;
  std::string execution_provider = "cpu";
  int threads = 0;
  bool spin = false;
  int fake_input = 640;
  int fake_classes = 80;
  double fake_latency_ms = 15.0;
  double fake_jitter_ms = 0.0;
};

struct SourceSettings {
  std::string uri = "synthetic";
  bool realtime = true;
  bool loop = false;
  int synthetic_width = 1280;
  int synthetic_height = 720;
  double synthetic_fps = 30.0;
};

[[nodiscard]] Backend make_backend(const BackendSettings& settings);
[[nodiscard]] std::unique_ptr<FrameSource> make_source(const SourceSettings& settings);
// Features compiled into this build, e.g. "onnxruntime opencv".
[[nodiscard]] std::string build_features();
[[nodiscard]] bool has_opencv() noexcept;

}  // namespace takt::app

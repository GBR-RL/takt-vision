#include "app_support.hpp"

#include <chrono>
#include <stdexcept>

#include "takt/infer/fake_backend.hpp"
#include "takt/io/synthetic_source.hpp"

#ifdef TAKT_HAS_ONNXRUNTIME
#include "takt/infer/onnx_backend.hpp"
#endif
#ifdef TAKT_HAS_OPENCV
#include "takt/io/opencv_source.hpp"
#endif

namespace takt::app {
namespace {

std::chrono::microseconds from_ms(double ms) {
  return std::chrono::microseconds(static_cast<std::int64_t>(ms * 1000.0));
}

}  // namespace

Backend make_backend(const BackendSettings& settings) {
  const std::string kind =
      settings.kind.empty() ? (settings.model.empty() ? "fake" : "onnx") : settings.kind;
  if (kind == "fake") {
    FakeBackendOptions options;
    options.input_width = settings.fake_input;
    options.input_height = settings.fake_input;
    options.num_classes = settings.fake_classes;
    // Anchor count of a YOLO head at this input size: strides 8, 16 and 32.
    const int s = settings.fake_input;
    options.num_anchors = (s / 8) * (s / 8) + (s / 16) * (s / 16) + (s / 32) * (s / 32);
    options.latency = from_ms(settings.fake_latency_ms);
    options.jitter = from_ms(settings.fake_jitter_ms);
    return FakeBackend(options);
  }
  if (kind == "onnx") {
#ifdef TAKT_HAS_ONNXRUNTIME
    if (settings.model.empty())
      throw std::invalid_argument("--model is required for the onnx backend");
    OnnxBackendOptions options;
    options.model_path = settings.model;
    options.execution_provider = settings.execution_provider;
    options.intra_op_threads = settings.threads;
    options.allow_spinning = settings.spin;
    return OnnxBackend(options);
#else
    throw std::invalid_argument(
        "this build has no ONNX Runtime backend (TAKT_WITH_ONNXRUNTIME=OFF)");
#endif
  }
  throw std::invalid_argument("unknown backend '" + kind + "' (expected onnx or fake)");
}

std::unique_ptr<FrameSource> make_source(const SourceSettings& settings) {
  if (settings.uri == "synthetic") {
    SyntheticSourceOptions options;
    options.width = settings.synthetic_width;
    options.height = settings.synthetic_height;
    options.fps = settings.realtime ? settings.synthetic_fps : 0.0;
    return std::make_unique<SyntheticSource>(options);
  }
#ifdef TAKT_HAS_OPENCV
  OpenCvSourceOptions options;
  options.uri = settings.uri;
  options.realtime = settings.realtime;
  options.loop = settings.loop;
  return std::make_unique<OpenCvSource>(options);
#else
  throw std::invalid_argument("this build has no OpenCV; only --source synthetic is available");
#endif
}

std::string build_features() {
  std::string features = "core";
#ifdef TAKT_HAS_ONNXRUNTIME
  features += " onnxruntime";
#endif
#ifdef TAKT_HAS_OPENCV
  features += " opencv";
#endif
  return features;
}

bool has_opencv() noexcept {
#ifdef TAKT_HAS_OPENCV
  return true;
#else
  return false;
#endif
}

}  // namespace takt::app

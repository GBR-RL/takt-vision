#pragma once

#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "takt/infer/backend.hpp"

namespace takt {

struct OnnxBackendOptions {
  std::filesystem::path model_path;
  std::string execution_provider = "cpu";  // "cpu" or "cuda" (needs a GPU build of ONNX Runtime)
  int intra_op_threads = 0;                // 0 = ONNX Runtime default (one per physical core)
  // ONNX Runtime's worker threads spin-wait between runs by default. That is great for a benchmark
  // loop and bad for a pipeline: spinning threads steal the cores that preprocessing, tracking and
  // the camera need. Off by default here; measure both with takt_bench.
  bool allow_spinning = false;
  // Used only if the model has a dynamic spatial input size.
  int default_input_size = 640;
};

// ONNX Runtime inference over caller-owned buffers: input and output tensors are bound directly
// to the pipeline frame's memory, so no tensor data is copied or allocated per inference.
class OnnxBackend {
 public:
  explicit OnnxBackend(const OnnxBackendOptions& options);
  ~OnnxBackend();
  OnnxBackend(OnnxBackend&&) noexcept;
  OnnxBackend& operator=(OnnxBackend&&) noexcept;
  OnnxBackend(const OnnxBackend&) = delete;
  OnnxBackend& operator=(const OnnxBackend&) = delete;

  [[nodiscard]] std::string_view name() const noexcept;
  [[nodiscard]] const TensorSpec& input_spec() const noexcept;
  [[nodiscard]] const TensorSpec& output_spec() const noexcept;
  void infer(std::span<const float> input, std::span<float> output);

 private:
  struct Impl;  // pimpl keeps onnxruntime_cxx_api.h out of every translation unit that uses this
  std::unique_ptr<Impl> impl_;
};

static_assert(InferenceBackend<OnnxBackend>);

}  // namespace takt

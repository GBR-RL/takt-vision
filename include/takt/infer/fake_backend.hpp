#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <random>
#include <span>
#include <string_view>

#include "takt/infer/backend.hpp"

namespace takt {

struct FakeBackendOptions {
  int input_width = 640;
  int input_height = 640;
  int num_classes = 80;
  int num_anchors = 8400;
  int num_objects = 3;                   // objects circling the frame centre
  std::chrono::microseconds latency{0};  // simulated forward-pass time
  std::chrono::microseconds jitter{0};   // uniform extra time in [0, jitter]
  std::uint64_t seed = 42;
};

// Deterministic stand-in for a YOLO model: emits a raw [1, 4 + classes, anchors] head containing
// `num_objects` confident boxes moving on a circle, after sleeping for a configurable latency.
//
// It makes the pipeline testable and benchmarkable without model weights or a GPU: tests use it
// to check ordering, back-pressure and tracking; the overload demo uses its latency knob to
// simulate a detector that cannot keep up with the camera.
class FakeBackend {
 public:
  explicit FakeBackend(FakeBackendOptions options = {});

  [[nodiscard]] std::string_view name() const noexcept { return "fake"; }
  [[nodiscard]] const TensorSpec& input_spec() const noexcept { return input_spec_; }
  [[nodiscard]] const TensorSpec& output_spec() const noexcept { return output_spec_; }
  void infer(std::span<const float> input, std::span<float> output);

  // Box emitted for object `index` on call number `call` (model-input pixels). Exposed for tests.
  [[nodiscard]] static std::array<float, 4> object_box(const FakeBackendOptions& options,
                                                       std::uint64_t call, int index) noexcept;

 private:
  FakeBackendOptions options_;
  TensorSpec input_spec_;
  TensorSpec output_spec_;
  std::uint64_t calls_ = 0;
  std::mt19937_64 rng_;
};

static_assert(InferenceBackend<FakeBackend>);

}  // namespace takt

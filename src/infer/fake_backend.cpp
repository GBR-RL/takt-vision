#include "takt/infer/fake_backend.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <thread>

#include "takt/core/clock.hpp"

namespace takt {

FakeBackend::FakeBackend(FakeBackendOptions options)
    : options_(options),
      input_spec_{"images", {1, 3, options.input_height, options.input_width}},
      output_spec_{"output0", {1, 4 + options.num_classes, options.num_anchors}},
      rng_(options.seed) {
  if (options.input_width <= 0 || options.input_height <= 0 || options.num_classes <= 0 ||
      options.num_anchors <= 0 || options.num_objects < 0 ||
      options.num_objects > options.num_anchors) {
    throw std::invalid_argument("FakeBackend: invalid options");
  }
}

std::array<float, 4> FakeBackend::object_box(const FakeBackendOptions& options, std::uint64_t call,
                                             int index) noexcept {
  constexpr double kFramesPerRevolution = 240.0;
  const double phase = static_cast<double>(call) / kFramesPerRevolution +
                       static_cast<double>(index) / std::max(1, options.num_objects);
  const double angle = 2.0 * std::numbers::pi * phase;
  const double w = options.input_width;
  const double h = options.input_height;
  const double cx = w / 2 + 0.3 * w * std::cos(angle);
  const double cy = h / 2 + 0.3 * h * std::sin(angle);
  const double bw = 0.08 * w + 0.02 * w * index;
  const double bh = 0.12 * h;
  return {static_cast<float>(cx), static_cast<float>(cy), static_cast<float>(bw),
          static_cast<float>(bh)};
}

void FakeBackend::infer(std::span<const float> input, std::span<float> output) {
  const TimePoint start = now();
  const std::size_t in_count = input_spec_.element_count();
  const std::size_t out_count = output_spec_.element_count();
  if (input.size() < in_count || output.size() < out_count) {
    throw std::invalid_argument("FakeBackend::infer: tensor size mismatch");
  }

  const auto anchors = static_cast<std::size_t>(options_.num_anchors);
  std::fill(output.begin(), output.begin() + static_cast<std::ptrdiff_t>(out_count), 0.0f);
  for (int k = 0; k < options_.num_objects; ++k) {
    // Spread objects over distinct anchors; box rows are cx, cy, w, h.
    const std::size_t anchor =
        static_cast<std::size_t>(k) * (anchors / static_cast<std::size_t>(options_.num_objects));
    const auto box = object_box(options_, calls_, k);
    for (std::size_t i = 0; i < 4; ++i) output[i * anchors + anchor] = box[i];
    const auto class_row = 4 + static_cast<std::size_t>(k % options_.num_classes);
    output[class_row * anchors + anchor] = 0.9f;
  }
  ++calls_;

  auto busy_until = start + options_.latency;
  if (options_.jitter.count() > 0) {
    std::uniform_int_distribution<std::int64_t> extra(0, options_.jitter.count());
    busy_until += std::chrono::microseconds(extra(rng_));
  }
  std::this_thread::sleep_until(busy_until);
}

}  // namespace takt

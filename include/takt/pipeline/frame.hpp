#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "takt/core/clock.hpp"
#include "takt/core/image.hpp"
#include "takt/track/byte_tracker.hpp"
#include "takt/vision/detection.hpp"
#include "takt/vision/letterbox.hpp"

namespace takt {

enum class StageId : std::uint8_t {
  kPreprocess,
  kInference,
  kPostprocess,
  kTracking,
  kSink,
  kCount
};

inline constexpr std::size_t kStageCount = static_cast<std::size_t>(StageId::kCount);

[[nodiscard]] constexpr std::size_t to_index(StageId id) noexcept {
  return static_cast<std::size_t>(id);
}

[[nodiscard]] constexpr std::string_view stage_name(StageId id) noexcept {
  switch (id) {
    case StageId::kPreprocess: return "preprocess";
    case StageId::kInference: return "inference";
    case StageId::kPostprocess: return "postprocess";
    case StageId::kTracking: return "tracking";
    case StageId::kSink: return "sink";
    case StageId::kCount: break;
  }
  return "unknown";
}

// Everything that travels through the pipeline for one camera frame. Frames are pooled and
// recycled, and every buffer keeps its capacity across reuse, so once the pool is warm the
// pipeline performs no heap allocation per frame.
struct Frame {
  std::uint64_t sequence = 0;
  TimePoint captured_at{};
  Image image;                // source pixels (BGR)
  std::vector<float> input;   // model input tensor, NCHW
  std::vector<float> output;  // raw model output
  LetterboxTransform letterbox{};
  std::vector<Detection> detections;  // after NMS, in source-image pixels
  std::vector<TrackedObject> tracks;
  std::array<Nanos, kStageCount> service_time{};  // time spent inside each stage

  void reset() noexcept {
    sequence = 0;
    captured_at = {};
    detections.clear();
    tracks.clear();
    service_time.fill(Nanos::zero());
  }
};

}  // namespace takt

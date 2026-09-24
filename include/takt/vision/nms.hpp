#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "takt/vision/detection.hpp"
#include "takt/vision/letterbox.hpp"

namespace takt {

// Defaults match Ultralytics predict(): iou=0.7, max_det=300, max_nms=30000, class-aware.
struct NmsOptions {
  float iou_threshold = 0.7f;
  std::size_t max_detections = 300;
  std::size_t max_candidates = 30000;
  bool class_agnostic = false;
};

// Greedy non-maximum suppression, same semantics as torchvision.ops.nms with per-class offsets:
// a box is suppressed when its IoU with a higher-scoring kept box of the same class is strictly
// greater than the threshold. Scratch memory is kept between calls, so steady-state calls do not
// allocate.
class Nms {
 public:
  explicit Nms(NmsOptions options = {}) : options_(options) {}

  // Filters `detections` in place; survivors are sorted by descending score.
  void apply(std::vector<Detection>& detections);

  [[nodiscard]] const NmsOptions& options() const noexcept { return options_; }

 private:
  NmsOptions options_;
  std::vector<std::uint8_t> suppressed_;
};

// Maps detections from model-input pixels back to source-image pixels.
void scale_to_source(std::span<Detection> detections, const LetterboxTransform& transform) noexcept;

}  // namespace takt

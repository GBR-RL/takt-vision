#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "takt/vision/detection.hpp"

namespace takt {

// Layout of an anchor-free YOLO head (YOLOv8 / YOLO11): per anchor, 4 box values (cx, cy, w, h
// in model-input pixels) followed by one score per class. There is no objectness term.
struct YoloLayout {
  std::size_t num_anchors = 0;
  std::size_t num_classes = 0;
  // true for Ultralytics' default export [1, 4 + classes, anchors]: each attribute is a
  // contiguous row across all anchors. false for [1, anchors, 4 + classes].
  bool channels_first = true;
};

// Infers the layout from an output tensor shape such as {1, 84, 8400}. Throws for anything that is
// not a raw (pre-NMS) YOLO head, e.g. end-to-end exports whose output is already [1, 300, 6].
[[nodiscard]] YoloLayout yolo_layout_from_shape(std::span<const std::int64_t> shape);

class YoloDecoder {
 public:
  explicit YoloDecoder(YoloLayout layout);

  // Replaces `out` with every anchor whose best class score exceeds `score_threshold`. Boxes are
  // in model-input pixels; run NMS and LetterboxTransform::to_source afterwards.
  void decode(std::span<const float> raw, float score_threshold, std::vector<Detection>& out);

  [[nodiscard]] const YoloLayout& layout() const noexcept { return layout_; }

 private:
  YoloLayout layout_;
  std::vector<float> best_score_;
  std::vector<int> best_class_;
};

}  // namespace takt

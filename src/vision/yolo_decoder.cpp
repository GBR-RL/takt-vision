#include "takt/vision/yolo_decoder.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace takt {

YoloLayout yolo_layout_from_shape(std::span<const std::int64_t> shape) {
  // Drop leading batch dimensions of size 1.
  while (shape.size() > 2 && shape.front() == 1) shape = shape.subspan(1);
  if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0) {
    throw std::invalid_argument("yolo_layout_from_shape: expected [1, 4 + classes, anchors]");
  }
  // A detection head always has far more anchors than attributes (8400 vs 84 at 640x640).
  const bool channels_first = shape[0] < shape[1];
  const auto attributes = static_cast<std::size_t>(channels_first ? shape[0] : shape[1]);
  const auto anchors = static_cast<std::size_t>(channels_first ? shape[1] : shape[0]);
  if (attributes < 5) {
    throw std::invalid_argument(
        "yolo_layout_from_shape: need at least 4 box values + 1 class, got " +
        std::to_string(attributes) + " (end-to-end / post-NMS exports are not supported)");
  }
  return {anchors, attributes - 4, channels_first};
}

YoloDecoder::YoloDecoder(YoloLayout layout)
    : layout_(layout), best_score_(layout.num_anchors), best_class_(layout.num_anchors) {
  if (layout.num_anchors == 0 || layout.num_classes == 0) {
    throw std::invalid_argument("YoloDecoder: empty layout");
  }
}

void YoloDecoder::decode(std::span<const float> raw, float score_threshold,
                         std::vector<Detection>& out) {
  const std::size_t n = layout_.num_anchors;
  const std::size_t nc = layout_.num_classes;
  if (raw.size() < n * (4 + nc))
    throw std::invalid_argument("YoloDecoder: output tensor too small");
  out.clear();

  if (layout_.channels_first) {
    // Class-major scan: walk each class row front to back so every read is sequential and the
    // inner loop vectorises. The naive anchor-major loop would stride by n floats (33 KB) per
    // class and miss the cache on every access.
    const float* first_class = raw.data() + 4 * n;
    std::copy(first_class, first_class + n, best_score_.begin());
    std::fill(best_class_.begin(), best_class_.end(), 0);
    for (std::size_t c = 1; c < nc; ++c) {
      const float* scores = raw.data() + (4 + c) * n;
      const int class_id = static_cast<int>(c);
      for (std::size_t a = 0; a < n; ++a) {
        if (scores[a] > best_score_[a]) {
          best_score_[a] = scores[a];
          best_class_[a] = class_id;
        }
      }
    }
    const float* cx = raw.data();
    const float* cy = cx + n;
    const float* w = cy + n;
    const float* h = w + n;
    for (std::size_t a = 0; a < n; ++a) {
      if (best_score_[a] > score_threshold) {
        out.push_back({Box::from_center(cx[a], cy[a], w[a], h[a]), best_score_[a], best_class_[a]});
      }
    }
    return;
  }

  const std::size_t stride = 4 + nc;
  for (std::size_t a = 0; a < n; ++a) {
    const float* row = raw.data() + a * stride;
    const float* best = std::max_element(row + 4, row + stride);
    if (*best > score_threshold) {
      out.push_back({Box::from_center(row[0], row[1], row[2], row[3]), *best,
                     static_cast<int>(best - (row + 4))});
    }
  }
}

}  // namespace takt

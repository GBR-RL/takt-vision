#include "takt/vision/nms.hpp"

#include <algorithm>

namespace takt {

void Nms::apply(std::vector<Detection>& detections) {
  std::sort(detections.begin(), detections.end(),
            [](const Detection& a, const Detection& b) { return a.score > b.score; });
  if (detections.size() > options_.max_candidates) detections.resize(options_.max_candidates);

  const std::size_t n = detections.size();
  suppressed_.assign(n, 0);  // reuses capacity after the first few frames

  std::size_t kept = 0;
  for (std::size_t i = 0; i < n && kept < options_.max_detections; ++i) {
    if (suppressed_[i] != 0) continue;
    const Detection current = detections[i];
    for (std::size_t j = i + 1; j < n; ++j) {
      if (suppressed_[j] != 0) continue;
      if (!options_.class_agnostic && detections[j].class_id != current.class_id) continue;
      if (iou(current.box, detections[j].box) > options_.iou_threshold) suppressed_[j] = 1;
    }
    // Compacting in place is safe: kept <= i, and only indices > i are read from now on.
    detections[kept++] = current;
  }
  detections.resize(kept);
}

void scale_to_source(std::span<Detection> detections,
                     const LetterboxTransform& transform) noexcept {
  for (Detection& d : detections) d.box = transform.to_source(d.box);
}

}  // namespace takt

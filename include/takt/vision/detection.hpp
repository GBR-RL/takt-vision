#pragma once

#include <algorithm>

namespace takt {

// Axis-aligned box in corner form (x1, y1) - (x2, y2), in pixels.
struct Box {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;

  [[nodiscard]] static constexpr Box from_center(float cx, float cy, float w, float h) noexcept {
    return {cx - 0.5f * w, cy - 0.5f * h, cx + 0.5f * w, cy + 0.5f * h};
  }

  [[nodiscard]] constexpr float width() const noexcept { return x2 - x1; }
  [[nodiscard]] constexpr float height() const noexcept { return y2 - y1; }
  [[nodiscard]] constexpr float center_x() const noexcept { return 0.5f * (x1 + x2); }
  [[nodiscard]] constexpr float center_y() const noexcept { return 0.5f * (y1 + y2); }
  [[nodiscard]] constexpr float area() const noexcept {
    return std::max(0.0f, width()) * std::max(0.0f, height());
  }

  friend constexpr bool operator==(const Box&, const Box&) = default;
};

[[nodiscard]] constexpr float iou(const Box& a, const Box& b) noexcept {
  const float iw = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
  const float ih = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  const float intersection = iw * ih;
  const float union_area = a.area() + b.area() - intersection;
  return union_area > 0.0f ? intersection / union_area : 0.0f;
}

struct Detection {
  Box box;
  float score = 0.0f;
  int class_id = 0;
};

}  // namespace takt

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "takt/core/image.hpp"
#include "takt/vision/detection.hpp"

namespace takt {

// How a source image was fitted into the model input: uniform scale, then centred padding.
struct LetterboxTransform {
  float scale = 1.0f;
  float pad_x = 0.0f;
  float pad_y = 0.0f;
  int source_width = 0;
  int source_height = 0;
  int resized_width = 0;
  int resized_height = 0;

  // Maps a box from model-input pixels back to source-image pixels, clipped to the image.
  [[nodiscard]] Box to_source(const Box& model_box) const noexcept;
  // Maps a box from source-image pixels into model-input pixels.
  [[nodiscard]] Box to_model(const Box& source_box) const noexcept;
};

// Same geometry as Ultralytics' LetterBox(auto=False, scaleup=True, center=True), so boxes decoded
// here line up with the Python reference implementation.
[[nodiscard]] LetterboxTransform compute_letterbox(int source_width, int source_height,
                                                   int target_width, int target_height);

// Fused preprocessing: bilinear resize + letterbox padding + BGR->RGB + scaling to [0, 1] +
// HWC->CHW, written straight into the model's input tensor in one pass over the output.
//
// The typical Python/OpenCV path (cv2.resize, cv2.copyMakeBorder, cvtColor, transpose, astype,
// divide) makes five or six full passes over the image and allocates an intermediate for each.
// Here every output float is written exactly once, and the interpolation tables are rebuilt only
// when the source resolution changes.
class Letterboxer {
 public:
  Letterboxer(int target_width, int target_height, std::uint8_t pad_value = 114);

  LetterboxTransform run(const ImageView& source, std::span<float> chw_output);

  [[nodiscard]] int target_width() const noexcept { return target_width_; }
  [[nodiscard]] int target_height() const noexcept { return target_height_; }
  [[nodiscard]] std::size_t tensor_size() const noexcept {
    return std::size_t{3} * static_cast<std::size_t>(target_width_) *
           static_cast<std::size_t>(target_height_);
  }

 private:
  void prepare(const ImageView& source);

  int target_width_;
  int target_height_;
  float pad_value_;  // already normalised to [0, 1]

  LetterboxTransform transform_{};
  int prepared_width_ = -1;
  int prepared_height_ = -1;
  PixelFormat prepared_format_ = PixelFormat::kBgr8;

  // Per output column: byte offsets of the left/right source neighbours and the blend weight.
  std::vector<std::ptrdiff_t> x_left_;
  std::vector<std::ptrdiff_t> x_right_;
  std::vector<float> x_weight_;
  // Per output row: indices of the upper/lower source rows and the blend weight.
  std::vector<int> y_top_;
  std::vector<int> y_bottom_;
  std::vector<float> y_weight_;
};

}  // namespace takt

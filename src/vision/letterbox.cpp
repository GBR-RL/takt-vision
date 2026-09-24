#include "takt/vision/letterbox.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace takt {

Box LetterboxTransform::to_source(const Box& b) const noexcept {
  const auto w = static_cast<float>(source_width);
  const auto h = static_cast<float>(source_height);
  return {std::clamp((b.x1 - pad_x) / scale, 0.0f, w), std::clamp((b.y1 - pad_y) / scale, 0.0f, h),
          std::clamp((b.x2 - pad_x) / scale, 0.0f, w), std::clamp((b.y2 - pad_y) / scale, 0.0f, h)};
}

Box LetterboxTransform::to_model(const Box& b) const noexcept {
  return {b.x1 * scale + pad_x, b.y1 * scale + pad_y, b.x2 * scale + pad_x, b.y2 * scale + pad_y};
}

LetterboxTransform compute_letterbox(int source_width, int source_height, int target_width,
                                     int target_height) {
  if (source_width <= 0 || source_height <= 0 || target_width <= 0 || target_height <= 0) {
    throw std::invalid_argument("compute_letterbox: dimensions must be positive");
  }
  // Computed in double like the Python reference, so rounding decisions match.
  const double ratio = std::min(static_cast<double>(target_height) / source_height,
                                static_cast<double>(target_width) / source_width);

  LetterboxTransform t;
  t.scale = static_cast<float>(ratio);
  t.source_width = source_width;
  t.source_height = source_height;
  t.resized_width = static_cast<int>(std::lround(source_width * ratio));
  t.resized_height = static_cast<int>(std::lround(source_height * ratio));
  // The -0.1 reproduces Ultralytics' round(dw - 0.1): odd padding puts the extra pixel on the
  // right/bottom.
  t.pad_x = static_cast<float>(std::lround((target_width - t.resized_width) / 2.0 - 0.1));
  t.pad_y = static_cast<float>(std::lround((target_height - t.resized_height) / 2.0 - 0.1));
  return t;
}

Letterboxer::Letterboxer(int target_width, int target_height, std::uint8_t pad_value)
    : target_width_(target_width),
      target_height_(target_height),
      pad_value_(static_cast<float>(pad_value) / 255.0f) {
  if (target_width <= 0 || target_height <= 0) {
    throw std::invalid_argument("Letterboxer: target dimensions must be positive");
  }
}

void Letterboxer::prepare(const ImageView& source) {
  if (source.width == prepared_width_ && source.height == prepared_height_ &&
      source.format == prepared_format_) {
    return;
  }
  transform_ = compute_letterbox(source.width, source.height, target_width_, target_height_);
  const int channels = channel_count(source.format);

  // Half-pixel-centre bilinear mapping, identical to cv::resize(INTER_LINEAR):
  // src = (dst + 0.5) * (src_size / dst_size) - 0.5, clamped at the borders.
  const auto build_axis = [](int source_size, int resized_size, auto&& emit) {
    const double step = static_cast<double>(source_size) / resized_size;
    for (int d = 0; d < resized_size; ++d) {
      const double position = (d + 0.5) * step - 0.5;
      int lo = static_cast<int>(std::floor(position));
      auto weight = static_cast<float>(position - lo);
      if (lo < 0) {
        lo = 0;
        weight = 0.0f;
      }
      int hi = lo + 1;
      if (lo >= source_size - 1) {
        lo = source_size - 1;
        hi = lo;
        weight = 0.0f;
      }
      emit(lo, hi, weight);
    }
  };

  const auto resized_w = static_cast<std::size_t>(transform_.resized_width);
  const auto resized_h = static_cast<std::size_t>(transform_.resized_height);
  x_left_.clear();
  x_right_.clear();
  x_weight_.clear();
  x_left_.reserve(resized_w);
  x_right_.reserve(resized_w);
  x_weight_.reserve(resized_w);
  build_axis(source.width, transform_.resized_width, [&](int lo, int hi, float w) {
    x_left_.push_back(static_cast<std::ptrdiff_t>(lo) * channels);
    x_right_.push_back(static_cast<std::ptrdiff_t>(hi) * channels);
    x_weight_.push_back(w);
  });

  y_top_.clear();
  y_bottom_.clear();
  y_weight_.clear();
  y_top_.reserve(resized_h);
  y_bottom_.reserve(resized_h);
  y_weight_.reserve(resized_h);
  build_axis(source.height, transform_.resized_height, [&](int lo, int hi, float w) {
    y_top_.push_back(lo);
    y_bottom_.push_back(hi);
    y_weight_.push_back(w);
  });

  prepared_width_ = source.width;
  prepared_height_ = source.height;
  prepared_format_ = source.format;
}

LetterboxTransform Letterboxer::run(const ImageView& source, std::span<float> chw_output) {
  if (source.empty()) throw std::invalid_argument("Letterboxer::run: empty source image");
  if (chw_output.size() < tensor_size()) {
    throw std::invalid_argument("Letterboxer::run: output tensor too small");
  }
  prepare(source);

  // Output planes are R, G, B. Pick the source channel feeding each plane.
  std::array<std::ptrdiff_t, 3> channel_of_plane{};
  switch (source.format) {
    case PixelFormat::kBgr8: channel_of_plane = {2, 1, 0}; break;
    case PixelFormat::kRgb8: channel_of_plane = {0, 1, 2}; break;
    case PixelFormat::kGray8: channel_of_plane = {0, 0, 0}; break;
  }

  const auto tw = static_cast<std::size_t>(target_width_);
  const std::size_t plane_size = tw * static_cast<std::size_t>(target_height_);
  const std::array<float*, 3> planes = {chw_output.data(), chw_output.data() + plane_size,
                                        chw_output.data() + 2 * plane_size};

  const auto left = static_cast<std::size_t>(transform_.pad_x);
  const auto top = static_cast<std::size_t>(transform_.pad_y);
  const auto resized_w = static_cast<std::size_t>(transform_.resized_width);
  const auto resized_h = static_cast<std::size_t>(transform_.resized_height);
  constexpr float kInv255 = 1.0f / 255.0f;

  // Padding rows above and below the image.
  for (float* plane : planes) {
    std::fill(plane, plane + top * tw, pad_value_);
    std::fill(plane + (top + resized_h) * tw, plane + plane_size, pad_value_);
  }

  for (std::size_t dy = 0; dy < resized_h; ++dy) {
    const std::uint8_t* row0 = source.row(y_top_[dy]);
    const std::uint8_t* row1 = source.row(y_bottom_[dy]);
    const float wy = y_weight_[dy];

    std::array<float*, 3> out{};
    for (std::size_t p = 0; p < 3; ++p) {
      out[p] = planes[p] + (top + dy) * tw;
      std::fill(out[p], out[p] + left, pad_value_);
      std::fill(out[p] + left + resized_w, out[p] + tw, pad_value_);
      out[p] += left;
    }

    for (std::size_t dx = 0; dx < resized_w; ++dx) {
      const std::ptrdiff_t xl = x_left_[dx];
      const std::ptrdiff_t xr = x_right_[dx];
      const float wx = x_weight_[dx];
      for (std::size_t p = 0; p < 3; ++p) {
        const std::ptrdiff_t c = channel_of_plane[p];
        const float a = row0[xl + c];
        const float upper = a + (static_cast<float>(row0[xr + c]) - a) * wx;
        const float b = row1[xl + c];
        const float lower = b + (static_cast<float>(row1[xr + c]) - b) * wx;
        out[p][dx] = (upper + (lower - upper) * wy) * kInv255;
      }
    }
  }
  return transform_;
}

}  // namespace takt

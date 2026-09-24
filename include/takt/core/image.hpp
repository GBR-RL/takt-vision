#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace takt {

enum class PixelFormat : std::uint8_t { kBgr8, kRgb8, kGray8 };

[[nodiscard]] constexpr int channel_count(PixelFormat format) noexcept {
  return format == PixelFormat::kGray8 ? 1 : 3;
}

// Non-owning view of an interleaved 8-bit image. Rows may be padded (stride >= width * channels),
// so a view can wrap camera-driver or OpenCV buffers without copying them.
struct ImageView {
  const std::uint8_t* data = nullptr;
  int width = 0;
  int height = 0;
  std::ptrdiff_t stride = 0;  // bytes between the starts of consecutive rows
  PixelFormat format = PixelFormat::kBgr8;

  [[nodiscard]] bool empty() const noexcept { return data == nullptr || width <= 0 || height <= 0; }
  [[nodiscard]] const std::uint8_t* row(int y) const noexcept { return data + y * stride; }
};

// Owning, tightly packed image. The buffer is reused across reshapes: a pooled frame is sized to
// the camera resolution once, and from then on capturing into it never allocates.
class Image {
 public:
  Image() = default;
  Image(int width, int height, PixelFormat format) { reshape(width, height, format); }

  void reshape(int width, int height, PixelFormat format);

  [[nodiscard]] std::uint8_t* data() noexcept { return buffer_.data(); }
  [[nodiscard]] const std::uint8_t* data() const noexcept { return buffer_.data(); }
  [[nodiscard]] std::uint8_t* row(int y) noexcept { return buffer_.data() + y * stride(); }
  [[nodiscard]] const std::uint8_t* row(int y) const noexcept {
    return buffer_.data() + y * stride();
  }

  [[nodiscard]] int width() const noexcept { return width_; }
  [[nodiscard]] int height() const noexcept { return height_; }
  [[nodiscard]] PixelFormat format() const noexcept { return format_; }
  [[nodiscard]] std::ptrdiff_t stride() const noexcept {
    return static_cast<std::ptrdiff_t>(width_) * channel_count(format_);
  }
  [[nodiscard]] std::size_t size_bytes() const noexcept { return buffer_.size(); }
  [[nodiscard]] bool empty() const noexcept { return buffer_.empty(); }
  [[nodiscard]] ImageView view() const noexcept {
    return {data(), width_, height_, stride(), format_};
  }

 private:
  std::vector<std::uint8_t> buffer_;
  int width_ = 0;
  int height_ = 0;
  PixelFormat format_ = PixelFormat::kBgr8;
};

}  // namespace takt

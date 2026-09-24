#include "takt/core/image.hpp"

#include <stdexcept>

namespace takt {

void Image::reshape(int width, int height, PixelFormat format) {
  if (width < 0 || height < 0) throw std::invalid_argument("Image::reshape: negative dimensions");
  width_ = width;
  height_ = height;
  format_ = format;
  // std::vector::resize keeps its capacity when shrinking, so a frame that is reshaped to the same
  // (or a smaller) resolution every time never touches the allocator.
  buffer_.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                 static_cast<std::size_t>(channel_count(format)));
}

}  // namespace takt

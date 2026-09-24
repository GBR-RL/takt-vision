#include "takt/io/synthetic_source.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <numbers>
#include <stdexcept>
#include <thread>

namespace takt {
namespace {

constexpr std::array<std::array<std::uint8_t, 3>, 6> kPalette = {{
    {60, 180, 255},
    {80, 220, 120},
    {230, 120, 60},
    {200, 90, 220},
    {70, 200, 230},
    {240, 200, 80},
}};

void fill_rect(Image& image, int x0, int y0, int x1, int y1,
               const std::array<std::uint8_t, 3>& bgr) {
  x0 = std::clamp(x0, 0, image.width());
  x1 = std::clamp(x1, 0, image.width());
  y0 = std::clamp(y0, 0, image.height());
  y1 = std::clamp(y1, 0, image.height());
  for (int y = y0; y < y1; ++y) {
    std::uint8_t* px = image.row(y) + static_cast<std::ptrdiff_t>(x0) * 3;
    for (int x = x0; x < x1; ++x, px += 3) std::memcpy(px, bgr.data(), 3);
  }
}

}  // namespace

SyntheticSource::SyntheticSource(SyntheticSourceOptions options) : options_(options) {
  if (options.width <= 0 || options.height <= 0 || options.fps < 0.0) {
    throw std::invalid_argument("SyntheticSource: invalid options");
  }
}

bool SyntheticSource::read(Image& out) {
  if (options_.frame_count != 0 && produced_ >= options_.frame_count) return false;

  if (options_.fps > 0.0) {
    // Free-running camera: frames are due on a fixed grid. If the consumer fell behind, deliver
    // immediately but re-anchor the grid, as a real sensor would have overwritten the missed
    // exposures rather than queueing them.
    const auto period =
        std::chrono::duration_cast<Nanos>(std::chrono::duration<double>(1.0 / options_.fps));
    const TimePoint t = now();
    if (produced_ == 0 || t > next_deadline_ + period) next_deadline_ = t;
    std::this_thread::sleep_until(next_deadline_);
    next_deadline_ += period;
  }

  out.reshape(options_.width, options_.height, PixelFormat::kBgr8);
  std::memset(out.data(), 28, out.size_bytes());

  const double t = static_cast<double>(produced_) / 60.0;
  const double w = options_.width;
  const double h = options_.height;
  for (int k = 0; k < options_.num_objects; ++k) {
    const double phase = t + k * 2.0 * std::numbers::pi / std::max(1, options_.num_objects);
    const double cx = w * (0.5 + 0.35 * std::cos(phase));
    const double cy = h * (0.5 + 0.30 * std::sin(1.3 * phase));
    const double half_w = w * (0.04 + 0.01 * k);
    const double half_h = h * 0.07;
    fill_rect(out, static_cast<int>(cx - half_w), static_cast<int>(cy - half_h),
              static_cast<int>(cx + half_w), static_cast<int>(cy + half_h),
              kPalette[static_cast<std::size_t>(k) % kPalette.size()]);
  }
  ++produced_;
  return true;
}

std::string SyntheticSource::describe() const {
  return std::format("synthetic {}x{}@{:g}fps", options_.width, options_.height, options_.fps);
}

}  // namespace takt

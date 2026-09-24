#pragma once

#include <chrono>

namespace takt {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using Nanos = std::chrono::nanoseconds;

[[nodiscard]] inline TimePoint now() noexcept {
  return Clock::now();
}

[[nodiscard]] constexpr double to_ms(Nanos duration) noexcept {
  return std::chrono::duration<double, std::milli>(duration).count();
}

}  // namespace takt

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

#include "takt/core/clock.hpp"

namespace takt {

struct LatencySummary {
  std::uint64_t count = 0;
  double mean_ms = 0.0;
  double min_ms = 0.0;
  double p50_ms = 0.0;
  double p90_ms = 0.0;
  double p95_ms = 0.0;
  double p99_ms = 0.0;
  double p999_ms = 0.0;
  double max_ms = 0.0;
};

// Log-linear latency histogram in the style of HdrHistogram, with microsecond resolution.
//
// Values below 128 us get one bucket each; above that every power-of-two range is split into 64
// sub-buckets, bounding the relative error of any reported percentile to 1/64 (~1.6%). 2240
// buckets cover 1 us to ~12.7 days in ~18 KB.
//
// record() is wait-free (relaxed fetch_add), so all pipeline threads can record into shared
// histograms concurrently and the reporting thread can read percentiles at any time. Tail
// latency, not the mean, is what decides whether an inspection station meets its takt time.
class LatencyHistogram {
 public:
  static constexpr int kSubBucketBits = 7;
  static constexpr std::uint64_t kSubBucketCount = std::uint64_t{1} << kSubBucketBits;  // 128
  static constexpr std::uint64_t kHalfSubBucketCount = kSubBucketCount / 2;             // 64
  static constexpr std::uint64_t kMaxValueUs = (std::uint64_t{1} << 40) - 1;
  static constexpr std::size_t kBucketCount = 2240;

  LatencyHistogram() = default;
  LatencyHistogram(const LatencyHistogram&) = delete;
  LatencyHistogram& operator=(const LatencyHistogram&) = delete;

  void record(Nanos latency) noexcept;
  void record_us(std::uint64_t microseconds) noexcept;
  void reset() noexcept;

  [[nodiscard]] std::uint64_t count() const noexcept {
    return count_.load(std::memory_order_relaxed);
  }
  // Returns the latency (ms) below which fraction `q` of samples fall. q in [0, 1].
  [[nodiscard]] double percentile_ms(double q) const noexcept;
  [[nodiscard]] LatencySummary summarize() const noexcept;

  [[nodiscard]] static std::size_t bucket_index(std::uint64_t value_us) noexcept;
  // Largest value that maps to bucket `index` (the value reported for that bucket).
  [[nodiscard]] static std::uint64_t bucket_upper_bound(std::size_t index) noexcept;

 private:
  std::array<std::atomic<std::uint64_t>, kBucketCount> buckets_{};
  std::atomic<std::uint64_t> count_{0};
  std::atomic<std::uint64_t> sum_us_{0};
  std::atomic<std::uint64_t> min_us_{UINT64_MAX};
  std::atomic<std::uint64_t> max_us_{0};
};

}  // namespace takt

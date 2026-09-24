#include "takt/core/latency_histogram.hpp"

#include <algorithm>
#include <bit>
#include <cmath>

namespace takt {
namespace {

constexpr double kUsPerMs = 1000.0;

// C++20 has no atomic fetch_min/fetch_max (they arrive in C++26), hence the CAS loops.
void update_min(std::atomic<std::uint64_t>& target, std::uint64_t value) noexcept {
  std::uint64_t current = target.load(std::memory_order_relaxed);
  while (value < current &&
         !target.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
  }
}

void update_max(std::atomic<std::uint64_t>& target, std::uint64_t value) noexcept {
  std::uint64_t current = target.load(std::memory_order_relaxed);
  while (value > current &&
         !target.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
  }
}

}  // namespace

std::size_t LatencyHistogram::bucket_index(std::uint64_t value_us) noexcept {
  value_us = std::min(value_us, kMaxValueUs);
  if (value_us < kSubBucketCount) return static_cast<std::size_t>(value_us);
  // msb >= kSubBucketBits here. Shift so the value keeps kSubBucketBits significant bits; the top
  // bit is always set, which leaves 64 distinct sub-buckets per power of two.
  const auto msb = static_cast<std::uint64_t>(std::bit_width(value_us)) - 1;
  const std::uint64_t shift = msb - (kSubBucketBits - 1);
  const std::uint64_t top = value_us >> shift;  // in [64, 128)
  return static_cast<std::size_t>(kSubBucketCount + (shift - 1) * kHalfSubBucketCount +
                                  (top - kHalfSubBucketCount));
}

std::uint64_t LatencyHistogram::bucket_upper_bound(std::size_t index) noexcept {
  if (index < kSubBucketCount) return index;
  const std::uint64_t offset = index - kSubBucketCount;
  const std::uint64_t shift = offset / kHalfSubBucketCount + 1;
  const std::uint64_t top = offset % kHalfSubBucketCount + kHalfSubBucketCount;
  return ((top + 1) << shift) - 1;
}

void LatencyHistogram::record(Nanos latency) noexcept {
  const auto us = std::chrono::duration_cast<std::chrono::microseconds>(latency).count();
  record_us(us < 0 ? 0 : static_cast<std::uint64_t>(us));
}

void LatencyHistogram::record_us(std::uint64_t microseconds) noexcept {
  buckets_[bucket_index(microseconds)].fetch_add(1, std::memory_order_relaxed);
  count_.fetch_add(1, std::memory_order_relaxed);
  sum_us_.fetch_add(microseconds, std::memory_order_relaxed);
  update_min(min_us_, microseconds);
  update_max(max_us_, microseconds);
}

void LatencyHistogram::reset() noexcept {
  for (auto& bucket : buckets_) bucket.store(0, std::memory_order_relaxed);
  count_.store(0, std::memory_order_relaxed);
  sum_us_.store(0, std::memory_order_relaxed);
  min_us_.store(UINT64_MAX, std::memory_order_relaxed);
  max_us_.store(0, std::memory_order_relaxed);
}

double LatencyHistogram::percentile_ms(double q) const noexcept {
  // Buckets are read without a global snapshot, so under concurrent recording the answer is
  // approximate by at most the handful of samples recorded during the scan.
  const std::uint64_t total = count_.load(std::memory_order_relaxed);
  if (total == 0) return 0.0;
  q = std::clamp(q, 0.0, 1.0);
  const auto rank = std::max<std::uint64_t>(
      1, static_cast<std::uint64_t>(std::ceil(q * static_cast<double>(total))));

  const std::uint64_t max_seen = max_us_.load(std::memory_order_relaxed);
  std::uint64_t cumulative = 0;
  for (std::size_t i = 0; i < kBucketCount; ++i) {
    cumulative += buckets_[i].load(std::memory_order_relaxed);
    if (cumulative >= rank) {
      // Report the bucket's upper bound (conservative for tail latency), but never more than
      // the largest value actually observed.
      return static_cast<double>(std::min(bucket_upper_bound(i), max_seen)) / kUsPerMs;
    }
  }
  return static_cast<double>(max_seen) / kUsPerMs;
}

LatencySummary LatencyHistogram::summarize() const noexcept {
  LatencySummary s;
  s.count = count_.load(std::memory_order_relaxed);
  if (s.count == 0) return s;
  s.mean_ms = static_cast<double>(sum_us_.load(std::memory_order_relaxed)) /
              static_cast<double>(s.count) / kUsPerMs;
  s.min_ms = static_cast<double>(min_us_.load(std::memory_order_relaxed)) / kUsPerMs;
  s.max_ms = static_cast<double>(max_us_.load(std::memory_order_relaxed)) / kUsPerMs;
  s.p50_ms = percentile_ms(0.50);
  s.p90_ms = percentile_ms(0.90);
  s.p95_ms = percentile_ms(0.95);
  s.p99_ms = percentile_ms(0.99);
  s.p999_ms = percentile_ms(0.999);
  return s;
}

}  // namespace takt

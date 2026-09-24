#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <thread>
#include <vector>

#include "takt/core/latency_histogram.hpp"

namespace takt {
namespace {

TEST(LatencyHistogram, BucketIndexIsMonotonicAndContiguous) {
  std::size_t previous = 0;
  for (std::uint64_t v = 0; v < 1'000'000; ++v) {
    const std::size_t index = LatencyHistogram::bucket_index(v);
    ASSERT_TRUE(index == previous || index == previous + 1) << "value " << v;
    ASSERT_LE(v, LatencyHistogram::bucket_upper_bound(index));
    previous = index;
  }
  EXPECT_LT(LatencyHistogram::bucket_index(LatencyHistogram::kMaxValueUs),
            LatencyHistogram::kBucketCount);
  EXPECT_EQ(LatencyHistogram::bucket_index(UINT64_MAX),
            LatencyHistogram::bucket_index(LatencyHistogram::kMaxValueUs));
}

TEST(LatencyHistogram, SmallValuesAreExact) {
  LatencyHistogram h;
  for (std::uint64_t v = 1; v <= 100; ++v) h.record_us(v);
  EXPECT_DOUBLE_EQ(h.percentile_ms(0.5), 0.050);
  EXPECT_DOUBLE_EQ(h.percentile_ms(0.99), 0.099);
  EXPECT_DOUBLE_EQ(h.percentile_ms(1.0), 0.100);
  const LatencySummary s = h.summarize();
  EXPECT_EQ(s.count, 100u);
  EXPECT_DOUBLE_EQ(s.min_ms, 0.001);
  EXPECT_DOUBLE_EQ(s.max_ms, 0.100);
  EXPECT_NEAR(s.mean_ms, 0.0505, 1e-9);
}

TEST(LatencyHistogram, PercentilesWithinRelativeErrorBound) {
  std::mt19937_64 rng(1);
  std::lognormal_distribution<double> dist(9.0, 0.6);  // ~8 ms median, long tail
  std::vector<std::uint64_t> values(50'000);
  LatencyHistogram h;
  for (auto& v : values) {
    v = static_cast<std::uint64_t>(dist(rng));
    h.record_us(v);
  }
  std::sort(values.begin(), values.end());
  for (const double q : {0.5, 0.9, 0.95, 0.99, 0.999}) {
    const auto rank =
        static_cast<std::size_t>(std::ceil(q * static_cast<double>(values.size()))) - 1;
    const double exact_ms = static_cast<double>(values[rank]) / 1000.0;
    EXPECT_NEAR(h.percentile_ms(q), exact_ms, exact_ms / 64.0 + 1e-3) << "q=" << q;
    EXPECT_GE(h.percentile_ms(q), exact_ms) << "reported percentiles are conservative";
  }
}

TEST(LatencyHistogram, ConcurrentRecordingCountsEverySample) {
  LatencyHistogram h;
  constexpr int kThreads = 4;
  constexpr int kPerThread = 50'000;
  {
    std::vector<std::jthread> threads;
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&h, t] {
        for (int i = 0; i < kPerThread; ++i)
          h.record_us(static_cast<std::uint64_t>(t * 1000 + i % 1000));
      });
    }
  }
  EXPECT_EQ(h.count(), static_cast<std::uint64_t>(kThreads * kPerThread));
  EXPECT_DOUBLE_EQ(h.summarize().max_ms, 3.999);
}

TEST(LatencyHistogram, ResetClearsEverything) {
  LatencyHistogram h;
  h.record(std::chrono::milliseconds(5));
  h.reset();
  EXPECT_EQ(h.count(), 0u);
  EXPECT_DOUBLE_EQ(h.percentile_ms(0.99), 0.0);
  EXPECT_EQ(h.summarize().count, 0u);
}

}  // namespace
}  // namespace takt

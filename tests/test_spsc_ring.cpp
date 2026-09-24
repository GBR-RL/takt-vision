#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>

#include "takt/core/spsc_ring.hpp"

namespace takt {
namespace {

TEST(SpscRing, CapacityRoundsUpToPowerOfTwo) {
  EXPECT_EQ(SpscRing<int>(0).capacity(), 1u);
  EXPECT_EQ(SpscRing<int>(3).capacity(), 4u);
  EXPECT_EQ(SpscRing<int>(8).capacity(), 8u);
}

TEST(SpscRing, FifoOrderAndFullEmptyBehaviour) {
  SpscRing<int> ring(4);
  EXPECT_FALSE(ring.try_pop().has_value());
  for (int i = 0; i < 4; ++i) {
    int v = i;
    ASSERT_TRUE(ring.try_push(v));
  }
  int overflow = 99;
  EXPECT_FALSE(ring.try_push(overflow));
  EXPECT_EQ(overflow, 99) << "a failed push must not consume the value";
  EXPECT_EQ(ring.size_approx(), 4u);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(ring.try_pop().value(), i);
  EXPECT_FALSE(ring.try_pop().has_value());
}

TEST(SpscRing, WrapsAroundManyTimes) {
  SpscRing<int> ring(2);
  for (int i = 0; i < 1000; ++i) {
    int v = i;
    ASSERT_TRUE(ring.try_push(v));
    ASSERT_EQ(ring.try_pop().value(), i);
  }
}

TEST(SpscRing, HoldsMoveOnlyTypes) {
  SpscRing<std::unique_ptr<int>> ring(2);
  auto p = std::make_unique<int>(7);
  ASSERT_TRUE(ring.try_push(p));
  EXPECT_EQ(p, nullptr);
  auto out = ring.try_pop();
  ASSERT_TRUE(out.has_value());
  EXPECT_EQ(**out, 7);
}

TEST(SpscRing, CloseDrainsRemainingElementsThenReportsEnd) {
  SpscRing<int> ring(4);
  int a = 1;
  int b = 2;
  ASSERT_TRUE(ring.try_push(a));
  ASSERT_TRUE(ring.try_push(b));
  ring.close();
  int c = 3;
  EXPECT_FALSE(ring.push(c)) << "push after close must fail";
  EXPECT_EQ(ring.pop().value(), 1);
  EXPECT_EQ(ring.pop().value(), 2);
  EXPECT_FALSE(ring.pop().has_value());
}

TEST(SpscRing, CloseWakesBlockedConsumer) {
  SpscRing<int> ring(2);
  std::atomic<bool> returned{false};
  std::jthread consumer([&] {
    EXPECT_FALSE(ring.pop().has_value());
    returned = true;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_FALSE(returned.load());
  ring.close();
  consumer.join();
  EXPECT_TRUE(returned.load());
}

TEST(SpscRing, CloseWakesBlockedProducer) {
  SpscRing<int> ring(1);
  int first = 1;
  ASSERT_TRUE(ring.try_push(first));
  std::jthread producer([&] {
    int second = 2;
    EXPECT_FALSE(ring.push(second));
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  ring.close();
}

// Two threads hammer a tiny ring through its blocking API; every value must arrive exactly once
// and in order. Run under ThreadSanitizer in CI to check the memory ordering.
TEST(SpscRing, ConcurrentTransferPreservesOrder) {
  constexpr std::uint64_t kCount = 200'000;
  SpscRing<std::uint64_t> ring(8);
  std::jthread producer([&] {
    for (std::uint64_t i = 0; i < kCount; ++i) {
      std::uint64_t v = i;
      ASSERT_TRUE(ring.push(v));
    }
    ring.close();
  });
  std::uint64_t expected = 0;
  while (auto v = ring.pop()) {
    ASSERT_EQ(*v, expected);
    ++expected;
  }
  EXPECT_EQ(expected, kCount);
}

}  // namespace
}  // namespace takt

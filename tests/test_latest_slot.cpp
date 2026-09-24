#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>

#include "takt/core/latest_slot.hpp"

namespace takt {
namespace {

TEST(LatestSlot, EmptyUntilPublished) {
  LatestSlot<int> slot;
  EXPECT_FALSE(slot.try_take().has_value());
  EXPECT_FALSE(slot.publish(1).has_value());
  EXPECT_EQ(slot.try_take().value(), 1);
  EXPECT_FALSE(slot.try_take().has_value()) << "a value is delivered at most once";
}

TEST(LatestSlot, NewerValueDisplacesUntakenOne) {
  LatestSlot<int> slot;
  EXPECT_FALSE(slot.publish(1).has_value());
  const auto displaced = slot.publish(2);
  ASSERT_TRUE(displaced.has_value());
  EXPECT_EQ(*displaced, 1);
  EXPECT_EQ(slot.try_take().value(), 2);
}

TEST(LatestSlot, TakenValuesAreNotReportedAsDisplaced) {
  LatestSlot<int> slot;
  for (int i = 0; i < 10; ++i) {
    EXPECT_FALSE(slot.publish(i).has_value());
    EXPECT_EQ(slot.try_take().value(), i);
  }
}

TEST(LatestSlot, MoveOnlyValuesAreHandedBack) {
  LatestSlot<std::unique_ptr<int>> slot;
  EXPECT_FALSE(slot.publish(std::make_unique<int>(1)).has_value());
  auto displaced = slot.publish(std::make_unique<int>(2));
  ASSERT_TRUE(displaced.has_value());
  EXPECT_EQ(**displaced, 1);
  EXPECT_EQ(**slot.try_take(), 2);
}

TEST(LatestSlot, CloseReturnsPublishedValueAndWakesConsumer) {
  LatestSlot<int> slot;
  std::jthread consumer([&] { EXPECT_FALSE(slot.take().has_value()); });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  slot.close();
  consumer.join();
  const auto rejected = slot.publish(5);
  ASSERT_TRUE(rejected.has_value());
  EXPECT_EQ(*rejected, 5);
}

// A fast producer and a slow consumer: the consumer must see strictly increasing values (never an
// old one after a newer one), and every published value is either taken or handed back.
TEST(LatestSlot, ConcurrentPublishIsLossFreeAccountingAndMonotonic) {
  constexpr std::uint64_t kCount = 100'000;
  LatestSlot<std::uint64_t> slot;
  std::uint64_t displaced = 0;
  std::jthread producer([&] {
    for (std::uint64_t i = 1; i <= kCount; ++i) {
      if (slot.publish(i).has_value()) ++displaced;
    }
    slot.close();
  });

  std::uint64_t taken = 0;
  std::uint64_t last = 0;
  while (auto v = slot.take()) {
    ASSERT_GT(*v, last);
    last = *v;
    ++taken;
  }
  producer.join();
  EXPECT_EQ(last, kCount) << "the final value must always be delivered";
  EXPECT_EQ(taken + displaced, kCount);
}

}  // namespace
}  // namespace takt

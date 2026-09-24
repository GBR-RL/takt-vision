#include <gtest/gtest.h>

#include <chrono>
#include <stop_token>
#include <thread>
#include <vector>

#include "takt/pipeline/frame_pool.hpp"

namespace takt {
namespace {

TEST(FramePool, PreallocatesBuffers) {
  FramePool pool(3, {12, 34, 5});
  EXPECT_EQ(pool.capacity(), 3u);
  auto frame = pool.try_acquire();
  ASSERT_TRUE(frame);
  EXPECT_EQ(frame->input.size(), 12u);
  EXPECT_EQ(frame->output.size(), 34u);
  EXPECT_GE(frame->detections.capacity(), 5u);
}

TEST(FramePool, DroppingAFramePtrReturnsItToThePool) {
  FramePool pool(2, {1, 1, 1});
  {
    auto a = pool.try_acquire();
    auto b = pool.try_acquire();
    EXPECT_FALSE(pool.try_acquire()) << "exhausted";
    EXPECT_EQ(pool.available(), 0u);
  }
  EXPECT_EQ(pool.available(), 2u);
}

TEST(FramePool, RecycledFramesAreResetButKeepCapacity) {
  FramePool pool(1, {4, 4, 16});
  Frame* raw = nullptr;
  {
    auto frame = pool.try_acquire();
    raw = frame.get();
    frame->sequence = 42;
    frame->detections.push_back({});
    frame->image.reshape(64, 48, PixelFormat::kBgr8);
  }
  auto again = pool.try_acquire();
  EXPECT_EQ(again.get(), raw);
  EXPECT_EQ(again->sequence, 0u);
  EXPECT_TRUE(again->detections.empty());
  EXPECT_GE(again->detections.capacity(), 16u);
  EXPECT_EQ(again->image.width(), 64) << "image buffer is kept for reuse";
}

TEST(FramePool, BlockingAcquireWakesWhenAFrameIsReleased) {
  FramePool pool(1, {1, 1, 1});
  auto held = pool.try_acquire();
  std::stop_source stop;
  std::jthread waiter([&] {
    auto frame = pool.acquire(stop.get_token());
    EXPECT_TRUE(frame);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  held.reset();
}

TEST(FramePool, BlockingAcquireGivesUpOnStop) {
  FramePool pool(1, {1, 1, 1});
  auto held = pool.try_acquire();
  std::stop_source stop;
  std::jthread waiter([&] { EXPECT_FALSE(pool.acquire(stop.get_token())); });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  stop.request_stop();
}

}  // namespace
}  // namespace takt

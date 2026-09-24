#include <gtest/gtest.h>

#include <vector>

#include "takt/track/byte_tracker.hpp"

namespace takt {
namespace {

Detection det(float x, float y, float score, int cls = 0) {
  return {{x, y, x + 40.0f, y + 80.0f}, score, cls};
}

std::vector<TrackedObject> step(ByteTracker& tracker, std::vector<Detection> detections) {
  std::vector<TrackedObject> out;
  tracker.update(detections, out);
  return out;
}

TEST(ByteTracker, KeepsIdOfSteadilyMovingObject) {
  ByteTracker tracker;
  const auto first = step(tracker, {det(0, 100, 0.9f)});
  ASSERT_EQ(first.size(), 1u) << "tracks on the very first frame are confirmed immediately";
  const int id = first[0].track_id;
  for (int t = 1; t < 50; ++t) {
    const auto out = step(tracker, {det(5.0f * static_cast<float>(t), 100, 0.9f)});
    ASSERT_EQ(out.size(), 1u) << "frame " << t;
    EXPECT_EQ(out[0].track_id, id);
    EXPECT_EQ(out[0].age_frames, t + 1);
  }
}

TEST(ByteTracker, NewTrackNeedsASecondSightingAfterFirstFrame) {
  ByteTracker tracker;
  EXPECT_TRUE(step(tracker, {}).empty());
  EXPECT_TRUE(step(tracker, {det(10, 10, 0.9f)}).empty()) << "tentative after one sighting";
  const auto out = step(tracker, {det(12, 10, 0.9f)});
  ASSERT_EQ(out.size(), 1u);
}

TEST(ByteTracker, SingleFrameFalsePositiveNeverBecomesATrack) {
  ByteTracker tracker;
  static_cast<void>(step(tracker, {}));
  static_cast<void>(step(tracker, {det(300, 300, 0.9f)}));
  for (int t = 0; t < 5; ++t) EXPECT_TRUE(step(tracker, {}).empty());
}

TEST(ByteTracker, LowScoreDetectionsKeepTrackAlive) {
  // The second association stage: a detection whose confidence dips (occlusion, blur) below the
  // high threshold still continues the existing track instead of dropping it.
  ByteTracker tracker;
  int id = -1;
  for (int t = 0; t < 10; ++t) {
    const auto out = step(tracker, {det(5.0f * static_cast<float>(t), 50, 0.9f)});
    ASSERT_EQ(out.size(), 1u);
    id = out[0].track_id;
  }
  for (int t = 10; t < 16; ++t) {
    const auto out = step(tracker, {det(5.0f * static_cast<float>(t), 50, 0.15f)});
    ASSERT_EQ(out.size(), 1u) << "frame " << t;
    EXPECT_EQ(out[0].track_id, id);
  }
}

TEST(ByteTracker, ReidentifiesAfterShortOcclusion) {
  ByteTracker tracker;
  int id = -1;
  for (int t = 0; t < 20; ++t) {
    const auto out = step(tracker, {det(4.0f * static_cast<float>(t), 200, 0.9f)});
    id = out.at(0).track_id;
  }
  for (int t = 20; t < 30; ++t)
    EXPECT_TRUE(step(tracker, {}).empty()) << "lost tracks are not reported";
  // Reappears where constant-velocity motion predicts it.
  const auto out = step(tracker, {det(4.0f * 30, 200, 0.9f)});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].track_id, id);
}

TEST(ByteTracker, ForgetsTracksLostLongerThanBuffer) {
  ByteTrackerOptions options;
  options.track_buffer = 10;
  ByteTracker tracker(options);
  int id = -1;
  for (int t = 0; t < 5; ++t) id = step(tracker, {det(100, 100, 0.9f)}).at(0).track_id;
  for (int t = 0; t < 15; ++t) static_cast<void>(step(tracker, {}));
  static_cast<void>(step(tracker, {det(100, 100, 0.9f)}));
  const auto out = step(tracker, {det(100, 100, 0.9f)});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_NE(out[0].track_id, id);
}

TEST(ByteTracker, CrossingObjectsKeepTheirIds) {
  ByteTracker tracker;
  int id_a = -1;
  int id_b = -1;
  for (int t = 0; t < 60; ++t) {
    const float ft = static_cast<float>(t);
    // A moves right, B moves left; their boxes overlap while they pass each other.
    const auto out =
        step(tracker, {det(6.0f * ft, 100, 0.9f, 0), det(360.0f - 6.0f * ft, 140, 0.85f, 1)});
    ASSERT_EQ(out.size(), 2u) << "frame " << t;
    for (const auto& obj : out) {
      if (t == 0) {
        (obj.class_id == 0 ? id_a : id_b) = obj.track_id;
      } else {
        EXPECT_EQ(obj.track_id, obj.class_id == 0 ? id_a : id_b)
            << "identity switch at frame " << t;
      }
    }
  }
}

TEST(ByteTracker, ResetRestartsIdsAndFrames) {
  ByteTracker tracker;
  static_cast<void>(step(tracker, {det(0, 0, 0.9f)}));
  tracker.reset();
  EXPECT_EQ(tracker.frame_index(), 0);
  const auto out = step(tracker, {det(0, 0, 0.9f)});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].track_id, 1);
}

}  // namespace
}  // namespace takt

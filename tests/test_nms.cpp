#include <gtest/gtest.h>

#include <vector>

#include "takt/vision/nms.hpp"

namespace takt {
namespace {

TEST(Iou, BasicCases) {
  const Box a{0, 0, 10, 10};
  EXPECT_FLOAT_EQ(iou(a, a), 1.0f);
  EXPECT_FLOAT_EQ(iou(a, {20, 20, 30, 30}), 0.0f);
  EXPECT_FLOAT_EQ(iou(a, {5, 0, 15, 10}), 50.0f / 150.0f);
  EXPECT_FLOAT_EQ(iou(a, {10, 0, 20, 10}), 0.0f) << "touching edges do not overlap";
  EXPECT_FLOAT_EQ(iou({0, 0, 0, 0}, {0, 0, 0, 0}), 0.0f)
      << "degenerate boxes do not divide by zero";
}

TEST(Nms, SuppressesOverlapsOfSameClassOnly) {
  std::vector<Detection> dets = {
      {{0, 0, 10, 10}, 0.8f, 0},
      {{1, 0, 11, 10}, 0.9f, 0},    // overlaps the first strongly, higher score
      {{1, 0, 11, 10}, 0.7f, 1},    // same place, other class: kept
      {{50, 50, 60, 60}, 0.6f, 0},  // far away: kept
  };
  Nms nms(NmsOptions{0.5f, 300, 30000, false});
  nms.apply(dets);
  ASSERT_EQ(dets.size(), 3u);
  EXPECT_FLOAT_EQ(dets[0].score, 0.9f);
  EXPECT_FLOAT_EQ(dets[1].score, 0.7f);
  EXPECT_FLOAT_EQ(dets[2].score, 0.6f);
}

TEST(Nms, ClassAgnosticSuppressesAcrossClasses) {
  std::vector<Detection> dets = {{{0, 0, 10, 10}, 0.9f, 0}, {{0, 0, 10, 10}, 0.8f, 1}};
  Nms nms(NmsOptions{0.5f, 300, 30000, true});
  nms.apply(dets);
  ASSERT_EQ(dets.size(), 1u);
  EXPECT_EQ(dets[0].class_id, 0);
}

TEST(Nms, ThresholdIsStrictlyGreater) {
  // IoU exactly 0.5: torchvision keeps both.
  std::vector<Detection> dets = {{{0, 0, 30, 10}, 0.9f, 0}, {{10, 0, 40, 10}, 0.8f, 0}};
  ASSERT_FLOAT_EQ(iou(dets[0].box, dets[1].box), 0.5f);
  Nms nms(NmsOptions{0.5f, 300, 30000, false});
  nms.apply(dets);
  EXPECT_EQ(dets.size(), 2u);
}

TEST(Nms, SuppressedBoxesDoNotSuppressOthers) {
  // B overlaps A (suppressed by A); C overlaps only B, so C must survive.
  std::vector<Detection> dets = {
      {{0, 0, 10, 10}, 0.9f, 0},  // A
      {{4, 0, 14, 10}, 0.8f, 0},  // B: IoU(A,B) = 6/14 > 0.4
      {{8, 0, 18, 10}, 0.7f, 0},  // C: IoU(A,C) = 2/18, IoU(B,C) = 6/14
  };
  Nms nms(NmsOptions{0.4f, 300, 30000, false});
  nms.apply(dets);
  ASSERT_EQ(dets.size(), 2u);
  EXPECT_FLOAT_EQ(dets[0].score, 0.9f);
  EXPECT_FLOAT_EQ(dets[1].score, 0.7f);
}

TEST(Nms, RespectsMaxDetectionsAndCandidates) {
  std::vector<Detection> dets;
  for (int i = 0; i < 10; ++i) {
    const auto x = static_cast<float>(i * 100);
    dets.push_back({{x, 0, x + 10, 10}, 0.1f * static_cast<float>(i), 0});
  }
  auto limited = dets;
  Nms(NmsOptions{0.5f, 3, 30000, false}).apply(limited);
  ASSERT_EQ(limited.size(), 3u);
  EXPECT_FLOAT_EQ(limited[0].score, 0.9f);

  auto candidates = dets;
  Nms(NmsOptions{0.5f, 300, 4, false}).apply(candidates);
  EXPECT_EQ(candidates.size(), 4u);
}

TEST(Nms, ScaleToSourceUndoesLetterbox) {
  const auto t = compute_letterbox(1280, 720, 640, 640);
  std::vector<Detection> dets = {{{0, 140, 640, 500}, 0.9f, 0}};
  scale_to_source(dets, t);
  EXPECT_EQ(dets[0].box, (Box{0, 0, 1280, 720}));
}

}  // namespace
}  // namespace takt

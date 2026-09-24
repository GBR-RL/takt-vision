#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "takt/vision/yolo_decoder.hpp"

namespace takt {
namespace {

TEST(YoloLayout, InfersBothOrientations) {
  const std::vector<std::int64_t> ultralytics{1, 84, 8400};
  const auto a = yolo_layout_from_shape(ultralytics);
  EXPECT_EQ(a.num_anchors, 8400u);
  EXPECT_EQ(a.num_classes, 80u);
  EXPECT_TRUE(a.channels_first);

  const std::vector<std::int64_t> transposed{1, 8400, 6};
  const auto b = yolo_layout_from_shape(transposed);
  EXPECT_EQ(b.num_anchors, 8400u);
  EXPECT_EQ(b.num_classes, 2u);
  EXPECT_FALSE(b.channels_first);
}

TEST(YoloLayout, RejectsPostNmsExports) {
  const std::vector<std::int64_t> end_to_end{1, 300, 4};
  EXPECT_THROW(static_cast<void>(yolo_layout_from_shape(end_to_end)), std::invalid_argument);
  const std::vector<std::int64_t> three_d{2, 84, 8400};
  EXPECT_THROW(static_cast<void>(yolo_layout_from_shape(three_d)), std::invalid_argument);
}

// Builds a channels-first head with 3 classes and 4 anchors.
std::vector<float> make_head(std::size_t anchors, std::size_t classes) {
  return std::vector<float>((4 + classes) * anchors, 0.0f);
}

TEST(YoloDecoder, PicksBestClassAndConvertsBoxes) {
  constexpr std::size_t n = 4;
  constexpr std::size_t nc = 3;
  auto raw = make_head(n, nc);
  const auto set = [&](std::size_t attribute, std::size_t anchor, float v) {
    raw[attribute * n + anchor] = v;
  };
  // Anchor 1: box centred (100, 50), 20x10; class scores 0.2, 0.9, 0.4 -> class 1.
  set(0, 1, 100.0f);
  set(1, 1, 50.0f);
  set(2, 1, 20.0f);
  set(3, 1, 10.0f);
  set(4, 1, 0.2f);
  set(5, 1, 0.9f);
  set(6, 1, 0.4f);
  // Anchor 3: below threshold.
  set(4, 3, 0.1f);

  YoloDecoder decoder({n, nc, true});
  std::vector<Detection> out{{}, {}};  // decode must replace, not append
  decoder.decode(raw, 0.25f, out);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].class_id, 1);
  EXPECT_FLOAT_EQ(out[0].score, 0.9f);
  EXPECT_EQ(out[0].box, (Box{90.0f, 45.0f, 110.0f, 55.0f}));
}

TEST(YoloDecoder, ThresholdIsStrict) {
  auto raw = make_head(2, 1);
  raw[4 * 2 + 0] = 0.25f;
  raw[4 * 2 + 1] = 0.2501f;
  YoloDecoder decoder({2, 1, true});
  std::vector<Detection> out;
  decoder.decode(raw, 0.25f, out);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_FLOAT_EQ(out[0].score, 0.2501f);
}

TEST(YoloDecoder, ChannelsLastGivesSameResult) {
  constexpr std::size_t n = 3;
  constexpr std::size_t nc = 2;
  std::vector<float> first = make_head(n, nc);
  std::vector<float> last(first.size());
  float v = 1.0f;
  for (std::size_t a = 0; a < n; ++a) {
    for (std::size_t attr = 0; attr < 4 + nc; ++attr) {
      const float value = attr < 4 ? 10.0f * v : 0.1f * static_cast<float>(attr + a);
      first[attr * n + a] = value;
      last[a * (4 + nc) + attr] = value;
      v += 1.0f;
    }
  }
  YoloDecoder cf({n, nc, true});
  YoloDecoder cl({n, nc, false});
  std::vector<Detection> a;
  std::vector<Detection> b;
  cf.decode(first, 0.3f, a);
  cl.decode(last, 0.3f, b);
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].box, b[i].box);
    EXPECT_EQ(a[i].class_id, b[i].class_id);
    EXPECT_FLOAT_EQ(a[i].score, b[i].score);
  }
}

TEST(YoloDecoder, RejectsShortTensor) {
  YoloDecoder decoder({10, 2, true});
  std::vector<float> raw(10);
  std::vector<Detection> out;
  EXPECT_THROW(decoder.decode(raw, 0.5f, out), std::invalid_argument);
}

}  // namespace
}  // namespace takt

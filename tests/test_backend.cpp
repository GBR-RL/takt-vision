#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "takt/infer/backend.hpp"
#include "takt/infer/fake_backend.hpp"
#include "takt/vision/yolo_decoder.hpp"

namespace takt {
namespace {

// A user-defined engine: satisfies the concept structurally, no inheritance.
struct DoublingBackend {
  TensorSpec in{"x", {1, 3, 2, 2}};
  TensorSpec out{"y", {1, 12}};
  int calls = 0;
  [[nodiscard]] std::string_view name() const { return "doubling"; }
  [[nodiscard]] const TensorSpec& input_spec() const { return in; }
  [[nodiscard]] const TensorSpec& output_spec() const { return out; }
  void infer(std::span<const float> input, std::span<float> output) {
    ++calls;
    for (std::size_t i = 0; i < 12; ++i) output[i] = 2.0f * input[i];
  }
};

struct MissingInfer {
  TensorSpec spec;
  [[nodiscard]] std::string_view name() const { return "broken"; }
  [[nodiscard]] const TensorSpec& input_spec() const { return spec; }
  [[nodiscard]] const TensorSpec& output_spec() const { return spec; }
};

static_assert(InferenceBackend<DoublingBackend>);
static_assert(!InferenceBackend<MissingInfer>, "the concept must reject incomplete engines");
static_assert(!InferenceBackend<int>);

TEST(Backend, TypeErasureForwardsToTheEngine) {
  Backend backend = DoublingBackend{};
  EXPECT_EQ(backend.name(), "doubling");
  EXPECT_EQ(backend.input_spec().element_count(), 12u);
  std::vector<float> in(12, 1.5f);
  std::vector<float> out(12);
  backend.infer(in, out);
  EXPECT_FLOAT_EQ(out[11], 3.0f);

  Backend moved = std::move(backend);
  EXPECT_EQ(moved.name(), "doubling");
}

TEST(TensorSpec, ElementCountRejectsDynamicDims) {
  EXPECT_EQ((TensorSpec{"t", {1, 84, 8400}}).element_count(), 84u * 8400u);
  EXPECT_THROW(static_cast<void>((TensorSpec{"t", {-1, 3}}).element_count()), std::logic_error);
}

TEST(FakeBackend, EmitsDecodableMovingObjects) {
  FakeBackendOptions options;
  options.input_width = 320;
  options.input_height = 320;
  options.num_anchors = 2100;
  options.num_classes = 4;
  options.num_objects = 3;
  Backend backend = FakeBackend(options);
  EXPECT_EQ(backend.output_spec().shape, (std::vector<std::int64_t>{1, 8, 2100}));

  std::vector<float> in(backend.input_spec().element_count());
  std::vector<float> out(backend.output_spec().element_count());
  YoloDecoder decoder(yolo_layout_from_shape(backend.output_spec().shape));
  std::vector<Detection> dets;

  backend.infer(in, out);
  decoder.decode(out, 0.5f, dets);
  ASSERT_EQ(dets.size(), 3u);
  const auto expected = FakeBackend::object_box(options, 0, 0);
  EXPECT_FLOAT_EQ(dets[0].box.center_x(), expected[0]);

  backend.infer(in, out);
  std::vector<Detection> next;
  decoder.decode(out, 0.5f, next);
  ASSERT_EQ(next.size(), 3u);
  EXPECT_NE(next[0].box, dets[0].box) << "objects move between calls";
}

TEST(FakeBackend, SimulatesLatency) {
  FakeBackendOptions options;
  options.latency = std::chrono::milliseconds(20);
  FakeBackend backend(options);
  std::vector<float> in(backend.input_spec().element_count());
  std::vector<float> out(backend.output_spec().element_count());
  const auto start = std::chrono::steady_clock::now();
  backend.infer(in, out);
  EXPECT_GE(std::chrono::steady_clock::now() - start, std::chrono::milliseconds(20));
}

}  // namespace
}  // namespace takt

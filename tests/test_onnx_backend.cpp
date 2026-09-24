// End-to-end test of the ONNX Runtime backend with a tiny generated model whose output is a known
// constant YOLO head (scripts/make_tiny_model.py). Checks tensor binding, shape discovery and the
// whole pipeline down to box coordinates in source pixels.

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <mutex>
#include <vector>

#include "takt/infer/onnx_backend.hpp"
#include "takt/io/synthetic_source.hpp"
#include "takt/pipeline/pipeline.hpp"

namespace takt {
namespace {

const std::filesystem::path kModel = TAKT_TEST_MODEL;

class OnnxBackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!std::filesystem::exists(kModel)) GTEST_SKIP() << "test model missing: pip install onnx";
  }
};

TEST_F(OnnxBackendTest, LoadsModelAndReportsShapes) {
  OnnxBackend backend(OnnxBackendOptions{kModel});
  EXPECT_EQ(backend.name(), "onnxruntime-cpu");
  EXPECT_EQ(backend.input_spec().name, "images");
  EXPECT_EQ(backend.input_spec().shape, (std::vector<std::int64_t>{1, 3, 64, 64}));
  EXPECT_EQ(backend.output_spec().shape, (std::vector<std::int64_t>{1, 6, 21}));
}

TEST_F(OnnxBackendTest, InfersIntoCallerBuffers) {
  OnnxBackend backend(OnnxBackendOptions{kModel});
  std::vector<float> input(backend.input_spec().element_count(), 0.3f);
  std::vector<float> output(backend.output_spec().element_count(), -1.0f);
  backend.infer(input, output);
  constexpr std::size_t n = 21;
  EXPECT_FLOAT_EQ(output[0 * n + 3], 32.0f);  // cx of anchor 3
  EXPECT_FLOAT_EQ(output[5 * n + 3], 0.9f);   // class-1 score of anchor 3
  EXPECT_FLOAT_EQ(output[4 * n + 0], 0.01f);  // background score
}

TEST_F(OnnxBackendTest, RejectsUnknownExecutionProvider) {
  OnnxBackendOptions options{kModel};
  options.execution_provider = "tpu";
  EXPECT_THROW(OnnxBackend{options}, std::invalid_argument);
}

TEST_F(OnnxBackendTest, MissingModelFailsWithUsefulMessage) {
  try {
    OnnxBackend backend(OnnxBackendOptions{"does_not_exist.onnx"});
    FAIL() << "expected an exception";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("does_not_exist.onnx"), std::string::npos);
  }
}

class CaptureSink final : public FrameSink {
 public:
  void consume(const Frame& frame) override { last = frame.detections; }
  std::vector<Detection> last;
};

TEST_F(OnnxBackendTest, PipelineProducesSourceSpaceBoxes) {
  PipelineConfig config;
  config.enable_tracking = false;
  config.ingress_policy = IngressPolicy::kBlock;
  Pipeline pipeline(config, OnnxBackend(OnnxBackendOptions{kModel}));
  auto sink = std::make_unique<CaptureSink>();
  CaptureSink* capture = sink.get();
  pipeline.add_sink(std::move(sink));

  // 128x128 source -> 64x64 model input: scale 0.5, no padding.
  SyntheticSource source(SyntheticSourceOptions{128, 128, 0.0, 3, 1});
  const RunReport report = pipeline.run(source);
  ASSERT_EQ(report.frames_completed, 3u);

  // Anchor 7 overlaps anchor 3 (same class, lower score) and must be removed by NMS.
  ASSERT_EQ(capture->last.size(), 2u);
  const Detection& a = capture->last[0];
  EXPECT_EQ(a.class_id, 1);
  EXPECT_FLOAT_EQ(a.score, 0.9f);
  EXPECT_EQ(a.box, (Box{44.0f, 54.0f, 84.0f, 74.0f}));
  const Detection& b = capture->last[1];
  EXPECT_EQ(b.class_id, 0);
  EXPECT_FLOAT_EQ(b.score, 0.6f);
  EXPECT_EQ(b.box, (Box{12.0f, 12.0f, 28.0f, 28.0f}));
}

}  // namespace
}  // namespace takt

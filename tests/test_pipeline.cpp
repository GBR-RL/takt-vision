#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "takt/infer/fake_backend.hpp"
#include "takt/io/synthetic_source.hpp"
#include "takt/pipeline/pipeline.hpp"

namespace takt {
namespace {

struct Collected {
  std::mutex mutex;
  std::vector<std::uint64_t> sequences;
  std::size_t frames_with_tracks = 0;
  int starts = 0;
  int finishes = 0;
};

class CollectingSink final : public FrameSink {
 public:
  explicit CollectingSink(Collected& out, std::uint64_t throw_at = UINT64_MAX)
      : out_(out), throw_at_(throw_at) {}
  void on_start(const RunInfo& info) override {
    EXPECT_NE(info.metrics, nullptr);
    std::scoped_lock lock(out_.mutex);
    ++out_.starts;
  }
  void consume(const Frame& frame) override {
    // >= rather than ==: under a dropping policy the exact frame may never arrive.
    if (frame.sequence >= throw_at_) throw std::runtime_error("sink failure");
    std::scoped_lock lock(out_.mutex);
    out_.sequences.push_back(frame.sequence);
    if (!frame.tracks.empty()) ++out_.frames_with_tracks;
  }
  void on_finish() override {
    std::scoped_lock lock(out_.mutex);
    ++out_.finishes;
  }

 private:
  Collected& out_;
  std::uint64_t throw_at_;
};

FakeBackend small_backend(std::chrono::microseconds latency = std::chrono::microseconds(0)) {
  FakeBackendOptions options;
  options.input_width = 160;
  options.input_height = 160;
  options.num_anchors = 525;
  options.num_classes = 2;
  options.latency = latency;
  return FakeBackend(options);
}

SyntheticSource unpaced_source(std::uint64_t frames) {
  return SyntheticSource(SyntheticSourceOptions{320, 240, 0.0, frames, 3});
}

bool strictly_increasing(const std::vector<std::uint64_t>& v) {
  for (std::size_t i = 1; i < v.size(); ++i) {
    if (v[i] <= v[i - 1]) return false;
  }
  return true;
}

TEST(Pipeline, BlockPolicyDeliversEveryFrameInOrder) {
  PipelineConfig config;
  config.ingress_policy = IngressPolicy::kBlock;
  Pipeline pipeline(config, small_backend());
  Collected collected;
  pipeline.add_sink(std::make_unique<CollectingSink>(collected));

  auto source = unpaced_source(200);
  const RunReport report = pipeline.run(source);

  EXPECT_EQ(report.frames_captured, 200u);
  EXPECT_EQ(report.frames_completed, 200u);
  EXPECT_EQ(report.frames_dropped, 0u);
  ASSERT_EQ(collected.sequences.size(), 200u);
  for (std::uint64_t i = 0; i < 200; ++i) EXPECT_EQ(collected.sequences[i], i);
  EXPECT_GT(collected.frames_with_tracks, 150u) << "the fake objects should be tracked";
  EXPECT_EQ(collected.starts, 1);
  EXPECT_EQ(collected.finishes, 1);
  EXPECT_EQ(report.end_to_end.count, 200u);
  EXPECT_EQ(report.stages[to_index(StageId::kInference)].count, 200u);
}

TEST(Pipeline, LatestPolicyShedsLoadButKeepsOrderAndAccounting) {
  PipelineConfig config;
  config.ingress_policy = IngressPolicy::kLatest;
  // Inference slower than an unpaced source: the pipeline must drop, not queue.
  Pipeline pipeline(config, small_backend(std::chrono::milliseconds(2)));
  Collected collected;
  pipeline.add_sink(std::make_unique<CollectingSink>(collected));

  auto source = unpaced_source(300);
  const RunReport report = pipeline.run(source);

  EXPECT_EQ(report.frames_captured, 300u);
  EXPECT_GT(report.frames_dropped, 0u);
  EXPECT_EQ(report.frames_completed + report.frames_dropped, report.frames_captured)
      << "every captured frame is either delivered or counted as dropped";
  EXPECT_TRUE(strictly_increasing(collected.sequences));
  EXPECT_EQ(collected.sequences.back(), 299u) << "the newest frame is never the one dropped";
}

TEST(Pipeline, DropNewestPolicyAccountsForEveryFrame) {
  PipelineConfig config;
  config.ingress_policy = IngressPolicy::kDropNewest;
  config.ingress_capacity = 4;
  Pipeline pipeline(config, small_backend(std::chrono::milliseconds(2)));
  Collected collected;
  pipeline.add_sink(std::make_unique<CollectingSink>(collected));

  auto source = unpaced_source(300);
  const RunReport report = pipeline.run(source);
  EXPECT_GT(report.frames_dropped, 0u);
  EXPECT_EQ(report.frames_completed + report.frames_dropped, 300u);
  EXPECT_TRUE(strictly_increasing(collected.sequences));
}

TEST(Pipeline, SequentialModeProcessesEveryFrame) {
  Pipeline pipeline(PipelineConfig{}, small_backend());
  Collected collected;
  pipeline.add_sink(std::make_unique<CollectingSink>(collected));
  auto source = unpaced_source(50);
  const RunReport report = pipeline.run_sequential(source);
  EXPECT_EQ(report.mode, "sequential");
  EXPECT_EQ(report.frames_completed, 50u);
  EXPECT_EQ(collected.sequences.size(), 50u);
}

TEST(Pipeline, MaxFramesLimitsTheRun) {
  PipelineConfig config;
  config.max_frames = 25;
  config.ingress_policy = IngressPolicy::kBlock;
  Pipeline pipeline(config, small_backend());
  auto source = unpaced_source(0);  // endless
  EXPECT_EQ(pipeline.run(source).frames_completed, 25u);
}

TEST(Pipeline, RequestStopEndsAnEndlessRun) {
  Pipeline pipeline(PipelineConfig{}, small_backend());
  SyntheticSource source(SyntheticSourceOptions{320, 240, 200.0, 0, 3});
  std::jthread stopper([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    pipeline.request_stop();
  });
  const RunReport report = pipeline.run(source);
  EXPECT_GT(report.frames_completed, 0u);
}

TEST(Pipeline, RequestStopReleasesACaptureThreadBlockedOnAFullQueue) {
  PipelineConfig config;
  config.ingress_policy = IngressPolicy::kBlock;
  Pipeline pipeline(config, small_backend(std::chrono::milliseconds(20)));
  auto source = unpaced_source(0);
  std::jthread stopper([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    pipeline.request_stop();
  });
  const RunReport report = pipeline.run(source);  // must return, not deadlock
  EXPECT_EQ(report.frames_completed + report.frames_dropped, report.frames_captured);
}

TEST(Pipeline, StageFailureIsRethrownWithoutDeadlock) {
  PipelineConfig config;
  config.ingress_policy = IngressPolicy::kBlock;
  Pipeline pipeline(config, small_backend());
  Collected collected;
  pipeline.add_sink(std::make_unique<CollectingSink>(collected, /*throw_at=*/10));
  auto source = unpaced_source(0);  // endless: only the failure can end the run
  EXPECT_THROW(static_cast<void>(pipeline.run(source)), std::runtime_error);
  EXPECT_EQ(collected.finishes, 1) << "sinks are finished even when a run fails";

  // The same pipeline is reusable after a failed run (5 frames stay below the failing one).
  auto second = unpaced_source(5);
  EXPECT_EQ(pipeline.run(second).frames_completed, 5u);
}

TEST(Pipeline, StatsCallbackRunsPeriodically) {
  PipelineConfig config;
  config.stats_interval = std::chrono::milliseconds(10);
  std::atomic<int> calls{0};
  config.on_stats = [&calls](const PipelineMetrics&) { ++calls; };
  Pipeline pipeline(config, small_backend());
  SyntheticSource source(SyntheticSourceOptions{320, 240, 100.0, 20, 3});
  static_cast<void>(pipeline.run(source));
  EXPECT_GE(calls.load(), 3);
}

TEST(Pipeline, ReportSerialisesToJson) {
  PipelineConfig config;
  config.ingress_policy = IngressPolicy::kBlock;  // no drops, so exactly 10 frames complete
  Pipeline pipeline(config, small_backend());
  auto source = unpaced_source(10);
  const RunReport report = pipeline.run(source);
  const std::string json = report.to_json();
  EXPECT_NE(json.find("\"frames_completed\":10"), std::string::npos) << json;
  EXPECT_NE(json.find("\"end_to_end\":{\"count\":10"), std::string::npos) << json;
  EXPECT_NE(report.to_text().find("end-to-end"), std::string::npos);
}

TEST(Pipeline, RejectsZeroCapacityQueues) {
  PipelineConfig config;
  config.stage_queue_capacity = 0;
  EXPECT_THROW(static_cast<void>(Pipeline(config, small_backend())), std::invalid_argument);
}

}  // namespace
}  // namespace takt

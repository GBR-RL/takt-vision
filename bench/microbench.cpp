// Micro-benchmarks of the hot paths, isolated from inference:
//   ./takt_microbench --benchmark_filter=Letterbox

#include <benchmark/benchmark.h>

#include <random>
#include <thread>
#include <vector>

#include "takt/core/latency_histogram.hpp"
#include "takt/core/spsc_ring.hpp"
#include "takt/io/synthetic_source.hpp"
#include "takt/track/byte_tracker.hpp"
#include "takt/vision/letterbox.hpp"
#include "takt/vision/nms.hpp"
#include "takt/vision/yolo_decoder.hpp"

namespace takt {
namespace {

void BM_Letterbox(benchmark::State& state) {
  const int width = static_cast<int>(state.range(0));
  const int height = width * 9 / 16;
  SyntheticSource source(SyntheticSourceOptions{width, height, 0.0, 1, 4});
  Image image;
  static_cast<void>(source.read(image));
  Letterboxer letterboxer(640, 640);
  std::vector<float> tensor(letterboxer.tensor_size());
  for (auto _ : state) {
    benchmark::DoNotOptimize(letterboxer.run(image.view(), tensor));
    benchmark::ClobberMemory();
  }
  state.SetLabel(std::to_string(width) + "x" + std::to_string(height) + " -> 640x640");
}
BENCHMARK(BM_Letterbox)->Arg(1280)->Arg(1920)->Unit(benchmark::kMicrosecond);

// YOLO11 head at 640x640: 84 x 8400 floats with realistic (mostly tiny) scores.
std::vector<float> random_head(std::size_t anchors, std::size_t classes) {
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> coord(0.0f, 640.0f);
  std::exponential_distribution<float> score(40.0f);
  std::vector<float> head((4 + classes) * anchors);
  for (std::size_t i = 0; i < 4 * anchors; ++i)
    head[i] = coord(rng) * (i >= 2 * anchors ? 0.1f : 1.0f);
  for (std::size_t i = 4 * anchors; i < head.size(); ++i) head[i] = std::min(1.0f, score(rng));
  return head;
}

void BM_DecodeAndNms(benchmark::State& state) {
  const auto head = random_head(8400, 80);
  YoloDecoder decoder({8400, 80, true});
  Nms nms;
  std::vector<Detection> detections;
  for (auto _ : state) {
    decoder.decode(head, 0.25f, detections);
    nms.apply(detections);
    benchmark::DoNotOptimize(detections.data());
  }
  state.counters["detections"] = static_cast<double>(detections.size());
}
BENCHMARK(BM_DecodeAndNms)->Unit(benchmark::kMicrosecond);

void BM_ByteTrackerUpdate(benchmark::State& state) {
  const int objects = static_cast<int>(state.range(0));
  ByteTracker tracker;
  std::vector<Detection> detections;
  std::vector<TrackedObject> tracks;
  int frame = 0;
  for (auto _ : state) {
    detections.clear();
    for (int i = 0; i < objects; ++i) {
      const float x = static_cast<float>((i % 10) * 120 + frame % 50);
      const float y = static_cast<float>((i / 10) * 150);
      detections.push_back({{x, y, x + 60, y + 120}, 0.8f, 0});
    }
    tracker.update(detections, tracks);
    ++frame;
  }
}
BENCHMARK(BM_ByteTrackerUpdate)->Arg(10)->Arg(50)->Unit(benchmark::kMicrosecond);

void BM_SpscRingTransfer(benchmark::State& state) {
  constexpr std::uint64_t kItems = 1 << 16;
  for (auto _ : state) {
    SpscRing<std::uint64_t> ring(64);
    std::jthread producer([&] {
      for (std::uint64_t i = 0; i < kItems; ++i) {
        std::uint64_t v = i;
        static_cast<void>(ring.push(v));
      }
      ring.close();
    });
    std::uint64_t sum = 0;
    while (auto v = ring.pop()) sum += *v;
    benchmark::DoNotOptimize(sum);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(kItems));
}
BENCHMARK(BM_SpscRingTransfer)->Unit(benchmark::kMillisecond)->UseRealTime();

void BM_HistogramRecord(benchmark::State& state) {
  LatencyHistogram histogram;
  std::uint64_t v = 1;
  for (auto _ : state) {
    histogram.record_us(v);
    v = (v * 2654435761u) % 100000;
  }
}
BENCHMARK(BM_HistogramRecord);

}  // namespace
}  // namespace takt

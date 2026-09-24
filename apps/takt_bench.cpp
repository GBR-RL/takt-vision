// takt_bench - per-stage latency of the C++ implementation on one thread, in the same JSON schema
// as scripts/python_baseline.py, so the two can be compared stage by stage.
//
//   takt_bench --model yolo11n.onnx --image bus.jpg --iterations 500 --json results/cpp.json

#include <CLI/CLI.hpp>

#include <cstdio>
#include <format>
#include <fstream>
#include <iostream>
#include <vector>

#include "app_support.hpp"
#include "takt/core/latency_histogram.hpp"
#include "takt/io/synthetic_source.hpp"
#include "takt/pipeline/metrics.hpp"
#include "takt/vision/letterbox.hpp"
#include "takt/vision/nms.hpp"
#include "takt/vision/yolo_decoder.hpp"

namespace {

struct Options {
  takt::app::BackendSettings backend;
  std::string image;
  int iterations = 500;
  int warmup = 50;
  float conf = 0.25f;
  float iou = 0.7f;
  std::string json;
};

template <class Fn>
void measure(takt::LatencyHistogram& histogram, Fn&& fn) {
  const takt::TimePoint start = takt::now();
  fn();
  histogram.record(takt::now() - start);
}

int run(const Options& o) {
  using namespace takt;

  Backend backend = app::make_backend(o.backend);
  const auto& in_shape = backend.input_spec().shape;
  const int width = static_cast<int>(in_shape.at(3));
  const int height = static_cast<int>(in_shape.at(2));

  Image image;
  std::string image_desc;
  if (!o.image.empty()) {
    app::SourceSettings settings;
    settings.uri = o.image;
    auto source = app::make_source(settings);
    if (!source->read(image)) throw std::runtime_error("cannot read " + o.image);
    image_desc = o.image;
  } else {
    SyntheticSource source(SyntheticSourceOptions{1280, 720, 0.0, 1, 4});
    static_cast<void>(source.read(image));
    image_desc = "synthetic 1280x720";
  }

  Letterboxer letterboxer(width, height);
  YoloDecoder decoder(yolo_layout_from_shape(backend.output_spec().shape));
  Nms nms(NmsOptions{o.iou, 300, 30000, false});
  std::vector<float> input(backend.input_spec().element_count());
  std::vector<float> output(backend.output_spec().element_count());
  std::vector<Detection> detections;
  LetterboxTransform transform;

  LatencyHistogram pre;
  LatencyHistogram inference;
  LatencyHistogram post;
  LatencyHistogram total;

  for (int i = 0; i < o.warmup + o.iterations; ++i) {
    const bool record = i >= o.warmup;
    const TimePoint frame_start = now();
    const TimePoint t0 = now();
    transform = letterboxer.run(image.view(), input);
    const TimePoint t1 = now();
    backend.infer(input, output);
    const TimePoint t2 = now();
    decoder.decode(output, o.conf, detections);
    nms.apply(detections);
    scale_to_source(detections, transform);
    const TimePoint t3 = now();
    if (record) {
      pre.record(t1 - t0);
      inference.record(t2 - t1);
      post.record(t3 - t2);
      total.record(t3 - frame_start);
    }
  }

  std::cout << std::format("takt_bench | {} | {} | {}x{} | {} iterations (+{} warm-up)\n",
                           backend.name(), image_desc, width, height, o.iterations, o.warmup);
  std::cout << std::format("  {:<12} {:>9} {:>9} {:>9} {:>9}\n", "stage (ms)", "mean", "p50", "p95",
                           "p99");
  const auto row = [](const char* name, const LatencySummary& s) {
    std::cout << std::format("  {:<12} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f}\n", name, s.mean_ms,
                             s.p50_ms, s.p95_ms, s.p99_ms);
  };
  row("preprocess", pre.summarize());
  row("inference", inference.summarize());
  row("postprocess", post.summarize());
  row("total", total.summarize());
  std::cout << std::format("  detections on last frame: {}\n", detections.size());

  if (!o.json.empty()) {
    std::ofstream out(o.json);
    out << std::format(
               R"({{"implementation":"cpp","backend":"{}","image":"{}","width":{},"height":{},"iterations":{},)"
               R"("detections":{},"stages":{{"preprocess":{},"inference":{},"postprocess":{},"total":{}}}}})",
               backend.name(), image_desc, width, height, o.iterations, detections.size(),
               to_json(pre.summarize()), to_json(inference.summarize()), to_json(post.summarize()),
               to_json(total.summarize()))
        << '\n';
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App cli{"takt_bench - per-stage latency of the C++ pipeline stages"};
  Options o;
  cli.add_option("-m,--model", o.backend.model, "ONNX model");
  cli.add_option("--backend", o.backend.kind, "onnx | fake");
  cli.add_option("--ep", o.backend.execution_provider, "cpu | cuda")->capture_default_str();
  cli.add_option("--threads", o.backend.threads, "ONNX Runtime intra-op threads (0 = default)");
  cli.add_flag("--spin", o.backend.spin, "let ONNX Runtime worker threads spin-wait");
  cli.add_option("--fake-latency-ms", o.backend.fake_latency_ms)->capture_default_str();
  cli.add_option("--image", o.image, "input image (default: synthetic 1280x720 frame)");
  cli.add_option("-n,--iterations", o.iterations)->capture_default_str();
  cli.add_option("--warmup", o.warmup)->capture_default_str();
  cli.add_option("--conf", o.conf)->capture_default_str();
  cli.add_option("--iou", o.iou)->capture_default_str();
  cli.add_option("--json", o.json, "write results as JSON");
  CLI11_PARSE(cli, argc, argv);
  try {
    return run(o);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}

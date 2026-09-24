// takt_run - run the real-time pipeline on a camera, video, image or synthetic source.
//
//   takt_run --model yolo11n.onnx --source video.mp4 --out-video annotated.mp4
//   takt_run --model yolo11n.onnx --source 0 --show                 # live webcam
//   takt_run --backend fake --fake-latency-ms 45 --policy block     # overload experiment

#include <CLI/CLI.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <exception>
#include <format>
#include <fstream>
#include <iostream>
#include <thread>

#include "app_support.hpp"
#include "takt/io/json_lines_sink.hpp"
#include "takt/pipeline/pipeline.hpp"

#ifdef TAKT_HAS_OPENCV
#include "takt/io/overlay_sink.hpp"
#endif

namespace {

std::atomic<bool> g_interrupted{false};  // lock-free, so safe to set from a signal handler

void on_signal(int /*signal*/) {
  g_interrupted.store(true);
}

struct Options {
  takt::app::BackendSettings backend;
  takt::app::SourceSettings source;
  std::string policy = "latest";
  std::size_t ingress_capacity = 2;
  std::size_t stage_queue = 1;
  float conf = 0.25f;
  float iou = 0.7f;
  std::size_t max_det = 300;
  bool agnostic = false;
  bool no_track = false;
  std::uint64_t max_frames = 0;
  bool sequential = false;
  int warmup = 3;
  int stats_ms = 1000;
  std::string out_video;
  std::string out_image;
  bool show = false;
  std::string names;
  std::string jsonl;
  std::string report;
};

// One status line per interval: throughput from the completion count, latency from histograms.
auto make_stats_printer() {
  return [last_count = std::uint64_t{0},
          last_time = takt::now()](const takt::PipelineMetrics& m) mutable {
    const auto t = takt::now();
    const std::uint64_t count = m.frames_completed.load();
    const double seconds = std::chrono::duration<double>(t - last_time).count();
    const double fps = seconds > 0 ? static_cast<double>(count - last_count) / seconds : 0.0;
    last_count = count;
    last_time = t;
    std::cerr << std::format(
        "\r[takt] {:6.1f} fps | e2e p50 {:6.1f} p99 {:6.1f} ms | infer p50 {:6.1f} ms | "
        "frames {} dropped {}   ",
        fps, m.end_to_end().percentile_ms(0.5), m.end_to_end().percentile_ms(0.99),
        m.stage(takt::StageId::kInference).percentile_ms(0.5), count, m.frames_dropped.load());
  };
}

int run(const Options& o) {
  using namespace takt;

  PipelineConfig config;
  const auto policy = parse_ingress_policy(o.policy);
  if (!policy) throw std::invalid_argument("unknown --policy " + o.policy);
  config.ingress_policy = *policy;
  config.ingress_capacity = o.ingress_capacity;
  config.stage_queue_capacity = o.stage_queue;
  config.score_threshold = o.conf;
  config.nms.iou_threshold = o.iou;
  config.nms.max_detections = o.max_det;
  config.nms.class_agnostic = o.agnostic;
  config.enable_tracking = !o.no_track;
  config.max_frames = o.max_frames;
  config.warmup_iterations = o.warmup;
  if (o.stats_ms > 0) {
    config.stats_interval = std::chrono::milliseconds(o.stats_ms);
    config.on_stats = make_stats_printer();
  }

  auto source = app::make_source(o.source);
  Pipeline pipeline(std::move(config), app::make_backend(o.backend));

  if (!o.jsonl.empty()) pipeline.add_sink(std::make_unique<JsonLinesSink>(o.jsonl));
  if (!o.out_video.empty() || !o.out_image.empty() || o.show) {
#ifdef TAKT_HAS_OPENCV
    OverlaySinkOptions sink;
    sink.video_path = o.out_video;
    sink.image_path = o.out_image;
    sink.show_window = o.show;
    if (!o.names.empty()) {
      sink.overlay.class_names = load_class_names(o.names);
    } else if (o.backend.kind != "fake") {
      sink.overlay.class_names = coco_class_names();
    }
    sink.on_quit = [&pipeline] { pipeline.request_stop(); };
    pipeline.add_sink(std::make_unique<OverlaySink>(std::move(sink)));
#else
    throw std::invalid_argument("--out-video/--out-image/--show need a build with OpenCV");
#endif
  }

  std::cerr << std::format("[takt] {} | backend {} | source {} | model {}x{} | features: {}\n",
                           o.sequential ? "sequential" : "pipelined", pipeline.backend_name(),
                           source->describe(), pipeline.model_width(), pipeline.model_height(),
                           app::build_features());

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  // Signal handlers may only touch lock-free atomics; a watcher thread turns the flag into a
  // proper stop request.
  std::jthread interrupt_watcher([&pipeline](std::stop_token stop) {
    while (!stop.stop_requested()) {
      if (g_interrupted.load()) {
        pipeline.request_stop();
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });

  const RunReport report = o.sequential ? pipeline.run_sequential(*source) : pipeline.run(*source);
  interrupt_watcher.request_stop();

  std::cerr << '\n';
  std::cout << report.to_text();
  if (!o.report.empty()) {
    std::ofstream out(o.report);
    out << report.to_json() << '\n';
    std::cerr << "[takt] report written to " << o.report << '\n';
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App cli{"takt_run - real-time C++20 vision inference pipeline"};
  Options o;

  cli.add_option("-m,--model", o.backend.model, "ONNX model (YOLOv8 / YOLO11 detection export)");
  cli.add_option("-s,--source", o.source.uri,
                 "camera index, video, image, RTSP/GStreamer URI, or 'synthetic'")
      ->capture_default_str();
  cli.add_option("--backend", o.backend.kind, "onnx | fake (default: onnx if --model is given)");
  cli.add_option("--ep", o.backend.execution_provider,
                 "ONNX Runtime execution provider: cpu | cuda")
      ->capture_default_str();
  cli.add_option("--threads", o.backend.threads, "ONNX Runtime intra-op threads (0 = default)");
  cli.add_flag("--spin", o.backend.spin, "let ONNX Runtime worker threads spin-wait");

  cli.add_option("--policy", o.policy, "ingress policy: latest | drop-newest | block")
      ->capture_default_str();
  cli.add_option("--ingress-capacity", o.ingress_capacity, "FIFO capacity for drop-newest / block")
      ->capture_default_str();
  cli.add_option("--stage-queue", o.stage_queue, "queue depth between internal stages")
      ->capture_default_str();
  cli.add_option("--conf", o.conf, "score threshold")->capture_default_str();
  cli.add_option("--iou", o.iou, "NMS IoU threshold")->capture_default_str();
  cli.add_option("--max-det", o.max_det, "maximum detections per frame")->capture_default_str();
  cli.add_flag("--agnostic", o.agnostic, "class-agnostic NMS");
  cli.add_flag("--no-track", o.no_track, "disable ByteTrack tracking");
  cli.add_option("--max-frames", o.max_frames, "stop after N frames (0 = until the source ends)");
  cli.add_flag("--sequential", o.sequential, "single-threaded baseline instead of the pipeline");
  cli.add_option("--warmup", o.warmup, "untimed warm-up inferences")->capture_default_str();
  cli.add_option("--stats-ms", o.stats_ms, "live status interval in ms (0 = off)")
      ->capture_default_str();

  cli.add_flag("!--no-realtime", o.source.realtime,
               "process files as fast as possible (no pacing)");
  cli.add_flag("--loop", o.source.loop, "loop video files / images");
  cli.add_option("--synthetic-width", o.source.synthetic_width)->capture_default_str();
  cli.add_option("--synthetic-height", o.source.synthetic_height)->capture_default_str();
  cli.add_option("--synthetic-fps", o.source.synthetic_fps)->capture_default_str();

  cli.add_option("--fake-input", o.backend.fake_input, "fake backend input size")
      ->capture_default_str();
  cli.add_option("--fake-latency-ms", o.backend.fake_latency_ms, "fake backend inference time")
      ->capture_default_str();
  cli.add_option("--fake-jitter-ms", o.backend.fake_jitter_ms, "fake backend extra random time")
      ->capture_default_str();

  cli.add_option("--out-video", o.out_video, "write annotated video (.mp4 / .avi)");
  cli.add_option("--out-image", o.out_image, "write the last annotated frame as an image");
  cli.add_flag("--show", o.show, "live preview window (q / Esc to stop)");
  cli.add_option("--names", o.names, "class names file, one per line (default: COCO)");
  cli.add_option("--jsonl", o.jsonl, "per-frame detections, tracks and latency as JSON lines");
  cli.add_option("--report", o.report, "run summary as JSON");

  CLI11_PARSE(cli, argc, argv);
  try {
    return run(o);
  } catch (const std::exception& e) {
    std::cerr << "\nerror: " << e.what() << '\n';
    return 1;
  }
}

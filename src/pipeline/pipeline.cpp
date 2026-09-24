#include "takt/pipeline/pipeline.hpp"

#include <algorithm>
#include <bit>
#include <condition_variable>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace takt {
namespace {

int input_dimension(const Backend& backend, std::size_t axis) {
  const auto& shape = backend.input_spec().shape;
  if (shape.size() != 4 || shape[1] != 3) {
    throw std::invalid_argument("Pipeline: backend input must be NCHW with 3 channels");
  }
  if (shape[axis] <= 0) throw std::invalid_argument("Pipeline: backend input has a dynamic size");
  return static_cast<int>(shape[axis]);
}

template <class Fn>
void timed(PipelineMetrics& metrics, Frame& frame, StageId id, Fn&& fn) {
  const TimePoint start = now();
  std::forward<Fn>(fn)();
  const Nanos elapsed = now() - start;
  frame.service_time[to_index(id)] = elapsed;
  metrics.stage(id).record(elapsed);
}

// Queues and error state shared by the threads of one run().
struct RunContext {
  RunContext(const PipelineConfig& config, std::stop_source stop_source,
             PipelineMetrics& run_metrics)
      : ingress(config.ingress_policy, config.ingress_capacity),
        to_infer(internal_policy(config), config.stage_queue_capacity),
        to_post(internal_policy(config), config.stage_queue_capacity),
        to_sink(internal_policy(config), config.stage_queue_capacity),
        stop(std::move(stop_source)),
        metrics(run_metrics) {}

  // kLatest sheds load at every boundary, so frames never wait in front of the bottleneck; the
  // FIFO policies block internally so every admitted frame is processed.
  static IngressPolicy internal_policy(const PipelineConfig& config) noexcept {
    return config.ingress_policy == IngressPolicy::kLatest ? IngressPolicy::kLatest
                                                           : IngressPolicy::kBlock;
  }

  // First failure wins. Closing every queue unblocks all stages; they drain the frames still in
  // flight without processing them, which returns those frames to the pool.
  void abort(std::exception_ptr failure) noexcept {
    {
      std::scoped_lock lock(error_mutex);
      if (!error) error = std::move(failure);
    }
    aborted.store(true, std::memory_order_release);
    stop.request_stop();
    ingress.close();
    to_infer.close();
    to_post.close();
    to_sink.close();
  }

  [[nodiscard]] bool is_aborted() const noexcept { return aborted.load(std::memory_order_acquire); }

  HandOff ingress;
  HandOff to_infer;
  HandOff to_post;
  HandOff to_sink;
  std::stop_source stop;
  PipelineMetrics& metrics;
  std::mutex error_mutex;
  std::exception_ptr error;
  std::atomic<bool> aborted{false};
};

// Body of every stage thread: pop, process, pass on. When the upstream queue closes and drains,
// close the downstream queue - shutdown ripples through the pipeline in order, and every frame
// admitted before shutdown is still delivered.
template <class Body>
void run_stage(RunContext& ctx, HandOff& input, HandOff* output, Body&& body) noexcept {
  try {
    while (std::optional<FramePtr> frame = input.pop()) {
      if (ctx.is_aborted()) continue;  // drop: the FramePtr returns to the pool here
      body(**frame);
      if (output != nullptr) {
        // A displaced stale frame returns to the pool inside push(); a kClosed result means the
        // run is aborting and this frame is released the same way.
        if (is_drop(output->push(std::move(*frame)))) {
          ctx.metrics.frames_dropped.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }
  } catch (...) {
    ctx.abort(std::current_exception());
  }
  if (output != nullptr) output->close();
}

// Clears `running` when a run ends, however it ends.
class RunGuard {
 public:
  explicit RunGuard(std::atomic<bool>& running) : running_(running) {
    if (running_.exchange(true)) throw std::logic_error("Pipeline: run() is already in progress");
  }
  RunGuard(const RunGuard&) = delete;
  RunGuard& operator=(const RunGuard&) = delete;
  ~RunGuard() { running_.store(false); }

 private:
  std::atomic<bool>& running_;
};

}  // namespace

Pipeline::Pipeline(PipelineConfig config, Backend backend)
    : config_(std::move(config)),
      backend_(std::move(backend)),
      model_width_(input_dimension(backend_, 3)),
      model_height_(input_dimension(backend_, 2)),
      decode_threshold_(config_.enable_tracking
                            ? std::min(config_.score_threshold, config_.tracker.low_threshold)
                            : config_.score_threshold),
      letterboxer_(model_width_, model_height_),
      decoder_(yolo_layout_from_shape(backend_.output_spec().shape)),
      nms_(config_.nms),
      tracker_(config_.tracker) {
  if (config_.stage_queue_capacity == 0 || config_.ingress_capacity == 0) {
    throw std::invalid_argument("Pipeline: queue capacities must be positive");
  }
}

Pipeline::~Pipeline() = default;

void Pipeline::add_sink(std::unique_ptr<FrameSink> sink) {
  if (running_.load()) throw std::logic_error("Pipeline: cannot add sinks while running");
  if (sink) sinks_.push_back(std::move(sink));
}

void Pipeline::request_stop() noexcept {
  std::scoped_lock lock(stop_mutex_);
  stop_source_.request_stop();
}

std::size_t Pipeline::pool_size() const noexcept {
  if (config_.pool_size > 0) return config_.pool_size;
  const std::size_t ingress_slots = config_.ingress_policy == IngressPolicy::kLatest
                                        ? 3
                                        : std::bit_ceil(config_.ingress_capacity);
  const std::size_t per_stage = config_.ingress_policy == IngressPolicy::kLatest
                                    ? 3
                                    : std::bit_ceil(config_.stage_queue_capacity);
  const std::size_t stage_slots = 3 * per_stage;
  constexpr std::size_t kInHands = 5;  // one frame held by capture and by each stage thread
  return ingress_slots + stage_slots + kInHands + 1;
}

FrameBufferSizes Pipeline::buffer_sizes() const {
  return {backend_.input_spec().element_count(), backend_.output_spec().element_count(),
          config_.nms.max_detections};
}

std::stop_token Pipeline::begin_run(FrameSource& source, std::string_view mode) {
  metrics_.reset();
  tracker_.reset();

  // Warm-up: the first inferences pay for lazy initialisation, memory arenas and cache misses.
  // Keeping them out of the measurement is what makes p99 honest.
  if (config_.warmup_iterations > 0) {
    std::vector<float> input(backend_.input_spec().element_count(), 0.5f);
    std::vector<float> output(backend_.output_spec().element_count());
    for (int i = 0; i < config_.warmup_iterations; ++i) backend_.infer(input, output);
  }

  RunInfo info;
  info.started_at = now();
  info.backend = std::string(backend_.name());
  info.source = source.describe();
  info.policy = mode == "sequential" ? "none" : std::string(to_string(config_.ingress_policy));
  info.source_fps = source.nominal_fps();
  info.model_width = model_width_;
  info.model_height = model_height_;
  info.metrics = &metrics_;
  for (const auto& sink : sinks_) sink->on_start(info);

  std::scoped_lock lock(stop_mutex_);
  stop_source_ = std::stop_source{};
  return stop_source_.get_token();
}

RunReport Pipeline::finish_run(const FrameSource& source, std::string_view mode,
                               TimePoint started) {
  for (const auto& sink : sinks_) sink->on_finish();

  RunReport report;
  report.mode = std::string(mode);
  report.backend = std::string(backend_.name());
  report.source = source.describe();
  report.policy = mode == "sequential" ? "none" : std::string(to_string(config_.ingress_policy));
  report.model_width = model_width_;
  report.model_height = model_height_;
  report.frames_captured = metrics_.frames_captured.load();
  report.frames_dropped = metrics_.frames_dropped.load();
  report.frames_completed = metrics_.frames_completed.load();
  report.wall_seconds = std::chrono::duration<double>(now() - started).count();
  report.throughput_fps = report.wall_seconds > 0.0
                              ? static_cast<double>(report.frames_completed) / report.wall_seconds
                              : 0.0;
  for (std::size_t i = 0; i < kStageCount; ++i) {
    report.stages[i] = metrics_.stage(static_cast<StageId>(i)).summarize();
  }
  report.end_to_end = metrics_.end_to_end().summarize();
  return report;
}

std::jthread Pipeline::start_reporter() {
  if (!config_.on_stats || config_.stats_interval.count() <= 0) return {};
  return std::jthread([this](std::stop_token stop) {
    std::mutex mutex;
    std::condition_variable_any wake;
    std::unique_lock lock(mutex);
    for (;;) {
      // Sleeps for the interval, but wakes immediately when the jthread is asked to stop.
      static_cast<void>(wake.wait_for(lock, stop, config_.stats_interval, [] { return false; }));
      if (stop.stop_requested()) return;
      config_.on_stats(metrics_);
    }
  });
}

void Pipeline::preprocess(Frame& frame) {
  timed(metrics_, frame, StageId::kPreprocess,
        [&] { frame.letterbox = letterboxer_.run(frame.image.view(), frame.input); });
}

void Pipeline::infer(Frame& frame) {
  timed(metrics_, frame, StageId::kInference, [&] { backend_.infer(frame.input, frame.output); });
}

void Pipeline::postprocess(Frame& frame) {
  timed(metrics_, frame, StageId::kPostprocess, [&] {
    decoder_.decode(frame.output, decode_threshold_, frame.detections);
    nms_.apply(frame.detections);
    scale_to_source(frame.detections, frame.letterbox);
  });
}

void Pipeline::track(Frame& frame) {
  timed(metrics_, frame, StageId::kTracking, [&] {
    if (config_.enable_tracking) tracker_.update(frame.detections, frame.tracks);
    // The tracker also consumed low-score boxes; downstream only sees confident detections.
    if (decode_threshold_ < config_.score_threshold) {
      const float threshold = config_.score_threshold;
      std::erase_if(frame.detections,
                    [threshold](const Detection& d) { return d.score <= threshold; });
    }
  });
}

void Pipeline::deliver(Frame& frame) {
  timed(metrics_, frame, StageId::kSink, [&] {
    for (const auto& sink : sinks_) sink->consume(frame);
  });
  metrics_.end_to_end().record(now() - frame.captured_at);
  metrics_.frames_completed.fetch_add(1, std::memory_order_relaxed);
}

RunReport Pipeline::run(FrameSource& source) {
  RunGuard guard(running_);
  const std::stop_token stop = begin_run(source, "pipelined");
  std::stop_source stop_source;
  {
    std::scoped_lock lock(stop_mutex_);
    stop_source = stop_source_;
  }

  FramePool pool(pool_size(), buffer_sizes());
  RunContext ctx(config_, stop_source, metrics_);
  const TimePoint started = now();
  {
    // A stop request must also release a capture thread blocked on a full kBlock queue.
    const std::stop_callback close_ingress_on_stop(stop, [&ctx] { ctx.ingress.close(); });
    std::jthread reporter = start_reporter();

    std::jthread sink_thread(
        [&] { run_stage(ctx, ctx.to_sink, nullptr, [this](Frame& f) { deliver(f); }); });
    std::jthread post_thread([&] {
      run_stage(ctx, ctx.to_post, &ctx.to_sink, [this](Frame& f) {
        postprocess(f);
        track(f);
      });
    });
    std::jthread infer_thread(
        [&] { run_stage(ctx, ctx.to_infer, &ctx.to_post, [this](Frame& f) { infer(f); }); });
    std::jthread preprocess_thread(
        [&] { run_stage(ctx, ctx.ingress, &ctx.to_infer, [this](Frame& f) { preprocess(f); }); });

    // Capture runs on the calling thread.
    try {
      std::uint64_t sequence = 0;
      while (!stop.stop_requested() && !ctx.is_aborted() &&
             (config_.max_frames == 0 || sequence < config_.max_frames)) {
        FramePtr frame = pool.acquire(stop);
        if (!frame) break;
        if (!source.read(frame->image)) break;
        frame->captured_at = now();
        frame->sequence = sequence++;

        const HandOffResult result = ctx.ingress.push(std::move(frame));
        if (result == HandOffResult::kClosed) break;  // shutting down: never admitted, not counted
        metrics_.frames_captured.fetch_add(1, std::memory_order_relaxed);
        if (result != HandOffResult::kAccepted) {
          metrics_.frames_dropped.fetch_add(1, std::memory_order_relaxed);
        }
      }
    } catch (...) {
      ctx.abort(std::current_exception());
    }
    ctx.ingress.close();
    // Leaving this scope joins the stage threads in reverse declaration order (preprocess,
    // inference, postprocess, sink), then the reporter. Queues and pool outlive all of them.
  }

  if (ctx.error) {
    for (const auto& sink : sinks_) sink->on_finish();
    std::rethrow_exception(ctx.error);
  }
  return finish_run(source, "pipelined", started);
}

RunReport Pipeline::run_sequential(FrameSource& source) {
  RunGuard guard(running_);
  const std::stop_token stop = begin_run(source, "sequential");

  FramePool pool(1, buffer_sizes());
  const TimePoint started = now();
  for (std::uint64_t sequence = 0;
       !stop.stop_requested() && (config_.max_frames == 0 || sequence < config_.max_frames);) {
    FramePtr frame = pool.try_acquire();  // the single frame is back in the pool every iteration
    if (!source.read(frame->image)) break;
    frame->captured_at = now();
    frame->sequence = sequence++;
    metrics_.frames_captured.fetch_add(1, std::memory_order_relaxed);
    preprocess(*frame);
    infer(*frame);
    postprocess(*frame);
    track(*frame);
    deliver(*frame);
  }
  return finish_run(source, "sequential", started);
}

}  // namespace takt

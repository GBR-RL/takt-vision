#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string_view>
#include <thread>
#include <vector>

#include "takt/infer/backend.hpp"
#include "takt/pipeline/frame_pool.hpp"
#include "takt/pipeline/hand_off.hpp"
#include "takt/pipeline/metrics.hpp"
#include "takt/pipeline/source_sink.hpp"
#include "takt/track/byte_tracker.hpp"
#include "takt/vision/letterbox.hpp"
#include "takt/vision/nms.hpp"
#include "takt/vision/yolo_decoder.hpp"

namespace takt {

struct PipelineConfig {
  IngressPolicy ingress_policy = IngressPolicy::kLatest;
  std::size_t ingress_capacity = 2;  // FIFO capacity for kDropNewest / kBlock
  // Between internal stages. Every slot in front of the slowest stage is a frame waiting a full
  // inference period, so depth 1 gives the lowest latency; deeper queues only absorb jitter.
  std::size_t stage_queue_capacity = 1;
  std::size_t pool_size = 0;  // 0 = derived from the queue capacities

  float score_threshold = 0.25f;  // detections reported downstream
  NmsOptions nms;
  bool enable_tracking = true;
  ByteTrackerOptions tracker;

  std::uint64_t max_frames = 0;  // 0 = until the source is exhausted or stop is requested
  int warmup_iterations = 3;     // untimed inferences before the first frame

  // Optional periodic callback on a separate thread, e.g. to print live statistics.
  std::chrono::milliseconds stats_interval{0};
  std::function<void(const PipelineMetrics&)> on_stats;
};

// Camera-to-decision pipeline: capture -> preprocess -> inference -> postprocess -> tracking ->
// sinks, each stage on its own thread, connected by bounded lock-free SPSC queues.
//
// Pipelining means throughput is set by the slowest stage instead of the sum of all stages, while
// small queues keep the frames in flight - and therefore latency - bounded. Each stage object
// (letterboxer, backend, decoder, tracker) is touched by exactly one thread, so none of them needs
// a lock.
class Pipeline {
 public:
  Pipeline(PipelineConfig config, Backend backend);
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;
  Pipeline(Pipeline&&) = delete;
  Pipeline& operator=(Pipeline&&) = delete;
  ~Pipeline();

  void add_sink(std::unique_ptr<FrameSink> sink);

  // Runs with one thread per stage until the source ends, max_frames is reached or
  // request_stop() is called. Frames already admitted are drained, not discarded. Blocks; rethrows
  // the first exception raised on any stage thread.
  RunReport run(FrameSource& source);

  // Same stages on the calling thread, one frame at a time. The baseline that shows what
  // pipelining buys.
  RunReport run_sequential(FrameSource& source);

  // Thread-safe. Ends the current run gracefully.
  void request_stop() noexcept;

  [[nodiscard]] const PipelineMetrics& metrics() const noexcept { return metrics_; }
  [[nodiscard]] const PipelineConfig& config() const noexcept { return config_; }
  [[nodiscard]] std::string_view backend_name() const { return backend_.name(); }
  [[nodiscard]] int model_width() const noexcept { return model_width_; }
  [[nodiscard]] int model_height() const noexcept { return model_height_; }
  [[nodiscard]] std::size_t pool_size() const noexcept;

 private:
  std::stop_token begin_run(FrameSource& source, std::string_view mode);
  RunReport finish_run(const FrameSource& source, std::string_view mode, TimePoint started);
  [[nodiscard]] std::jthread start_reporter();
  [[nodiscard]] FrameBufferSizes buffer_sizes() const;

  void preprocess(Frame& frame);
  void infer(Frame& frame);
  void postprocess(Frame& frame);
  void track(Frame& frame);
  void deliver(Frame& frame);

  PipelineConfig config_;
  Backend backend_;
  int model_width_ = 0;
  int model_height_ = 0;
  float decode_threshold_ = 0.0f;
  Letterboxer letterboxer_;
  YoloDecoder decoder_;
  Nms nms_;
  ByteTracker tracker_;
  std::vector<std::unique_ptr<FrameSink>> sinks_;
  PipelineMetrics metrics_;

  std::mutex stop_mutex_;
  std::stop_source stop_source_;
  std::atomic<bool> running_{false};
};

}  // namespace takt

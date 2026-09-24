#pragma once

#include <string>

#include "takt/core/clock.hpp"
#include "takt/core/image.hpp"
#include "takt/pipeline/frame.hpp"

namespace takt {

class PipelineMetrics;

// Produces camera frames. read() runs on the pipeline's capture thread and may block to pace a
// real-time source (a camera naturally blocks until the next exposure is ready).
class FrameSource {
 public:
  FrameSource() = default;
  FrameSource(const FrameSource&) = delete;
  FrameSource& operator=(const FrameSource&) = delete;
  FrameSource(FrameSource&&) = delete;
  FrameSource& operator=(FrameSource&&) = delete;
  virtual ~FrameSource() = default;

  // Writes the next frame into `out` (reusing its buffer). Returns false at end of stream.
  virtual bool read(Image& out) = 0;
  [[nodiscard]] virtual std::string describe() const = 0;
  // Frames per second the source delivers when paced; 0 if unknown or unpaced.
  [[nodiscard]] virtual double nominal_fps() const { return 0.0; }
};

struct RunInfo {
  TimePoint started_at{};
  std::string backend;
  std::string source;
  std::string policy;
  double source_fps = 0.0;
  int model_width = 0;
  int model_height = 0;
  const PipelineMetrics* metrics = nullptr;  // live metrics, e.g. for an on-screen HUD
};

// Consumes finished frames on the pipeline's sink thread, in capture order.
class FrameSink {
 public:
  FrameSink() = default;
  FrameSink(const FrameSink&) = delete;
  FrameSink& operator=(const FrameSink&) = delete;
  FrameSink(FrameSink&&) = delete;
  FrameSink& operator=(FrameSink&&) = delete;
  virtual ~FrameSink() = default;

  virtual void on_start(const RunInfo& /*info*/) {}
  virtual void consume(const Frame& frame) = 0;
  virtual void on_finish() {}
};

}  // namespace takt

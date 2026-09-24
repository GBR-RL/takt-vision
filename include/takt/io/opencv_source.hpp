#pragma once

#include <memory>
#include <string>

#include "takt/core/clock.hpp"
#include "takt/pipeline/source_sink.hpp"

namespace takt {

struct OpenCvSourceOptions {
  // Camera index ("0"), video file, still image, RTSP URL or GStreamer pipeline (contains '!').
  std::string uri = "0";
  // Pace video files at their native frame rate, like a live camera. Off = decode as fast as
  // possible (offline batch processing).
  bool realtime = true;
  bool loop = false;      // restart video files at the end
  int request_width = 0;  // camera capture size request; 0 = driver default
  int request_height = 0;
};

// Frames from anything cv::VideoCapture can open, plus still images (delivered once, or forever
// with loop=true - handy for benchmarking a single test image).
class OpenCvSource final : public FrameSource {
 public:
  explicit OpenCvSource(OpenCvSourceOptions options);
  ~OpenCvSource() override;

  bool read(Image& out) override;
  [[nodiscard]] std::string describe() const override;
  [[nodiscard]] double nominal_fps() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace takt

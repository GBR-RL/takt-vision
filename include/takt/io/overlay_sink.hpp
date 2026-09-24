#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <string>

#include "takt/io/overlay.hpp"
#include "takt/pipeline/source_sink.hpp"

namespace takt {

struct OverlaySinkOptions {
  std::filesystem::path video_path;  // annotated video (.mp4 / .avi); empty = none
  std::filesystem::path image_path;  // annotated still of the latest frame; empty = none
  bool show_window = false;          // live preview window
  double video_fps = 0.0;            // 0 = the source's frame rate (or 30)
  std::string window_title = "takt-vision";
  OverlayOptions overlay;
  std::function<void()> on_quit;  // invoked when 'q' or Esc is pressed in the preview window
};

// Renders the overlay and writes it to a video file, an image file and/or a preview window.
class OverlaySink final : public FrameSink {
 public:
  explicit OverlaySink(OverlaySinkOptions options);
  ~OverlaySink() override;

  void on_start(const RunInfo& info) override;
  void consume(const Frame& frame) override;
  void on_finish() override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace takt

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <opencv2/core.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "takt/pipeline/source_sink.hpp"

namespace takt {

struct OverlayOptions {
  std::vector<std::string> class_names;  // empty = "class N"
  bool draw_hud = true;
  bool draw_trails = true;
  std::size_t trail_length = 40;
};

// The 80 COCO class names, in the order Ultralytics models use.
[[nodiscard]] std::vector<std::string> coco_class_names();
// One class name per line.
[[nodiscard]] std::vector<std::string> load_class_names(const std::filesystem::path& path);

// Draws detections, track IDs with motion trails, and a live telemetry HUD (FPS, end-to-end
// p50/p99, per-stage timings, drop count) onto a frame. This is what the demo videos show.
class OverlayRenderer {
 public:
  explicit OverlayRenderer(OverlayOptions options = {});

  void on_start(const RunInfo& info);
  // `canvas` must hold a BGR copy of frame.image; annotations are drawn onto it.
  void render(const Frame& frame, cv::Mat& canvas);

 private:
  void update_telemetry(const Frame& frame);
  void draw_hud(cv::Mat& canvas, std::size_t visible_objects) const;
  [[nodiscard]] std::string label_for(int class_id) const;

  OverlayOptions options_;
  RunInfo info_;
  std::unordered_map<int, std::deque<cv::Point>> trails_;
  std::unordered_map<int, std::uint64_t> last_seen_;

  // Telemetry, refreshed a few times per second rather than every frame.
  TimePoint last_frame_time_{};
  TimePoint last_refresh_{};
  double fps_ = 0.0;
  std::vector<std::string> hud_lines_;
  std::vector<double> stage_p50_ms_;
};

}  // namespace takt

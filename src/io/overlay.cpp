#include "takt/io/overlay.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

#include "takt/pipeline/metrics.hpp"

namespace takt {
namespace {

// Distinct, stable colour per track ID: hue stepped by the golden ratio.
cv::Scalar color_for(int id) {
  const double hue = std::fmod(static_cast<double>(id) * 0.618033988749895, 1.0) * 180.0;
  cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 200, 255));
  cv::Mat bgr;
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
  const auto px = bgr.at<cv::Vec3b>(0, 0);
  return {static_cast<double>(px[0]), static_cast<double>(px[1]), static_cast<double>(px[2])};
}

cv::Point to_point(float x, float y) {
  return {static_cast<int>(std::lround(x)), static_cast<int>(std::lround(y))};
}

void draw_label(cv::Mat& canvas, const std::string& text, cv::Point anchor, const cv::Scalar& color,
                double scale) {
  int baseline = 0;
  const int thickness = 1;
  const cv::Size size =
      cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
  const int top = std::max(0, anchor.y - size.height - baseline - 4);
  const cv::Rect box(anchor.x, top, size.width + 6, size.height + baseline + 4);
  cv::rectangle(canvas, box, color, cv::FILLED);
  cv::putText(canvas, text, {anchor.x + 3, top + size.height + 2}, cv::FONT_HERSHEY_SIMPLEX, scale,
              {20, 20, 20}, thickness, cv::LINE_AA);
}

constexpr std::array<StageId, 4> kHudStages = {StageId::kPreprocess, StageId::kInference,
                                               StageId::kPostprocess, StageId::kTracking};
const std::array<cv::Scalar, 4> kStageColors = {cv::Scalar(235, 180, 60), cv::Scalar(80, 120, 250),
                                                cv::Scalar(90, 210, 120),
                                                cv::Scalar(200, 110, 220)};

}  // namespace

std::vector<std::string> coco_class_names() {
  return {"person",        "bicycle",      "car",
          "motorcycle",    "airplane",     "bus",
          "train",         "truck",        "boat",
          "traffic light", "fire hydrant", "stop sign",
          "parking meter", "bench",        "bird",
          "cat",           "dog",          "horse",
          "sheep",         "cow",          "elephant",
          "bear",          "zebra",        "giraffe",
          "backpack",      "umbrella",     "handbag",
          "tie",           "suitcase",     "frisbee",
          "skis",          "snowboard",    "sports ball",
          "kite",          "baseball bat", "baseball glove",
          "skateboard",    "surfboard",    "tennis racket",
          "bottle",        "wine glass",   "cup",
          "fork",          "knife",        "spoon",
          "bowl",          "banana",       "apple",
          "sandwich",      "orange",       "broccoli",
          "carrot",        "hot dog",      "pizza",
          "donut",         "cake",         "chair",
          "couch",         "potted plant", "bed",
          "dining table",  "toilet",       "tv",
          "laptop",        "mouse",        "remote",
          "keyboard",      "cell phone",   "microwave",
          "oven",          "toaster",      "sink",
          "refrigerator",  "book",         "clock",
          "vase",          "scissors",     "teddy bear",
          "hair drier",    "toothbrush"};
}

std::vector<std::string> load_class_names(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open class names file " + path.string());
  std::vector<std::string> names;
  for (std::string line; std::getline(in, line);) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (!line.empty()) names.push_back(line);
  }
  return names;
}

OverlayRenderer::OverlayRenderer(OverlayOptions options) : options_(std::move(options)) {}

void OverlayRenderer::on_start(const RunInfo& info) {
  info_ = info;
  trails_.clear();
  last_seen_.clear();
  fps_ = 0.0;
  last_frame_time_ = {};
  last_refresh_ = {};
  hud_lines_.clear();
  stage_p50_ms_.assign(kHudStages.size(), 0.0);
}

std::string OverlayRenderer::label_for(int class_id) const {
  if (class_id >= 0 && static_cast<std::size_t>(class_id) < options_.class_names.size()) {
    return options_.class_names[static_cast<std::size_t>(class_id)];
  }
  return std::format("class {}", class_id);
}

void OverlayRenderer::update_telemetry(const Frame& frame) {
  const TimePoint t = now();
  if (last_frame_time_ != TimePoint{}) {
    const double dt = std::chrono::duration<double>(t - last_frame_time_).count();
    if (dt > 0.0) fps_ = fps_ == 0.0 ? 1.0 / dt : 0.9 * fps_ + 0.1 / dt;
  }
  last_frame_time_ = t;

  if (info_.metrics == nullptr ||
      (t - last_refresh_ < std::chrono::milliseconds(250) && !hud_lines_.empty())) {
    return;
  }
  last_refresh_ = t;
  const PipelineMetrics& m = *info_.metrics;
  for (std::size_t i = 0; i < kHudStages.size(); ++i) {
    stage_p50_ms_[i] = m.stage(kHudStages[i]).percentile_ms(0.5);
  }
  const std::uint64_t captured = m.frames_captured.load(std::memory_order_relaxed);
  const std::uint64_t dropped = m.frames_dropped.load(std::memory_order_relaxed);
  const double drop_pct =
      captured > 0 ? 100.0 * static_cast<double>(dropped) / static_cast<double>(captured) : 0.0;

  hud_lines_ = {
      std::format("takt-vision | {} | {}x{} | policy: {}", info_.backend, info_.model_width,
                  info_.model_height, info_.policy),
      std::format("FPS {:5.1f}   end-to-end p50 {:6.1f} ms   p99 {:6.1f} ms", fps_,
                  m.end_to_end().percentile_ms(0.5), m.end_to_end().percentile_ms(0.99)),
      std::format("stage p50 (ms): pre {:.1f}  infer {:.1f}  post {:.2f}  track {:.2f}",
                  stage_p50_ms_[0], stage_p50_ms_[1], stage_p50_ms_[2], stage_p50_ms_[3]),
      std::format("frame {}   dropped {} ({:.1f}%)", frame.sequence, dropped, drop_pct),
  };
}

void OverlayRenderer::draw_hud(cv::Mat& canvas, std::size_t visible_objects) const {
  const double scale = std::max(0.36, canvas.rows / 1500.0);
  const int line_height = static_cast<int>(std::lround(26 * scale / 0.55));
  const int panel_width = std::min(canvas.cols, static_cast<int>(std::lround(640 * scale / 0.55)));
  const int bar_height = line_height / 2;
  const int panel_height =
      line_height * (static_cast<int>(hud_lines_.size()) + 1) + bar_height + 16;

  // Darken the panel area in place instead of blending a second image.
  cv::Mat panel = canvas(cv::Rect(0, 0, panel_width, std::min(canvas.rows, panel_height)));
  panel.convertTo(panel, -1, 0.3, 0.0);

  int y = line_height;
  for (const std::string& line : hud_lines_) {
    cv::putText(canvas, line, {12, y}, cv::FONT_HERSHEY_SIMPLEX, scale, {240, 240, 240}, 1,
                cv::LINE_AA);
    y += line_height;
  }
  cv::putText(canvas, std::format("objects {}", visible_objects), {12, y}, cv::FONT_HERSHEY_SIMPLEX,
              scale, {240, 240, 240}, 1, cv::LINE_AA);

  // Stacked bar: where the time of an average frame goes.
  double total = 0.0;
  for (const double v : stage_p50_ms_) total += v;
  if (total <= 0.0) return;
  int x = 12;
  const int bar_width = panel_width - 24;
  const int bar_y = y + 8;
  for (std::size_t i = 0; i < stage_p50_ms_.size(); ++i) {
    const int w = static_cast<int>(std::lround(bar_width * stage_p50_ms_[i] / total));
    if (w > 0)
      cv::rectangle(canvas, cv::Rect(x, bar_y, w, bar_height), kStageColors[i], cv::FILLED);
    x += w;
  }
}

void OverlayRenderer::render(const Frame& frame, cv::Mat& canvas) {
  update_telemetry(frame);
  const double label_scale = std::max(0.4, canvas.rows / 1600.0);
  const int thickness = std::max(2, canvas.rows / 360);

  std::size_t visible = 0;
  if (!frame.tracks.empty() || !frame.detections.empty()) {
    if (!frame.tracks.empty()) {
      for (const TrackedObject& t : frame.tracks) {
        const cv::Scalar color = color_for(t.track_id);
        cv::rectangle(canvas, to_point(t.box.x1, t.box.y1), to_point(t.box.x2, t.box.y2), color,
                      thickness);
        draw_label(canvas, std::format("#{} {} {:.2f}", t.track_id, label_for(t.class_id), t.score),
                   to_point(t.box.x1, t.box.y1), color, label_scale);
        if (options_.draw_trails) {
          auto& trail = trails_[t.track_id];
          trail.push_back(to_point(t.box.center_x(), t.box.y2));
          while (trail.size() > options_.trail_length) trail.pop_front();
          last_seen_[t.track_id] = frame.sequence;
        }
      }
      visible = frame.tracks.size();
    } else {
      for (const Detection& d : frame.detections) {
        const cv::Scalar color = color_for(d.class_id + 1);
        cv::rectangle(canvas, to_point(d.box.x1, d.box.y1), to_point(d.box.x2, d.box.y2), color,
                      thickness);
        draw_label(canvas, std::format("{} {:.2f}", label_for(d.class_id), d.score),
                   to_point(d.box.x1, d.box.y1), color, label_scale);
      }
      visible = frame.detections.size();
    }
  }

  if (options_.draw_trails) {
    for (auto it = trails_.begin(); it != trails_.end();) {
      if (frame.sequence > last_seen_[it->first] + 30) {  // forget tracks gone for a second
        last_seen_.erase(it->first);
        it = trails_.erase(it);
        continue;
      }
      const auto& points = it->second;
      const cv::Scalar color = color_for(it->first);
      for (std::size_t i = 1; i < points.size(); ++i) {
        cv::line(canvas, points[i - 1], points[i], color, std::max(1, thickness - 1), cv::LINE_AA);
      }
      ++it;
    }
  }

  if (options_.draw_hud) draw_hud(canvas, visible);
}

}  // namespace takt

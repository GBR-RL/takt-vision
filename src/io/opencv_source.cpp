#include "takt/io/opencv_source.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <format>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <stdexcept>
#include <thread>

namespace takt {
namespace {

bool is_camera_index(const std::string& uri) {
  return !uri.empty() &&
         std::all_of(uri.begin(), uri.end(), [](unsigned char c) { return std::isdigit(c); });
}

bool is_still_image(const std::string& uri) {
  std::string ext = std::filesystem::path(uri).extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tif" ||
         ext == ".tiff" || ext == ".webp";
}

// Copies a cv::Mat (any of 8UC1 / 8UC3 / 8UC4, possibly non-contiguous) into a BGR Image,
// reusing the image's buffer.
void copy_to_image(const cv::Mat& mat, cv::Mat& scratch, Image& out) {
  const cv::Mat* bgr = &mat;
  if (mat.type() == CV_8UC1) {
    cv::cvtColor(mat, scratch, cv::COLOR_GRAY2BGR);
    bgr = &scratch;
  } else if (mat.type() == CV_8UC4) {
    cv::cvtColor(mat, scratch, cv::COLOR_BGRA2BGR);
    bgr = &scratch;
  } else if (mat.type() != CV_8UC3) {
    throw std::runtime_error("OpenCvSource: unsupported pixel type");
  }
  out.reshape(bgr->cols, bgr->rows, PixelFormat::kBgr8);
  const auto row_bytes = static_cast<std::size_t>(bgr->cols) * 3;
  for (int y = 0; y < bgr->rows; ++y) std::memcpy(out.row(y), bgr->ptr<std::uint8_t>(y), row_bytes);
}

}  // namespace

struct OpenCvSource::Impl {
  OpenCvSourceOptions options;
  cv::VideoCapture capture;
  cv::Mat still;    // decoded still image, if the URI is one
  cv::Mat frame;    // decode target, reused
  cv::Mat scratch;  // colour-conversion target, reused
  bool is_still = false;
  bool still_delivered = false;
  bool is_camera = false;
  double fps = 0.0;
  Nanos period{0};
  TimePoint next_deadline{};
  std::uint64_t frames_read = 0;
};

OpenCvSource::OpenCvSource(OpenCvSourceOptions options) : impl_(std::make_unique<Impl>()) {
  Impl& im = *impl_;
  im.options = std::move(options);
  const std::string& uri = im.options.uri;

  if (is_still_image(uri)) {
    im.still = cv::imread(uri, cv::IMREAD_COLOR);
    if (im.still.empty()) throw std::runtime_error("OpenCvSource: cannot read image " + uri);
    im.is_still = true;
    return;
  }

  if (is_camera_index(uri)) {
    im.is_camera = true;
    im.capture.open(std::stoi(uri));
    if (im.options.request_width > 0)
      im.capture.set(cv::CAP_PROP_FRAME_WIDTH, im.options.request_width);
    if (im.options.request_height > 0)
      im.capture.set(cv::CAP_PROP_FRAME_HEIGHT, im.options.request_height);
  } else if (uri.find('!') != std::string::npos) {
    im.capture.open(uri, cv::CAP_GSTREAMER);
  } else {
    im.capture.open(uri);
  }
  if (!im.capture.isOpened()) throw std::runtime_error("OpenCvSource: cannot open " + uri);

  im.fps = im.capture.get(cv::CAP_PROP_FPS);
  if (!(im.fps > 0.0 && im.fps < 1000.0)) im.fps = 30.0;
  // Cameras pace themselves; files are paced here if realtime is requested.
  if (!im.is_camera && im.options.realtime) {
    im.period = std::chrono::duration_cast<Nanos>(std::chrono::duration<double>(1.0 / im.fps));
  }
}

OpenCvSource::~OpenCvSource() = default;

bool OpenCvSource::read(Image& out) {
  Impl& im = *impl_;
  if (im.is_still) {
    if (im.still_delivered && !im.options.loop) return false;
    im.still_delivered = true;
    copy_to_image(im.still, im.scratch, out);
    return true;
  }

  if (im.period.count() > 0) {
    const TimePoint t = now();
    if (im.frames_read == 0 || t > im.next_deadline + im.period) im.next_deadline = t;
    std::this_thread::sleep_until(im.next_deadline);
    im.next_deadline += im.period;
  }

  if (!im.capture.read(im.frame) || im.frame.empty()) {
    if (!im.options.loop || im.is_camera) return false;
    im.capture.set(cv::CAP_PROP_POS_FRAMES, 0);
    if (!im.capture.read(im.frame) || im.frame.empty()) return false;
  }
  copy_to_image(im.frame, im.scratch, out);
  ++im.frames_read;
  return true;
}

std::string OpenCvSource::describe() const {
  const Impl& im = *impl_;
  if (im.is_still)
    return std::format("image {} ({}x{})", im.options.uri, im.still.cols, im.still.rows);
  return std::format("{} {} @{:.1f}fps{}", im.is_camera ? "camera" : "video", im.options.uri,
                     im.fps, im.period.count() > 0 ? " (paced)" : "");
}

double OpenCvSource::nominal_fps() const {
  return impl_->is_still ? 0.0 : impl_->fps;
}

}  // namespace takt

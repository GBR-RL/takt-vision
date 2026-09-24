#include "takt/io/overlay_sink.hpp"

#include <algorithm>
#include <cctype>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <stdexcept>

namespace takt {

struct OverlaySink::Impl {
  OverlaySinkOptions options;
  OverlayRenderer renderer;
  cv::VideoWriter writer;
  cv::Mat canvas;
  double fps = 30.0;
  bool quit_requested = false;
  bool window_open = false;

  explicit Impl(OverlaySinkOptions opts) : options(std::move(opts)), renderer(options.overlay) {}
};

OverlaySink::OverlaySink(OverlaySinkOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

OverlaySink::~OverlaySink() = default;

void OverlaySink::on_start(const RunInfo& info) {
  Impl& im = *impl_;
  im.renderer.on_start(info);
  im.fps = im.options.video_fps > 0.0 ? im.options.video_fps
                                      : (info.source_fps > 0.0 ? info.source_fps : 30.0);
  im.quit_requested = false;
}

void OverlaySink::consume(const Frame& frame) {
  Impl& im = *impl_;
  const Image& image = frame.image;
  // cv::Mat has no const-view constructor; the header only reads through this pointer.
  const cv::Mat source(image.height(), image.width(), CV_8UC3,
                       const_cast<std::uint8_t*>(image.data()),
                       static_cast<std::size_t>(image.stride()));
  source.copyTo(im.canvas);  // reallocates only if the resolution changes
  im.renderer.render(frame, im.canvas);

  if (!im.options.video_path.empty()) {
    if (!im.writer.isOpened()) {
      std::string ext = im.options.video_path.extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(),
                     [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      const int fourcc = ext == ".avi" ? cv::VideoWriter::fourcc('M', 'J', 'P', 'G')
                                       : cv::VideoWriter::fourcc('m', 'p', '4', 'v');
      im.writer.open(im.options.video_path.string(), fourcc, im.fps, im.canvas.size());
      if (!im.writer.isOpened()) {
        throw std::runtime_error("OverlaySink: cannot open video writer for " +
                                 im.options.video_path.string());
      }
    }
    im.writer.write(im.canvas);
  }

  if (!im.options.image_path.empty()) cv::imwrite(im.options.image_path.string(), im.canvas);

  if (im.options.show_window) {
    cv::imshow(im.options.window_title, im.canvas);
    im.window_open = true;
    const int key = cv::waitKey(1);
    if ((key == 'q' || key == 27) && !im.quit_requested) {
      im.quit_requested = true;
      if (im.options.on_quit) im.options.on_quit();
    }
  }
}

void OverlaySink::on_finish() {
  Impl& im = *impl_;
  if (im.writer.isOpened()) im.writer.release();
  if (im.window_open) {
    cv::destroyWindow(im.options.window_title);
    im.window_open = false;
  }
}

}  // namespace takt

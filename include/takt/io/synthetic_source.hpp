#pragma once

#include <cstdint>
#include <string>

#include "takt/core/clock.hpp"
#include "takt/pipeline/source_sink.hpp"

namespace takt {

struct SyntheticSourceOptions {
  int width = 1280;
  int height = 720;
  double fps = 30.0;              // 0 = deliver frames as fast as they are requested
  std::uint64_t frame_count = 0;  // 0 = endless
  int num_objects = 4;
};

// Camera simulator: renders coloured blocks moving over a dark background and paces delivery to a
// fixed frame rate like a free-running camera. Needs no files and no OpenCV, so the pipeline can be
// demonstrated, tested and benchmarked anywhere.
class SyntheticSource final : public FrameSource {
 public:
  explicit SyntheticSource(SyntheticSourceOptions options = {});

  bool read(Image& out) override;
  [[nodiscard]] std::string describe() const override;
  [[nodiscard]] double nominal_fps() const override { return options_.fps; }

 private:
  SyntheticSourceOptions options_;
  std::uint64_t produced_ = 0;
  TimePoint next_deadline_{};
};

}  // namespace takt

#pragma once

#include <filesystem>
#include <fstream>
#include <string>

#include "takt/pipeline/source_sink.hpp"

namespace takt {

// Writes one JSON object per frame: sequence, capture time, per-stage and end-to-end latency,
// detections and tracks. Machine-readable output for downstream systems (MES, PLC gateways,
// dashboards) and the input of scripts/plot_latency.py.
class JsonLinesSink final : public FrameSink {
 public:
  explicit JsonLinesSink(const std::filesystem::path& path);

  void on_start(const RunInfo& info) override;
  void consume(const Frame& frame) override;
  void on_finish() override;

 private:
  std::ofstream out_;
  std::string line_;  // reused buffer: formatting does not allocate once it has grown
  TimePoint started_at_{};
};

}  // namespace takt

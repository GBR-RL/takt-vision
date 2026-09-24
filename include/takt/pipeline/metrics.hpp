#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <string>

#include "takt/core/clock.hpp"
#include "takt/core/latency_histogram.hpp"
#include "takt/pipeline/frame.hpp"

namespace takt {

// Live counters and latency distributions, written by all pipeline threads and readable at any
// time (e.g. by the on-screen HUD) without stopping the pipeline.
class PipelineMetrics {
 public:
  void reset() noexcept;

  [[nodiscard]] LatencyHistogram& stage(StageId id) noexcept { return stages_[to_index(id)]; }
  [[nodiscard]] const LatencyHistogram& stage(StageId id) const noexcept {
    return stages_[to_index(id)];
  }
  [[nodiscard]] LatencyHistogram& end_to_end() noexcept { return end_to_end_; }
  [[nodiscard]] const LatencyHistogram& end_to_end() const noexcept { return end_to_end_; }

  std::atomic<std::uint64_t> frames_captured{0};
  std::atomic<std::uint64_t> frames_dropped{0};  // shed by a hand-off policy, at any boundary
  std::atomic<std::uint64_t> frames_completed{0};

 private:
  std::array<LatencyHistogram, kStageCount> stages_{};
  LatencyHistogram end_to_end_;
};

// {"count":..,"mean_ms":..,"p50_ms":.., ...}
[[nodiscard]] std::string to_json(const LatencySummary& summary);

// Summary of one run, suitable for printing or for the JSON consumed by scripts/plot_*.py.
struct RunReport {
  std::string mode;  // "pipelined" or "sequential"
  std::string backend;
  std::string source;
  std::string policy;
  int model_width = 0;
  int model_height = 0;
  std::uint64_t frames_captured = 0;
  std::uint64_t frames_dropped = 0;
  std::uint64_t frames_completed = 0;
  double wall_seconds = 0.0;
  double throughput_fps = 0.0;
  std::array<LatencySummary, kStageCount> stages{};
  LatencySummary end_to_end{};

  [[nodiscard]] std::string to_text() const;
  [[nodiscard]] std::string to_json() const;
};

}  // namespace takt

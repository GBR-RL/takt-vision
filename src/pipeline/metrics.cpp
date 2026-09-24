#include "takt/pipeline/metrics.hpp"

#include <format>
#include <iterator>
#include <string_view>

namespace takt {
namespace {

void append_json_string(std::string& out, std::string_view text) {
  out += '"';
  for (const char c : text) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          std::format_to(std::back_inserter(out), "\\u{:04x}", static_cast<unsigned>(c));
        } else {
          out += c;
        }
    }
  }
  out += '"';
}

void append_summary(std::string& out, const LatencySummary& s) {
  std::format_to(std::back_inserter(out),
                 R"({{"count":{},"mean_ms":{:.4f},"min_ms":{:.4f},"p50_ms":{:.4f},"p90_ms":{:.4f},)"
                 R"("p95_ms":{:.4f},"p99_ms":{:.4f},"p999_ms":{:.4f},"max_ms":{:.4f}}})",
                 s.count, s.mean_ms, s.min_ms, s.p50_ms, s.p90_ms, s.p95_ms, s.p99_ms, s.p999_ms,
                 s.max_ms);
}

}  // namespace

std::string to_json(const LatencySummary& summary) {
  std::string out;
  append_summary(out, summary);
  return out;
}

void PipelineMetrics::reset() noexcept {
  for (auto& h : stages_) h.reset();
  end_to_end_.reset();
  frames_captured.store(0, std::memory_order_relaxed);
  frames_dropped.store(0, std::memory_order_relaxed);
  frames_completed.store(0, std::memory_order_relaxed);
}

std::string RunReport::to_text() const {
  std::string out;
  auto it = std::back_inserter(out);
  std::format_to(it, "takt-vision run ({}, {}, policy={})\n", mode, backend, policy);
  std::format_to(it, "  source     : {}\n", source);
  std::format_to(it, "  model input: {}x{}\n", model_width, model_height);
  std::format_to(it, "  frames     : {} captured, {} completed, {} dropped\n", frames_captured,
                 frames_completed, frames_dropped);
  std::format_to(it, "  throughput : {:.1f} fps over {:.2f} s\n", throughput_fps, wall_seconds);
  std::format_to(it, "  {:<12} {:>9} {:>9} {:>9} {:>9} {:>9}\n", "latency(ms)", "mean", "p50",
                 "p95", "p99", "max");
  const auto row = [&](std::string_view name, const LatencySummary& s) {
    if (s.count == 0) return;
    std::format_to(it, "  {:<12} {:>9.2f} {:>9.2f} {:>9.2f} {:>9.2f} {:>9.2f}\n", name, s.mean_ms,
                   s.p50_ms, s.p95_ms, s.p99_ms, s.max_ms);
  };
  for (std::size_t i = 0; i < kStageCount; ++i) row(stage_name(static_cast<StageId>(i)), stages[i]);
  row("end-to-end", end_to_end);
  return out;
}

std::string RunReport::to_json() const {
  std::string out = "{";
  auto it = std::back_inserter(out);
  out += R"("mode":)";
  append_json_string(out, mode);
  out += R"(,"backend":)";
  append_json_string(out, backend);
  out += R"(,"source":)";
  append_json_string(out, source);
  out += R"(,"policy":)";
  append_json_string(out, policy);
  std::format_to(
      it,
      R"(,"model_width":{},"model_height":{},"frames_captured":{},"frames_dropped":{},)"
      R"("frames_completed":{},"wall_seconds":{:.4f},"throughput_fps":{:.3f},"stages":{{)",
      model_width, model_height, frames_captured, frames_dropped, frames_completed, wall_seconds,
      throughput_fps);
  for (std::size_t i = 0; i < kStageCount; ++i) {
    if (i > 0) out += ',';
    append_json_string(out, stage_name(static_cast<StageId>(i)));
    out += ':';
    append_summary(out, stages[i]);
  }
  out += R"(},"end_to_end":)";
  append_summary(out, end_to_end);
  out += "}";
  return out;
}

}  // namespace takt

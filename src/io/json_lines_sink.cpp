#include "takt/io/json_lines_sink.hpp"

#include <format>
#include <iterator>
#include <stdexcept>

namespace takt {

JsonLinesSink::JsonLinesSink(const std::filesystem::path& path) : out_(path) {
  if (!out_) throw std::runtime_error("JsonLinesSink: cannot open " + path.string());
  line_.reserve(4096);
}

void JsonLinesSink::on_start(const RunInfo& info) {
  started_at_ = info.started_at;
}

void JsonLinesSink::consume(const Frame& frame) {
  line_.clear();
  auto it = std::back_inserter(line_);
  const auto stage_ms = [&](StageId id) { return to_ms(frame.service_time[to_index(id)]); };
  std::format_to(
      it,
      R"({{"seq":{},"t_ms":{:.3f},"e2e_ms":{:.3f},"stage_ms":{{"pre":{:.3f},"infer":{:.3f},)"
      R"("post":{:.3f},"track":{:.3f}}},"detections":[)",
      frame.sequence, to_ms(frame.captured_at - started_at_), to_ms(now() - frame.captured_at),
      stage_ms(StageId::kPreprocess), stage_ms(StageId::kInference),
      stage_ms(StageId::kPostprocess), stage_ms(StageId::kTracking));
  for (std::size_t i = 0; i < frame.detections.size(); ++i) {
    const Detection& d = frame.detections[i];
    std::format_to(it, R"({}{{"cls":{},"score":{:.4f},"box":[{:.1f},{:.1f},{:.1f},{:.1f}]}})",
                   i == 0 ? "" : ",", d.class_id, d.score, d.box.x1, d.box.y1, d.box.x2, d.box.y2);
  }
  line_ += R"(],"tracks":[)";
  for (std::size_t i = 0; i < frame.tracks.size(); ++i) {
    const TrackedObject& t = frame.tracks[i];
    std::format_to(
        it, R"({}{{"id":{},"cls":{},"score":{:.4f},"age":{},"box":[{:.1f},{:.1f},{:.1f},{:.1f}]}})",
        i == 0 ? "" : ",", t.track_id, t.class_id, t.score, t.age_frames, t.box.x1, t.box.y1,
        t.box.x2, t.box.y2);
  }
  line_ += "]}\n";
  out_.write(line_.data(), static_cast<std::streamsize>(line_.size()));
}

void JsonLinesSink::on_finish() {
  out_.flush();
}

}  // namespace takt

#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "takt/track/kalman_filter.hpp"
#include "takt/track/linear_assignment.hpp"
#include "takt/vision/detection.hpp"

namespace takt {

struct TrackedObject {
  int track_id = 0;
  Box box;  // Kalman-filtered box in source-image pixels
  float score = 0.0f;
  int class_id = 0;
  int age_frames = 0;
};

// Defaults follow Ultralytics' bytetrack.yaml.
struct ByteTrackerOptions {
  float high_threshold = 0.25f;       // detections above this take part in the first association
  float low_threshold = 0.1f;         // (low, high) detections only rescue existing tracks
  float new_track_threshold = 0.25f;  // minimum score to start a new track
  float match_threshold = 0.8f;       // max (1 - IoU) cost in the first association
  int track_buffer = 30;              // frames a lost track is kept for re-identification
  float frame_rate = 30.0f;
  bool fuse_score = true;  // weight IoU by detection confidence
};

// ByteTrack (Zhang et al., ECCV 2022): associate every detection box, not just confident ones.
//   1. Kalman-predict all live tracks.
//   2. Match high-score detections to tracked + lost tracks by IoU (Hungarian, gated).
//   3. Match low-score detections to the still-unmatched tracked tracks - this is what keeps IDs
//      alive through occlusion and motion blur, where a detector's confidence dips.
//   4. Confirm or discard tentative tracks, start new ones, and age out lost ones.
//
// Detections of any class are associated together, like the reference implementation; the
// reported class is that of the most recent matching detection.
class ByteTracker {
 public:
  explicit ByteTracker(ByteTrackerOptions options = {});

  // Detections must be in source-image pixels and should include low-score boxes (run the
  // decoder with score threshold <= low_threshold). Writes confirmed tracks to `out`.
  void update(std::span<const Detection> detections, std::vector<TrackedObject>& out);
  void reset();

  [[nodiscard]] int frame_index() const noexcept { return frame_id_; }
  [[nodiscard]] const ByteTrackerOptions& options() const noexcept { return options_; }

 private:
  enum class TrackState : std::uint8_t { kTracked, kLost, kRemoved };

  struct Track {
    int id = 0;
    BoxKalmanFilter::State kalman;
    float score = 0.0f;
    int class_id = 0;
    TrackState state = TrackState::kTracked;
    bool activated = false;
    int start_frame = 0;
    int last_frame = 0;

    [[nodiscard]] Box box() const noexcept { return BoxKalmanFilter::to_box(kalman.mean); }
  };

  void associate(std::span<const int> track_indices, std::span<const int> detection_indices,
                 std::span<const Detection> detections, float cost_limit, bool fuse_score);
  void apply_match(Track& track, const Detection& detection);

  ByteTrackerOptions options_;
  std::vector<Track> tracks_;
  int frame_id_ = 0;
  int next_id_ = 1;

  // Per-frame scratch, kept to avoid reallocating every frame.
  std::vector<int> high_;
  std::vector<int> low_;
  std::vector<int> pool_;
  std::vector<int> unconfirmed_;
  std::vector<int> remaining_tracks_;
  std::vector<int> remaining_high_;
  std::vector<int> new_candidates_;
  std::vector<float> cost_;
  LinearAssignment solver_;
  AssignmentResult assignment_;
};

}  // namespace takt

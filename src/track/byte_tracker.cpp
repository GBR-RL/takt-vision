#include "takt/track/byte_tracker.hpp"

#include <cstddef>

namespace takt {
namespace {

constexpr float kSecondStageCostLimit = 0.5f;  // low-score detections vs remaining tracks
constexpr float kUnconfirmedCostLimit = 0.7f;  // tentative tracks vs leftover detections

std::size_t idx(int i) noexcept {
  return static_cast<std::size_t>(i);
}

}  // namespace

ByteTracker::ByteTracker(ByteTrackerOptions options) : options_(options) {}

void ByteTracker::reset() {
  tracks_.clear();
  frame_id_ = 0;
  next_id_ = 1;
}

void ByteTracker::associate(std::span<const int> track_indices,
                            std::span<const int> detection_indices,
                            std::span<const Detection> detections, float cost_limit,
                            bool fuse_score) {
  const std::size_t rows = track_indices.size();
  const std::size_t cols = detection_indices.size();
  cost_.resize(rows * cols);
  for (std::size_t r = 0; r < rows; ++r) {
    const Box track_box = tracks_[idx(track_indices[r])].box();
    for (std::size_t c = 0; c < cols; ++c) {
      const Detection& d = detections[idx(detection_indices[c])];
      float similarity = iou(track_box, d.box);
      if (fuse_score) similarity *= d.score;
      cost_[r * cols + c] = 1.0f - similarity;
    }
  }
  solver_.solve(cost_, static_cast<int>(rows), static_cast<int>(cols), cost_limit, assignment_);
}

void ByteTracker::apply_match(Track& track, const Detection& detection) {
  BoxKalmanFilter::update(track.kalman, BoxKalmanFilter::to_measurement(detection.box));
  track.score = detection.score;
  track.class_id = detection.class_id;
  track.state = TrackState::kTracked;
  track.activated = true;
  track.last_frame = frame_id_;
}

void ByteTracker::update(std::span<const Detection> detections, std::vector<TrackedObject>& out) {
  ++frame_id_;

  high_.clear();
  low_.clear();
  for (std::size_t i = 0; i < detections.size(); ++i) {
    const float s = detections[i].score;
    if (s >= options_.high_threshold) {
      high_.push_back(static_cast<int>(i));
    } else if (s > options_.low_threshold) {
      low_.push_back(static_cast<int>(i));
    }
  }

  // Split live tracks: confirmed or lost ones form the matching pool; tentative ones (seen once,
  // not yet confirmed) only get a chance against leftovers.
  pool_.clear();
  unconfirmed_.clear();
  for (std::size_t t = 0; t < tracks_.size(); ++t) {
    Track& track = tracks_[t];
    if (track.state == TrackState::kTracked && !track.activated) {
      unconfirmed_.push_back(static_cast<int>(t));
    } else if (track.state != TrackState::kRemoved) {
      pool_.push_back(static_cast<int>(t));
      // A lost track stops growing: freeze its height velocity before predicting.
      if (track.state != TrackState::kTracked) track.kalman.mean(7, 0) = 0.0f;
      BoxKalmanFilter::predict(track.kalman);
    }
  }

  // --- 1st association: high-score detections vs tracked + lost tracks ----------------------
  associate(pool_, high_, detections, options_.match_threshold, options_.fuse_score);
  for (const auto& [r, c] : assignment_.matches) {
    apply_match(tracks_[idx(pool_[idx(r)])], detections[idx(high_[idx(c)])]);
  }
  remaining_tracks_.clear();
  for (const int r : assignment_.unmatched_rows) {
    const int t = pool_[idx(r)];
    if (tracks_[idx(t)].state == TrackState::kTracked) remaining_tracks_.push_back(t);
  }
  remaining_high_.clear();
  for (const int c : assignment_.unmatched_cols) remaining_high_.push_back(high_[idx(c)]);

  // --- 2nd association: low-score detections vs still-unmatched tracked tracks --------------
  associate(remaining_tracks_, low_, detections, kSecondStageCostLimit, /*fuse_score=*/false);
  for (const auto& [r, c] : assignment_.matches) {
    apply_match(tracks_[idx(remaining_tracks_[idx(r)])], detections[idx(low_[idx(c)])]);
  }
  for (const int r : assignment_.unmatched_rows) {
    tracks_[idx(remaining_tracks_[idx(r)])].state = TrackState::kLost;
  }

  // --- 3rd association: tentative tracks vs leftover high-score detections ------------------
  associate(unconfirmed_, remaining_high_, detections, kUnconfirmedCostLimit, options_.fuse_score);
  for (const auto& [r, c] : assignment_.matches) {
    apply_match(tracks_[idx(unconfirmed_[idx(r)])], detections[idx(remaining_high_[idx(c)])]);
  }
  for (const int r : assignment_.unmatched_rows) {
    tracks_[idx(unconfirmed_[idx(r)])].state = TrackState::kRemoved;  // one-frame flicker
  }
  new_candidates_.clear();
  for (const int c : assignment_.unmatched_cols) new_candidates_.push_back(remaining_high_[idx(c)]);

  // --- Start new tracks (appending may reallocate tracks_, so this comes last) --------------
  for (const int d : new_candidates_) {
    const Detection& det = detections[idx(d)];
    if (det.score < options_.new_track_threshold) continue;
    Track track;
    track.id = next_id_++;
    track.kalman = BoxKalmanFilter::initiate(BoxKalmanFilter::to_measurement(det.box));
    track.score = det.score;
    track.class_id = det.class_id;
    track.state = TrackState::kTracked;
    track.activated = frame_id_ == 1;  // on the first frame there is nothing to confirm against
    track.start_frame = frame_id_;
    track.last_frame = frame_id_;
    tracks_.push_back(track);
  }

  // --- Age out lost tracks and drop removed ones ------------------------------------------
  const int max_time_lost =
      static_cast<int>(options_.frame_rate / 30.0f * static_cast<float>(options_.track_buffer));
  for (Track& track : tracks_) {
    if (track.state == TrackState::kLost && frame_id_ - track.last_frame > max_time_lost) {
      track.state = TrackState::kRemoved;
    }
  }
  std::erase_if(tracks_, [](const Track& t) { return t.state == TrackState::kRemoved; });

  out.clear();
  for (const Track& track : tracks_) {
    if (track.state == TrackState::kTracked && track.activated) {
      out.push_back(
          {track.id, track.box(), track.score, track.class_id, frame_id_ - track.start_frame + 1});
    }
  }
}

}  // namespace takt

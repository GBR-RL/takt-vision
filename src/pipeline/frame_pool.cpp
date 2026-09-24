#include "takt/pipeline/frame_pool.hpp"

#include <stdexcept>

namespace takt {

void FrameRecycler::operator()(Frame* frame) const noexcept {
  if (pool != nullptr && frame != nullptr) pool->recycle(frame);
}

FramePool::FramePool(std::size_t frame_count, const FrameBufferSizes& sizes) {
  if (frame_count == 0) throw std::invalid_argument("FramePool: frame_count must be positive");
  storage_.reserve(frame_count);
  free_.reserve(frame_count);  // recycle() must never allocate
  for (std::size_t i = 0; i < frame_count; ++i) {
    auto frame = std::make_unique<Frame>();
    frame->input.resize(sizes.input_elements);
    frame->output.resize(sizes.output_elements);
    frame->detections.reserve(sizes.max_detections);
    frame->tracks.reserve(sizes.max_detections);
    free_.push_back(frame.get());
    storage_.push_back(std::move(frame));
  }
}

FramePtr FramePool::take_locked() {
  Frame* frame = free_.back();
  free_.pop_back();
  return FramePtr(frame, FrameRecycler{this});
}

FramePtr FramePool::try_acquire() {
  std::scoped_lock lock(mutex_);
  if (free_.empty()) return FramePtr(nullptr, FrameRecycler{this});
  return take_locked();
}

FramePtr FramePool::acquire(std::stop_token stop) {
  std::unique_lock lock(mutex_);
  // C++20: condition_variable_any::wait with a stop_token returns early when stop is requested,
  // without the classic "set a flag, then notify" shutdown dance.
  if (!frame_available_.wait(lock, stop, [this] { return !free_.empty(); })) {
    return FramePtr(nullptr, FrameRecycler{this});
  }
  return take_locked();
}

std::size_t FramePool::available() const {
  std::scoped_lock lock(mutex_);
  return free_.size();
}

void FramePool::recycle(Frame* frame) noexcept {
  frame->reset();
  {
    std::scoped_lock lock(mutex_);
    free_.push_back(frame);  // capacity reserved up front: cannot throw
  }
  frame_available_.notify_one();
}

}  // namespace takt

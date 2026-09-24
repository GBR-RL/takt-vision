#pragma once

#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stop_token>
#include <vector>

#include "takt/pipeline/frame.hpp"

namespace takt {

class FramePool;

// Deleter that hands a frame back to its pool instead of freeing it.
struct FrameRecycler {
  FramePool* pool = nullptr;
  void operator()(Frame* frame) const noexcept;
};

// Unique ownership of a pooled frame. Dropping it anywhere - in a full queue, after an exception,
// at shutdown - returns the frame to the pool automatically. That is RAII doing the bookkeeping a
// hand-written "release" call would get wrong on some error path.
using FramePtr = std::unique_ptr<Frame, FrameRecycler>;

struct FrameBufferSizes {
  std::size_t input_elements = 0;
  std::size_t output_elements = 0;
  std::size_t max_detections = 300;
};

// Fixed set of preallocated frames. Acquire happens on the capture thread; frames are released
// from whichever thread drops them. A mutex guards the free list: it is touched twice per frame,
// orders of magnitude less often than the ring buffers, so lock-free machinery would buy nothing.
//
// The pool must outlive every FramePtr it hands out.
class FramePool {
 public:
  FramePool(std::size_t frame_count, const FrameBufferSizes& sizes);
  FramePool(const FramePool&) = delete;
  FramePool& operator=(const FramePool&) = delete;
  FramePool(FramePool&&) = delete;
  FramePool& operator=(FramePool&&) = delete;
  ~FramePool() = default;

  // Returns an empty pointer if every frame is in use.
  [[nodiscard]] FramePtr try_acquire();
  // Blocks until a frame is free; returns an empty pointer if `stop` is requested first.
  [[nodiscard]] FramePtr acquire(std::stop_token stop);

  [[nodiscard]] std::size_t capacity() const noexcept { return storage_.size(); }
  [[nodiscard]] std::size_t available() const;

  // Gives every free frame an image buffer of at least `bytes`. Called once the first camera
  // frame reveals the resolution: the pool is LIFO, so frames deep in it may stay unused for a
  // long time, and without this their first use would allocate - a latency spike in the
  // middle of a run (caught by tests/test_zero_alloc.cpp).
  void reserve_image_bytes(std::size_t bytes);

 private:
  friend struct FrameRecycler;
  void recycle(Frame* frame) noexcept;
  FramePtr take_locked();

  std::vector<std::unique_ptr<Frame>> storage_;
  mutable std::mutex mutex_;
  std::condition_variable_any frame_available_;
  std::vector<Frame*> free_;  // LIFO: the most recently used frame is the one hot in cache
};

}  // namespace takt

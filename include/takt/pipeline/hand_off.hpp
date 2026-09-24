#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <variant>

#include "takt/core/latest_slot.hpp"
#include "takt/core/spsc_ring.hpp"
#include "takt/pipeline/frame_pool.hpp"

namespace takt {

// What happens when the camera produces frames faster than the pipeline consumes them.
enum class IngressPolicy : std::uint8_t {
  kLatest,      // keep only the newest frame: lowest latency, drops stale frames (default)
  kDropNewest,  // bounded FIFO, reject new frames while full: every kept frame is processed
  kBlock,       // bounded FIFO, stall the camera while full: nothing is dropped by the pipeline
};

[[nodiscard]] std::string_view to_string(IngressPolicy policy) noexcept;
[[nodiscard]] std::optional<IngressPolicy> parse_ingress_policy(std::string_view text) noexcept;

enum class HandOffResult : std::uint8_t {
  kAccepted,         // queued
  kReplacedStale,    // queued; an older frame nobody picked up was dropped (kLatest)
  kDroppedIncoming,  // queue full, this frame was dropped (kDropNewest)
  kClosed,           // pipeline is shutting down; the frame was released
};

[[nodiscard]] constexpr bool is_drop(HandOffResult result) noexcept {
  return result == HandOffResult::kReplacedStale || result == HandOffResult::kDroppedIncoming;
}

// Hand-off of frames between two pipeline threads, with a policy for what happens when the
// consumer is slower than the producer:
//   kLatest      a lock-free "latest value" mailbox: never blocks, replaces stale frames
//   kDropNewest  a bounded lock-free FIFO that rejects frames while full
//   kBlock       a bounded lock-free FIFO that makes the producer wait
//
// With kLatest every stage boundary sheds load, so frames are dropped right in front of whichever
// stage is the bottleneck instead of going stale in the hands of the stage before it. With the
// FIFO policies only the camera-side hand-off drops (or blocks), internal hand-offs block, and
// every admitted frame is processed.
class HandOff {
 public:
  HandOff(IngressPolicy policy, std::size_t capacity);

  // Producer thread only.
  HandOffResult push(FramePtr frame);
  // Consumer thread only. Blocks; returns nullopt once closed and drained.
  [[nodiscard]] std::optional<FramePtr> pop();
  void close() noexcept;

  [[nodiscard]] IngressPolicy policy() const noexcept { return policy_; }

 private:
  using Storage = std::variant<SpscRing<FramePtr>, LatestSlot<FramePtr>>;
  static Storage make_storage(IngressPolicy policy, std::size_t capacity);

  IngressPolicy policy_;
  Storage storage_;
};

}  // namespace takt

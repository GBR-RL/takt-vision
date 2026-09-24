#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <optional>
#include <utility>

#include "takt/core/spsc_ring.hpp"  // QueueElement, kCacheLineSize

namespace takt {

// Single-producer / single-consumer "latest value" mailbox, implemented as a lock-free triple
// buffer.
//
// This is the right hand-off between a camera and a detector that is sometimes slower than the
// camera: the producer never blocks and never queues, it replaces whatever the consumer has not
// picked up yet. The consumer therefore always gets the freshest frame and end-to-end latency
// stays bounded no matter how far behind inference falls. Displaced values are handed back to the
// producer so it can recycle them and count them as drops.
//
// Three slots rotate between producer (back), consumer (front) and the shared middle. The middle
// index plus a "fresh" bit live in one atomic byte, so publishing and taking are each a single
// atomic exchange - wait-free for the producer.
template <QueueElement T>
class LatestSlot {
 public:
  LatestSlot() = default;
  LatestSlot(const LatestSlot&) = delete;
  LatestSlot& operator=(const LatestSlot&) = delete;
  LatestSlot(LatestSlot&&) = delete;
  LatestSlot& operator=(LatestSlot&&) = delete;
  ~LatestSlot() = default;

  // Producer only. Publishes `value` and returns the previously published value if the consumer
  // never took it. After close() nothing is published and `value` itself is handed back.
  [[nodiscard]] std::optional<T> publish(T value) {
    if (closed_.load(std::memory_order_acquire)) return std::optional<T>{std::move(value)};

    slots_[back_] = std::move(value);
    const std::uint8_t previous =
        middle_.exchange(static_cast<std::uint8_t>(back_ | kFresh), std::memory_order_acq_rel);
    back_ = static_cast<std::uint8_t>(previous & kIndexMask);

    // Invariant: a middle slot without the fresh bit is empty (the consumer moved out of it), so
    // the producer's new back slot is either empty or holds a value nobody consumed.
    std::optional<T> displaced;
    if ((previous & kFresh) != 0) displaced = std::exchange(slots_[back_], std::nullopt);

    published_.fetch_add(1, std::memory_order_release);
    published_.notify_one();
    return displaced;
  }

  // Consumer only.
  [[nodiscard]] std::optional<T> try_take() {
    if ((middle_.load(std::memory_order_relaxed) & kFresh) == 0) return std::nullopt;
    // Only the consumer clears the fresh bit, so it is still set here even if the producer has
    // published again in the meantime.
    const std::uint8_t previous = middle_.exchange(front_, std::memory_order_acq_rel);
    front_ = static_cast<std::uint8_t>(previous & kIndexMask);
    return std::exchange(slots_[front_], std::nullopt);
  }

  // Consumer only. Blocks until a value is published; returns nullopt once closed and empty.
  [[nodiscard]] std::optional<T> take() {
    for (;;) {
      const std::uint32_t seen = published_.load(std::memory_order_acquire);
      if (auto value = try_take()) return value;
      if (closed_.load(std::memory_order_acquire)) return try_take();
      published_.wait(seen, std::memory_order_acquire);
    }
  }

  void close() noexcept {
    closed_.store(true, std::memory_order_release);
    published_.fetch_add(1, std::memory_order_release);
    published_.notify_all();
  }

  [[nodiscard]] bool closed() const noexcept { return closed_.load(std::memory_order_acquire); }

 private:
  static constexpr std::uint8_t kFresh = 0x4;
  static constexpr std::uint8_t kIndexMask = 0x3;

  std::array<std::optional<T>, 3> slots_{};
  alignas(kCacheLineSize) std::atomic<std::uint8_t> middle_{1};
  alignas(kCacheLineSize) std::uint8_t back_ = 0;   // producer-private
  alignas(kCacheLineSize) std::uint8_t front_ = 2;  // consumer-private
  alignas(kCacheLineSize) std::atomic<std::uint32_t> published_{0};
  std::atomic<bool> closed_{false};
};

}  // namespace takt

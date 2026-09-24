#pragma once

#include <atomic>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

namespace takt {

// 64 bytes on every target we care about (x86-64 and 64-bit ARM). Hard-coded rather than
// std::hardware_destructive_interference_size, whose value GCC warns may differ between TUs.
inline constexpr std::size_t kCacheLineSize = 64;

template <class T>
concept QueueElement = std::movable<T> && std::default_initializable<T>;

// Bounded single-producer / single-consumer ring buffer.
//
// The fast path is lock-free: one relaxed load of the thread's own index, one acquire load of the
// other side's index (skipped entirely while the cached copy says there is room), and one release
// store. The producer's and consumer's indices live on separate cache lines so the two threads do
// not false-share.
//
// Blocking push()/pop() park the thread with C++20 atomic wait/notify on an epoch counter instead
// of spinning, so an idle pipeline stage costs no CPU - important on machines with few cores, where
// every spinning thread steals a core from inference.
template <QueueElement T>
class SpscRing {
 public:
  explicit SpscRing(std::size_t min_capacity)
      : capacity_(std::bit_ceil(min_capacity == 0 ? std::size_t{1} : min_capacity)),
        mask_(capacity_ - 1),
        slots_(std::make_unique<T[]>(capacity_)) {}

  SpscRing(const SpscRing&) = delete;
  SpscRing& operator=(const SpscRing&) = delete;
  SpscRing(SpscRing&&) = delete;
  SpscRing& operator=(SpscRing&&) = delete;
  ~SpscRing() = default;

  // Producer only. Moves from `value` only when it returns true, so on failure the caller still
  // owns the element and decides what to do with it (e.g. count it as a dropped frame).
  [[nodiscard]] bool try_push(T& value) {
    const std::size_t tail = tail_.load(std::memory_order_relaxed);
    if (tail - cached_head_ == capacity_) {
      cached_head_ = head_.load(std::memory_order_acquire);
      if (tail - cached_head_ == capacity_) return false;
    }
    slots_[tail & mask_] = std::move(value);
    tail_.store(tail + 1, std::memory_order_release);
    signal(pushed_);
    return true;
  }

  // Producer only. Blocks while the ring is full. Returns false, leaving `value` untouched, once
  // the ring has been closed.
  [[nodiscard]] bool push(T& value) {
    for (;;) {
      const std::uint32_t seen = popped_.load(std::memory_order_acquire);
      if (closed_.load(std::memory_order_acquire)) return false;
      if (try_push(value)) return true;
      popped_.wait(seen, std::memory_order_acquire);
    }
  }

  // Consumer only.
  [[nodiscard]] std::optional<T> try_pop() {
    const std::size_t head = head_.load(std::memory_order_relaxed);
    if (head == cached_tail_) {
      cached_tail_ = tail_.load(std::memory_order_acquire);
      if (head == cached_tail_) return std::nullopt;
    }
    std::optional<T> value{std::move(slots_[head & mask_])};
    head_.store(head + 1, std::memory_order_release);
    signal(popped_);
    return value;
  }

  // Consumer only. Blocks while the ring is empty. Returns nullopt only once the ring is closed
  // *and* drained, so no element pushed before close() is ever lost.
  [[nodiscard]] std::optional<T> pop() {
    for (;;) {
      const std::uint32_t seen = pushed_.load(std::memory_order_acquire);
      if (auto value = try_pop()) return value;
      if (closed_.load(std::memory_order_acquire)) return try_pop();
      pushed_.wait(seen, std::memory_order_acquire);
    }
  }

  // Any thread. Wakes blocked producers and consumers; subsequent push() calls fail.
  void close() noexcept {
    closed_.store(true, std::memory_order_release);
    pushed_.fetch_add(1, std::memory_order_release);
    pushed_.notify_all();
    popped_.fetch_add(1, std::memory_order_release);
    popped_.notify_all();
  }

  [[nodiscard]] bool closed() const noexcept { return closed_.load(std::memory_order_acquire); }
  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::size_t size_approx() const noexcept {
    return tail_.load(std::memory_order_relaxed) - head_.load(std::memory_order_relaxed);
  }

 private:
  static void signal(std::atomic<std::uint32_t>& epoch) noexcept {
    epoch.fetch_add(1, std::memory_order_release);
    epoch.notify_one();
  }

  const std::size_t capacity_;
  const std::size_t mask_;
  std::unique_ptr<T[]> slots_;

  alignas(kCacheLineSize) std::atomic<std::size_t> head_{0};  // written by the consumer
  std::size_t cached_tail_ = 0;                               // consumer-private
  alignas(kCacheLineSize) std::atomic<std::size_t> tail_{0};  // written by the producer
  std::size_t cached_head_ = 0;                               // producer-private

  // Wake-up epochs for blocking callers. Bumped on every push/pop and on close, so a waiter that
  // loaded the epoch before re-checking the ring can never miss a wake-up.
  alignas(kCacheLineSize) std::atomic<std::uint32_t> pushed_{0};
  alignas(kCacheLineSize) std::atomic<std::uint32_t> popped_{0};
  std::atomic<bool> closed_{false};
};

}  // namespace takt

#include "takt/pipeline/hand_off.hpp"

#include <utility>

namespace takt {
namespace {

template <class... Fs>
struct Overloaded : Fs... {
  using Fs::operator()...;
};
template <class... Fs>
Overloaded(Fs...) -> Overloaded<Fs...>;

}  // namespace

std::string_view to_string(IngressPolicy policy) noexcept {
  switch (policy) {
    case IngressPolicy::kLatest: return "latest";
    case IngressPolicy::kDropNewest: return "drop-newest";
    case IngressPolicy::kBlock: return "block";
  }
  return "unknown";
}

std::optional<IngressPolicy> parse_ingress_policy(std::string_view text) noexcept {
  if (text == "latest") return IngressPolicy::kLatest;
  if (text == "drop-newest") return IngressPolicy::kDropNewest;
  if (text == "block") return IngressPolicy::kBlock;
  return std::nullopt;
}

HandOff::Storage HandOff::make_storage(IngressPolicy policy, std::size_t capacity) {
  // Neither alternative is movable (they own atomics), so each branch returns a prvalue that is
  // constructed in place - guaranteed copy elision, no move constructor required.
  if (policy == IngressPolicy::kLatest) return Storage{std::in_place_type<LatestSlot<FramePtr>>};
  return Storage{std::in_place_type<SpscRing<FramePtr>>, capacity};
}

HandOff::HandOff(IngressPolicy policy, std::size_t capacity)
    : policy_(policy), storage_(make_storage(policy, capacity)) {}

HandOffResult HandOff::push(FramePtr frame) {
  return std::visit(
      Overloaded{
          [&](LatestSlot<FramePtr>& slot) {
            if (slot.closed()) return HandOffResult::kClosed;
            // The displaced frame (if any) goes out of scope here and returns to the pool.
            const std::optional<FramePtr> displaced = slot.publish(std::move(frame));
            return displaced ? HandOffResult::kReplacedStale : HandOffResult::kAccepted;
          },
          [&](SpscRing<FramePtr>& ring) {
            if (policy_ == IngressPolicy::kBlock) {
              return ring.push(frame) ? HandOffResult::kAccepted : HandOffResult::kClosed;
            }
            if (ring.closed()) return HandOffResult::kClosed;
            return ring.try_push(frame) ? HandOffResult::kAccepted
                                        : HandOffResult::kDroppedIncoming;
          },
      },
      storage_);
}

std::optional<FramePtr> HandOff::pop() {
  return std::visit(Overloaded{
                        [](LatestSlot<FramePtr>& slot) { return slot.take(); },
                        [](SpscRing<FramePtr>& ring) { return ring.pop(); },
                    },
                    storage_);
}

void HandOff::close() noexcept {
  std::visit([](auto& queue) { queue.close(); }, storage_);
}

}  // namespace takt

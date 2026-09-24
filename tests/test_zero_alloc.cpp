// Verifies the core design claim: once warmed up, the pipeline performs no heap allocation per
// frame on any thread. Global operator new is replaced to count allocations inside a window
// opened and closed by a sink after the warm-up frames.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>

#if defined(__linux__) && defined(__GLIBC__)
#include <execinfo.h>
#define TAKT_HAVE_BACKTRACE 1
#endif

#include "takt/infer/fake_backend.hpp"
#include "takt/io/synthetic_source.hpp"
#include "takt/pipeline/pipeline.hpp"

namespace {

std::atomic<bool> g_counting{false};
std::atomic<std::size_t> g_allocations{0};

// On failure the test prints where the first few counted allocations came from.
constexpr std::size_t kRecorded = 8;
constexpr int kFrames = 32;
struct AllocationSite {
  std::size_t size = 0;
  int depth = 0;
  std::array<void*, kFrames> frames{};
};
std::array<AllocationSite, kRecorded> g_sites{};
thread_local bool t_in_hook = false;  // backtrace() may itself allocate on first use

void* counted_alloc(std::size_t size) {
  if (g_counting.load(std::memory_order_relaxed) && !t_in_hook) {
    const std::size_t index = g_allocations.fetch_add(1, std::memory_order_relaxed);
    if (index < kRecorded) {
      g_sites[index].size = size;
#ifdef TAKT_HAVE_BACKTRACE
      t_in_hook = true;
      g_sites[index].depth = backtrace(g_sites[index].frames.data(), kFrames);
      t_in_hook = false;
#endif
    }
  }
  if (void* p = std::malloc(size == 0 ? 1 : size)) return p;
  throw std::bad_alloc();
}

void print_allocation_sites() {
  const std::size_t n = std::min(g_allocations.load(), kRecorded);
  for (std::size_t i = 0; i < n; ++i) {
    std::fprintf(stderr, "--- allocation %zu: %zu bytes\n", i, g_sites[i].size);
#ifdef TAKT_HAVE_BACKTRACE
    backtrace_symbols_fd(g_sites[i].frames.data(), g_sites[i].depth, 2);
#endif
  }
}

}  // namespace

void* operator new(std::size_t size) {
  return counted_alloc(size);
}
void* operator new[](std::size_t size) {
  return counted_alloc(size);
}
void operator delete(void* p) noexcept {
  std::free(p);
}
void operator delete[](void* p) noexcept {
  std::free(p);
}
void operator delete(void* p, std::size_t) noexcept {
  std::free(p);
}
void operator delete[](void* p, std::size_t) noexcept {
  std::free(p);
}

namespace takt {
namespace {

constexpr std::uint64_t kWarmupFrames = 100;
constexpr std::uint64_t kMeasuredFrames = 400;

class AllocationWindowSink final : public FrameSink {
 public:
  void consume(const Frame& frame) override {
    if (frame.sequence == kWarmupFrames) g_counting.store(true);
    if (frame.sequence == kWarmupFrames + kMeasuredFrames) g_counting.store(false);
  }
};

TEST(ZeroAllocation, SteadyStatePipelineDoesNotAllocate) {
  FakeBackendOptions backend;
  backend.input_width = 160;
  backend.input_height = 160;
  backend.num_anchors = 525;
  backend.num_classes = 2;
  backend.num_objects = 3;

  PipelineConfig config;
  config.ingress_policy = IngressPolicy::kBlock;  // every frame flows through every stage
  config.enable_tracking = true;
  Pipeline pipeline(config, FakeBackend(backend));
  pipeline.add_sink(std::make_unique<AllocationWindowSink>());

  SyntheticSource source(
      SyntheticSourceOptions{320, 240, 0.0, kWarmupFrames + kMeasuredFrames + 50, 3});
  const RunReport report = pipeline.run(source);

  ASSERT_EQ(report.frames_completed, kWarmupFrames + kMeasuredFrames + 50);
  if (g_allocations.load() != 0) print_allocation_sites();
  EXPECT_EQ(g_allocations.load(), 0u) << "heap allocations during " << kMeasuredFrames
                                      << " steady-state frames across all pipeline threads";
}

}  // namespace
}  // namespace takt

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

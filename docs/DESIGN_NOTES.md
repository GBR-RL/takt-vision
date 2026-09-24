# Design notes

The decisions behind the code, written as the questions a reviewer or interviewer is likely to
ask. Each answer points at the code and, where possible, at a test or measurement that backs it.

## Contents

1. [Latency: why "latest" everywhere?](#latency)
2. [Why a lock-free SPSC ring, and how is it correct?](#spsc)
3. [Why a triple buffer for the "latest" mailbox?](#triple-buffer)
4. [Blocking without spinning: C++20 `atomic::wait`](#atomic-wait)
5. [Zero allocations per frame - how, and how is it proven?](#zero-alloc)
6. [Why `unique_ptr` with a custom deleter for frames?](#frameptr)
7. [Concepts + type erasure for backends](#backends)
8. [Why fuse preprocessing into one pass?](#letterbox)
9. [Why decode class-major?](#decode)
10. [Tail latency and the histogram](#histogram)
11. [Why compile-time-sized matrices instead of Eigen?](#matrix)
12. [The assignment gating detail](#assignment)
13. [Shutdown and exceptions across threads](#shutdown)
14. [ONNX Runtime thread spinning](#spinning)
15. [When does pipelining pay off?](#pipelining)
16. [Parity with Ultralytics](#parity)
17. [Known limitations](#limitations)

---

<a id="latency"></a>
## 1. Latency: why "latest" everywhere?

A pipeline is only as fast as its slowest stage (the bottleneck, normally inference). Any frame
waiting *in front of* the bottleneck waits a full inference period per slot. The first version
dropped stale frames only at the camera, with depth-2 queues between stages. With a 40 ms fake
detector the measured end-to-end p50 was **231 ms**: the preprocess thread took the newest frame,
then sat blocked holding it until inference was free, while another frame sat in the queue.

Two fixes, each measured (`takt_run --backend fake --fake-latency-ms 40 --policy latest`):

| Change | p50 end-to-end (dev laptop) |
|---|---:|
| Depth-2 internal queues, drop only at camera | 231 ms |
| Depth-1 internal queues | 188 ms |
| **Latest-mailbox at every hand-off (current)** | **124 ms** |

*(The development laptop is Windows with a libc++ build whose `atomic::wait` falls back to
sleep-polling, which inflates all three. On Linux in CI the current design measures **57 ms p50 /
74 ms p99** for the same 40 ms detector, close to the theoretical 1.5 inference periods.)*

Rule learned: **shed load immediately in front of the bottleneck, not just at the entrance.**
The `block` and `drop-newest` policies keep blocking FIFOs internally, because their promise is
different: every admitted frame gets processed.

Code: `src/pipeline/pipeline.cpp` (`RunContext::internal_policy`), `include/takt/pipeline/hand_off.hpp`.

<a id="spsc"></a>
## 2. Why a lock-free SPSC ring, and how is it correct?

Each hand-off has exactly one producer thread and one consumer thread, so the general MPMC
problem does not arise. With a single writer per index:

- The producer owns `tail_`, the consumer owns `head_`. Each loads its own index `relaxed`
  (nobody else writes it) and the other side's index with `acquire`.
- The producer writes the slot, *then* publishes `tail_ + 1` with `release`. The consumer's
  `acquire` load of `tail_` therefore sees the slot contents. Symmetrically for `head_`, which
  tells the producer a slot is free to overwrite.
- Each side caches the other side's index (`cached_head_`, `cached_tail_`) and only re-reads
  the shared atomic when the cache says full/empty. In steady state most pushes touch no
  cache line the other thread writes.
- `head_` and `tail_` sit on separate 64-byte cache lines (`alignas`) to avoid false sharing.
- Indices are unbounded 64-bit counters masked into a power-of-two array; `tail - head` is
  the size, and wrap-around would take centuries.

Tested by `SpscRing.ConcurrentTransferPreservesOrder` (200 000 items through an 8-slot ring),
which CI runs under ThreadSanitizer.

<a id="triple-buffer"></a>
## 3. Why a triple buffer for the "latest" mailbox?

The producer must never block (a camera cannot wait) and the consumer must always get the newest
value. Three slots rotate: the producer writes its private *back* slot, then atomically exchanges
it with the shared *middle* slot, setting a "fresh" bit. The consumer, if the fresh bit is set,
exchanges its private *front* slot with the middle. Both operations are one `exchange` on a
single byte, so publish is wait-free.

The subtle part is ownership of displaced values: if the middle slot was still fresh when the
producer swapped, nobody consumed it, so the producer gets it back and returns the frame to the
pool (and counts a drop). Invariant: a non-fresh middle slot is always empty.
`LatestSlot.ConcurrentPublishIsLossFreeAccountingAndMonotonic` checks `taken + displaced ==
published` and strictly increasing values under concurrency.

<a id="atomic-wait"></a>
## 4. Blocking without spinning: C++20 `atomic::wait`

An idle stage should cost no CPU, particularly on a 4-core Raspberry Pi where a spinning thread
steals a core from inference. Blocking push/pop wait on an *epoch* counter:

```cpp
const auto seen = pushed_.load(std::memory_order_acquire);  // 1. read the epoch
if (auto v = try_pop()) return v;                           // 2. re-check the ring
pushed_.wait(seen);                                         // 3. sleep only if unchanged
```

Because the producer bumps the epoch *after* publishing an element, a push between steps 2 and
3 changes the epoch and `wait` returns immediately. No lost wake-ups, no mutex, no condition
variable. `close()` bumps both epochs and uses `notify_all`.

<a id="zero-alloc"></a>
## 5. Zero allocations per frame - how, and how is it proven?

- All frames are allocated once in the `FramePool`, with tensors sized from the model.
- Every per-frame container (`Image`, detections, tracks, NMS scratch, tracker scratch, Hungarian
  matrices, the JSON line buffer) is reused: `clear()` keeps capacity, and `resize()` to the same
  size does not allocate.
- The free list reserves capacity up front, so recycling cannot allocate (or throw).

**A real bug this test caught.** It failed intermittently in CI with exactly 2 allocations.
The pool is LIFO (the hottest frame is reused first), so during warm-up only some frames were
ever used; when a scheduling hiccup later drained the pool deeper, two never-used frames
allocated their image buffers mid-run. In production that is a random latency spike. The fix:
once the first frame reveals the resolution, `FramePool::reserve_image_bytes` pre-sizes every
frame.

`tests/test_zero_alloc.cpp` replaces global `operator new`, runs 100 warm-up frames through the
full threaded pipeline with tracking, then counts allocations across *all* threads for the next
400 frames: the test asserts **0**. ONNX Runtime's internal allocations are outside this claim
(the test uses the fake backend); takt's own code is allocation-free.

<a id="frameptr"></a>
## 6. Why `unique_ptr` with a custom deleter for frames?

`FramePtr = std::unique_ptr<Frame, FrameRecycler>`. The deleter returns the frame to the pool
instead of freeing it. A frame can leave the pipeline in many ways: delivered, displaced by a
newer frame, rejected by a full queue, abandoned during shutdown, or unwound by an exception.
With a manual `pool.release(frame)` call, every one of those paths needs remembering. With
RAII, all of them are correct automatically. `FramePool.DroppingAFramePtrReturnsItToThePool`.

<a id="backends"></a>
## 7. Concepts + type erasure for backends

`InferenceBackend` is a C++20 concept: anything with `name()`, `input_spec()`, `output_spec()`
and `infer(span<const float>, span<float>)` qualifies, and misuse fails at compile time with a
readable error. The pipeline needs to choose the engine at *runtime* (a CLI flag), so `Backend`
wraps any conforming type behind an internal virtual interface (Sean Parent's "type erasure").
Engines stay plain value types, testable on their own, with no base class to inherit.
`static_assert(!InferenceBackend<MissingInfer>)` in `tests/test_backend.cpp` checks the concept
rejects incomplete engines.

<a id="letterbox"></a>
## 8. Why fuse preprocessing into one pass?

The usual Python path is `resize → copyMakeBorder → BGR→RGB → transpose → astype → /255`: six
passes over the image, most allocating a temporary. `Letterboxer::run` computes each output float
once from four source pixels (bilinear, half-pixel centres like `cv::resize`) and writes it
straight into the model's CHW input tensor. Interpolation tables are rebuilt only when the source
resolution changes. The geometry reproduces Ultralytics' `LetterBox` exactly, including the
`round(dw - 0.1)` padding rule, so boxes line up with the reference implementation
(`Letterbox.GeometryMatchesUltralytics`).

<a id="decode"></a>
## 9. Why decode class-major?

The YOLO11 head is `[1, 84, 8400]`: row *c* holds class *c*'s score for all 8400 anchors. Looping
anchor-first reads with a stride of 8400 floats (33 KB), which misses the cache on nearly every
access. Looping class-first reads each row sequentially while updating a best-score array, and
the inner loop auto-vectorises.

<a id="histogram"></a>
## 10. Tail latency and the histogram

A production line runs on a takt time. A detector averaging 8 ms that spikes to 40 ms every
hundredth frame fails a line that a steady 12 ms passes, so means are not enough. The histogram
is log-linear (HdrHistogram-style): exact below 128 µs, then 64 sub-buckets per power of two,
bounding percentile error to 1/64. `record()` is a few relaxed atomic increments, so all threads
record concurrently and the HUD reads percentiles live. Reported percentiles are the bucket's
upper bound (conservative), clamped to the true maximum.

<a id="matrix"></a>
## 11. Why compile-time-sized matrices instead of Eigen?

The tracker needs 8×8, 8×4 and 4×4 products and one 4×4 SPD solve per track per frame. A
~150-line `Matrix<R, C>` gives stack allocation, compile-time dimension checking (multiplying
mismatched shapes does not compile), `constexpr` use, and no dependency. The Kalman gain is
computed with a Cholesky solve rather than an explicit inverse (faster, numerically stabler), and
exploits that the measurement matrix just selects the first four states.

<a id="assignment"></a>
## 12. The assignment gating detail

ByteTrack matches tracks to detections by IoU with a maximum cost. The shortcut "solve the
assignment, then discard pairs above the limit" is not optimal *under* the limit: it can force a
bad pair in order to enable another. `LinearAssignment` does what `lap.lapjv(extend_cost=True,
cost_limit=L)` does: it extends the matrix so every row and column may go unmatched at cost L/2,
then solves exactly. It is verified against brute force on 300 random rectangular problems
(`LinearAssignment.MatchesBruteForceOnRandomRectangularProblems`).

<a id="shutdown"></a>
## 13. Shutdown and exceptions across threads

- `std::jthread` joins in its destructor, and threads are declared after the queues and the
  pool they use, so destruction order guarantees nothing outlives what it references.
- A `std::stop_callback` closes the ingress when a stop is requested. Without it, a capture
  thread blocked on a full `block` queue could never observe the stop.
- `FramePool::acquire` uses `condition_variable_any::wait(lock, stop_token, pred)`, which wakes
  on stop without a hand-rolled flag.
- Any stage exception is captured (`std::exception_ptr`), all hand-offs are closed, and the
  stages drain without processing. `run()` rethrows on the caller's thread.
  `Pipeline.StageFailureIsRethrownWithoutDeadlock` then reuses the same pipeline successfully.

<a id="spinning"></a>
## 14. ONNX Runtime thread spinning

ONNX Runtime's intra-op worker threads spin-wait between runs by default. In a benchmark loop
that lowers latency. In a pipeline it burns cores that preprocessing, tracking and capture need,
and on a 4-core edge device that should hurt. takt disables spinning by default
(`OnnxBackendOptions::allow_spinning`). This is a hypothesis until measured: the M4 plan sweeps
spinning × thread count on the Raspberry Pi 5 and sets the default from the data.

<a id="pipelining"></a>
## 15. When does pipelining pay off?

Measured in CI (4 vCPU, YOLO11n on ONNX Runtime CPU, offline video with `--policy block`):
pipelined **17.0 fps** vs sequential **16.9 fps**, so no gain. Inference is ~55 ms of a ~59 ms
frame, and with stages on separate threads it slows slightly (~59 ms) because the other stages
compete for the same four cores.

Pipelined throughput is bounded by the slowest stage, sequential by the sum of stages. The gain
is therefore at most (sum / max), here 59 / 55 ≈ 1.07×, and CPU contention eats that. Pipelining
matters when inference leaves the CPU (CUDA, TensorRT, an NPU such as Hailo), because the
remaining CPU stages then overlap with it rather than compete. Recording this as a measured null
result, not a claim, is deliberate; the Raspberry Pi 5 + Hailo and Jetson runs in M4/M5 are where
the design earns its keep.

<a id="parity"></a>
## 16. Parity with Ultralytics

The demo workflow runs Ultralytics' own `predict()` and `takt_run` on the *same ONNX file*, so any
difference comes from the parts takt reimplements: letterbox, decoding, NMS and box scaling
(`scripts/parity_reference.py`, `scripts/check_parity.py`). Every detection must have a
same-class partner, boxes must agree within 1 pixel of the model input, and scores within 0.01.

The first run failed on `zidane.jpg` with a 1.09 px deviation (source pixels) while `bus.jpg`
passed at 0.21 px. The investigation, reproduced in Python on the same model:

| Input path | bus (scale 0.59) | zidane (scale 0.50) |
|---|---:|---:|
| float bilinear (takt) vs 8-bit OpenCV resize (Ultralytics), source px | 0.18 | 1.12 |
| same, model-input px | 0.10 | 0.56 |
| float bilinear rounded to 8-bit, source px | 0.30 | 1.11 |

The C++ output matched the float simulation, so the implementation was right. The residual comes
from OpenCV resizing in 8-bit fixed point (11-bit weights) while takt interpolates in float.
Inputs differ by at most half a grey level, which moves boxes by about half a model pixel; mapping
back to a 2x-downscaled source doubles that. Rounding to 8-bit does not close the gap, because
OpenCV's weights are fixed-point too. Bit-exact parity would mean reimplementing OpenCV's
arithmetic for no accuracy benefit, so the tolerance is defined in model-input pixels, the
detector's own resolution, where both images agree to well under one pixel.

<a id="limitations"></a>
## 17. Known limitations

- Batch size 1 and one model per pipeline (by design for per-camera latency).
- Only float32 raw YOLO heads; end-to-end (NMS-in-graph) exports are rejected with a clear error.
- The overlay sink renders on the sink thread; at high resolution it can become the bottleneck
  (visible in the HUD stage bar).
- The gated Hungarian solver is O((tracks + detections)³): about 1 ms at 50 objects. See M3.

# Implementation plan

takt-vision is built in milestones. Each one ships on its own, is backed by tests and CI, and
leaves the repository in a state worth showing. This file is the source of truth for scope and
status.

**Legend:** ✅ done · 🔄 in progress · ⬜ planned

## Why this project, and why this shape

The project was derived from an analysis of 101 job descriptions the author applied to in
September 2026 (computer vision, perception, robotics and machine-vision roles in Germany):

| Requirement mentioned | JDs | What that means for this repo |
|---|---:|---|
| C++ | 30 | Real systems C++, not a thin API wrapper |
| Performance optimisation | 13 of the C++ JDs | Measured latency and throughput, per stage |
| Unit / integration testing | 12 of the C++ JDs | GoogleTest, sanitizers, CI on every push |
| ROS / ROS 2 | 10 of the C++ JDs | A thin ROS 2 adapter, not a ROS-first design |
| Real-time / latency | 8 of the C++ JDs | Tail latency (p99), back-pressure, drop policies |
| Multithreading (C++17/20) | explicit in C++-first roles | Lock-free hand-offs, `jthread`, `stop_token` |
| TensorRT | 3 of 101 | One backend among several, not the whole project |

So the core is a *runtime*: the part between the camera and a decision, where the C++
engineering lives. Inference engines plug into it.

---

## M0 - Foundations ✅

- CMake ≥ 3.25 with presets (`release`, `debug`, `core`, `asan`, `tsan`, `bench`)
- Strict warnings (`-Wconversion -Wsign-conversion -Wshadow ...`) as errors in CI
- Dependencies fetched reproducibly: ONNX Runtime prebuilt per platform, GoogleTest, CLI11,
  Google Benchmark
- CI matrix: GCC 14 x86-64, GCC ARM64, Clang 18 ASan+UBSan, Clang 18 TSan, clang-format check

## M1 - Correct single-frame inference ✅ (parity test 🔄)

- Fused letterbox preprocessing (resize + pad + BGR→RGB + normalise + HWC→CHW in one pass)
- ONNX Runtime backend bound to caller-owned buffers (no per-frame tensor allocation)
- YOLOv8 / YOLO11 head decoder (class-major scan) and torchvision-equivalent NMS
- `takt_bench`: per-stage p50/p95/p99 on one thread
- 🔄 **Parity test:** C++ detections on `bus.jpg` match Ultralytics within 1 px / 0.01 score

## M2 - Real-time pipeline ✅

- One thread per stage, connected by `HandOff`s: SPSC ring buffer (block / drop-newest) or a
  triple-buffer "latest" mailbox
- Load shedding in front of the bottleneck (see [DESIGN_NOTES.md](DESIGN_NOTES.md#latency))
- Frame pool with RAII recycling; **zero heap allocations per frame in steady state** (tested)
- Latency histograms (HdrHistogram-style, wait-free record) per stage and end to end
- Graceful stop (`std::stop_token`), exception propagation from any stage thread
- `takt_run` CLI: camera / video / image / RTSP / synthetic sources, overlay video, JSON lines

## M3 - Multi-object tracking ✅

- ByteTrack: Kalman filter on compile-time-sized matrices, gated optimal assignment
  (Hungarian with lapjv-style cost-limit extension), three-stage association
- Tests: identity kept through crossings, short occlusions and confidence dips
- ⬜ Optimisation: 50 simultaneous objects cost ~1 ms because the gated solver is O(n³) on the
  extended matrix. Split into connected components first (most frames are many small problems).

## M4 - Benchmarks and edge deployment 🔄

- ✅ Overload experiment (camera faster than detector) and chart, fully reproducible in CI:
  `latest` holds 57 ms p50 / 74 ms p99 with a 40 ms detector; an unbounded FIFO grows past 4 s
- ✅ C++ vs Python stage comparison on the same model and machine: pre + post 3.5 ms vs 6.1 ms (1.7×)
- ✅ Pipelined vs sequential on a 4-vCPU CPU-only runner: no gain (17.0 vs 16.9 fps), explained in
  DESIGN_NOTES §15. Re-measure with inference on an accelerator.
- ⬜ Raspberry Pi 5 (ARM64, 4 cores): build natively, publish the latency table
- ⬜ `allow_spinning` on/off and `--threads` sweep: does ORT's spin-waiting steal cores from
  the pipeline on a 4-core device? (Hypothesis in DESIGN_NOTES; measure, then decide the default.)
- ⬜ NEON/AVX2 letterbox kernel, measured against the scalar version and `cv::resize`

## M5 - Accelerators ⬜

- TensorRT backend (C++ API, `enqueueV3`, pinned staging buffers, CUDA stream) behind the same
  `InferenceBackend` concept; compile-checked in CI inside NVIDIA's TensorRT container
- FP32 / FP16 / INT8 table on a Jetson Orin Nano (or a CLAIX GPU node for the desktop numbers)
- INT8 calibration from a calibration set; accuracy delta reported next to the speed-up

## M6 - Integration ⬜

- ROS 2 node (`takt_ros`): subscribes `sensor_msgs/Image`, publishes `vision_msgs/Detection2DArray`,
  runs the same pipeline; a separate colcon package so the core keeps zero ROS dependencies
- MQTT sink for PLC / MES gateways (industrial integration)
- MJPEG-over-HTTP sink: open the live annotated stream from a phone during an interview

## M7 - Showcase ⬜

- Hero GIF recorded on the Raspberry Pi with a USB camera over a moving part (own footage)
- 60-second screen recording for LinkedIn, a short write-up of the latency findings
- CV bullet (below) updated with the measured numbers

---

## Demos recruiters can see

| Demo | Shows | Where it comes from |
|---|---|---|
| Annotated video with live HUD (FPS, p50/p99, stage bar, drops, track IDs + trails) | It works, in real time | `takt_run --out-video` / CI artefact |
| Overload chart: unbounded FIFO vs drop-newest vs latest | Understanding of real-time systems | `scripts/overload_experiment.sh` |
| C++ vs Python per stage | Where C++ pays off (pre/post-processing), honestly | `takt_bench` + `python_baseline.py` |
| Green CI with sanitizers on x86-64 and ARM64 | Engineering discipline | badge in README |
| Zero-allocation test | Memory discipline, provable | `tests/test_zero_alloc.cpp` |
| Live webcam demo (`--show`) | Interview-ready in 10 seconds | laptop or Raspberry Pi |

Details and recording instructions: [DEMOS.md](DEMOS.md).

## CV bullet (fill in the numbers as milestones land)

> Built **takt-vision**, a multi-threaded C++20 inference pipeline for edge vision (lock-free
> SPSC/triple-buffer hand-offs, zero steady-state heap allocations, ONNX Runtime backend,
> ByteTrack tracking). Bounded end-to-end latency under overload (57 ms p50 / 74 ms p99 with a
> 40 ms detector, vs. >4 s for a naive queue) and 1.7× faster pre/post-processing than the
> equivalent Python pipeline; CI on x86-64 and ARM64 with ASan, UBSan and TSan.
>
> *(Replace the numbers with Raspberry Pi 5 measurements once M4 lands.)*

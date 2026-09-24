# Demos

How to produce every demo in the README, plus the ones worth having ready for an interview.

## 1. Annotated video with live telemetry (the hero GIF)

```bash
python scripts/export_yolo.py --weights yolo11n.pt --out models
./build/release/apps/takt_run --model models/yolo11n.onnx --source input.mp4 \
    --policy block --no-realtime --out-video demo.mp4
ffmpeg -i demo.mp4 -vf "fps=12,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" -t 12 demo.gif
```

The overlay shows boxes with track IDs and motion trails, and a HUD with FPS, end-to-end
p50/p99, per-stage p50 times, a stacked bar of where each frame's time goes, and the drop count.

**No local build?** Run the *Demo & benchmarks* workflow on GitHub (Actions → Run workflow),
optionally with your own `video_url`. It uploads `demo.mp4`, `demo.gif` and all charts as an
artefact.

**Footage.** Record your own: a phone or USB camera over parts moving on a desk or small
conveyor fits the industrial-inspection story and avoids licensing questions. The CI default
(OpenCV's `vtest.avi`) is only for smoke-testing the workflow.

## 2. Live camera, in an interview

```bash
./build/release/apps/takt_run --model models/yolo11n.onnx --source 0 --show
```

Press `q` or Esc to stop; the run report prints on exit. On a Raspberry Pi 5 with a USB camera
this is the most convincing demo there is: it runs, on the edge, right now.

## 3. The overload experiment (no model needed)

```bash
scripts/overload_experiment.sh ./build/release/apps/takt_run 20
```

A 30 fps synthetic camera feeds a simulated 40 ms detector under three policies, and the script
plots end-to-end latency over time. The unbounded FIFO's latency climbs without limit, which is
the classic bug in naive producer/consumer code. `drop-newest` and `latest` stay flat, and
`latest` stays lowest. This chart makes the real-time argument in one picture.

## 4. C++ vs Python, stage by stage

```bash
./build/release/apps/takt_bench --model models/yolo11n.onnx --image bus.jpg --json results/cpp.json
python scripts/python_baseline.py --model models/yolo11n.onnx --image bus.jpg --json results/python.json
python scripts/plot_latency.py compare --run "C++=results/cpp.json" --run "Python=results/python.json" --out docs/assets/cpp_vs_python
```

Present this one honestly: inference takes the same time in both, because it is the same ONNX
Runtime engine. The gains are in preprocessing and postprocessing, and in the pipeline overlapping
stages. That precision is itself a signal to a technical reviewer.

## 5. Pipelined vs sequential throughput

```bash
./build/release/apps/takt_run --model models/yolo11n.onnx --source input.mp4 --no-realtime --policy block
./build/release/apps/takt_run --model models/yolo11n.onnx --source input.mp4 --no-realtime --sequential
```

Offline comparisons need the lossless `block` policy. With stages on separate threads, throughput
approaches 1 / (slowest stage) instead of 1 / (sum of stages). On a CPU-only machine where
inference dominates, expect little gain; see DESIGN_NOTES §15.

## 6. Parity with Ultralytics

```bash
python scripts/parity_reference.py --model models/yolo11n.onnx --image bus.jpg --json ref.json
./build/release/apps/takt_run --model models/yolo11n.onnx --source bus.jpg --policy block --no-track --jsonl takt.jsonl
python scripts/check_parity.py ref.json takt.jsonl
```

"How do you know your C++ reimplementation is correct?" is the first question a sceptical
reviewer asks. This is the answer, and it runs in CI.

## 7. The engineering evidence

- CI badge: GCC and Clang, x86-64 and ARM64, ASan + UBSan + TSan, on every push.
- `tests/test_zero_alloc.cpp`: zero heap allocations per frame in steady state, asserted.
- `docs/DESIGN_NOTES.md`: the reasoning, with measurements.

## Recording checklist for LinkedIn / portfolio

1. 60 seconds, screen plus camera: the live `--show` window with the HUD visible.
2. Cut to the overload chart and explain it in one sentence.
3. End on the repository page with the green CI badge.

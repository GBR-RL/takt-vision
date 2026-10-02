# Reproducing the results

Every chart and video in the README comes from one of the commands below. The *Demo &
benchmarks* workflow (Actions → Run workflow) runs all of them on a GitHub-hosted runner and
uploads the outputs as an artefact; an optional `video_url` input replaces the test clip.

## Annotated video with live telemetry

```bash
python scripts/export_yolo.py --weights yolo11n.pt --out models
./build/release/apps/takt_run --model models/yolo11n.onnx --source input.mp4 \
    --policy block --no-realtime --out-video demo.mp4
ffmpeg -i demo.mp4 -vf "fps=12,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" -t 12 demo.gif
```

The overlay draws boxes with track IDs and motion trails, and a HUD with FPS, end-to-end p50/p99,
per-stage p50 times, a stacked bar of where each frame's time goes, and the drop count. The CI
default clip (OpenCV's `vtest.avi`) only smoke-tests the workflow.

## Live camera

```bash
./build/release/apps/takt_run --model models/yolo11n.onnx --source 0 --show
```

`q` or Esc stops the run; the run report prints on exit.

## Overload experiment (no model needed)

```bash
scripts/overload_experiment.sh ./build/release/apps/takt_run 20
```

A 30 fps synthetic camera feeds a simulated 40 ms detector under three ingress policies, and the
script plots end-to-end latency over time. The unbounded FIFO's latency grows without limit;
`drop-newest` and `latest` stay flat, `latest` lowest.

## C++ vs Python, stage by stage

```bash
./build/release/apps/takt_bench --model models/yolo11n.onnx --image bus.jpg --json results/cpp.json
python scripts/python_baseline.py --model models/yolo11n.onnx --image bus.jpg --json results/python.json
python scripts/plot_latency.py compare --run "C++=results/cpp.json" --run "Python=results/python.json" --out docs/assets/cpp_vs_python
```

Inference takes the same time on both sides because it is the same ONNX Runtime engine; the
differences are in preprocessing and postprocessing.

## Pipelined vs sequential throughput

```bash
./build/release/apps/takt_run --model models/yolo11n.onnx --source input.mp4 --no-realtime --policy block
./build/release/apps/takt_run --model models/yolo11n.onnx --source input.mp4 --no-realtime --sequential
```

Offline comparisons need the lossless `block` policy. With stages on separate threads, throughput
approaches 1 / (slowest stage) instead of 1 / (sum of stages); where inference dominates on a
CPU-only machine the gain is small (see [design notes](DESIGN_NOTES.md#pipelining)).

## Parity with Ultralytics

```bash
python scripts/parity_reference.py --model models/yolo11n.onnx --image bus.jpg --json ref.json
./build/release/apps/takt_run --model models/yolo11n.onnx --source bus.jpg --policy block --no-track --jsonl takt.jsonl
python scripts/check_parity.py ref.json takt.jsonl
```

The same check runs in CI on every push.

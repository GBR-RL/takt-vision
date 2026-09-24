#!/usr/bin/env bash
# The overload experiment: a 30 fps camera feeding a detector that manages only ~25 fps.
# Runs the same scenario under each ingress policy and plots end-to-end latency over time.
#
#   scripts/overload_experiment.sh build/release/apps/takt_run [seconds]
#
# No model or GPU needed: the fake backend simulates a 40 ms inference.
set -euo pipefail

TAKT_RUN=${1:?usage: overload_experiment.sh path/to/takt_run [seconds]}
SECONDS_TO_RUN=${2:-20}
FPS=30
FRAMES=$((SECONDS_TO_RUN * FPS))
OUT=results/overload
mkdir -p "$OUT"

run() {  # name, extra args...
  local name=$1
  shift
  "$TAKT_RUN" --backend fake --fake-input 160 --fake-latency-ms 40 \
    --source synthetic --synthetic-width 640 --synthetic-height 360 --synthetic-fps $FPS \
    --max-frames "$FRAMES" --stats-ms 0 --jsonl "$OUT/$name.jsonl" --report "$OUT/$name.json" "$@"
}

# "Naive FIFO": 256 slots never fill in a 20 s run (the backlog grows ~5 frames/s), so latency
# grows without bound - exactly what an unbounded std::queue between threads does.
run block --policy block --ingress-capacity 256
run drop-newest --policy drop-newest --ingress-capacity 4
run latest --policy latest

python3 "$(dirname "$0")/plot_latency.py" overload \
  --run "unbounded FIFO=$OUT/block.jsonl" \
  --run "drop-newest (4)=$OUT/drop-newest.jsonl" \
  --run "latest (takt default)=$OUT/latest.jsonl" \
  --subtitle "Camera 30 fps, detector 40 ms (~25 fps). Same pipeline, three ingress policies." \
  --out docs/assets/overload

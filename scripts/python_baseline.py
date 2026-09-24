"""Per-stage latency of a typical Python deployment of the same model, for comparison with
takt_bench. Same stages, same ONNX Runtime settings, same JSON schema:

    python scripts/python_baseline.py --model models/yolo11n.onnx --image bus.jpg --json results/python.json

The Python pipeline mirrors common YOLO ONNX deployment code: cv2.resize + copyMakeBorder for the
letterbox, NumPy for layout and normalisation, NumPy decoding and cv2.dnn.NMSBoxesBatched.
Inference itself runs in the same C++ engine either way, so differences come from pre/post-
processing and interpreter overhead - which is exactly what the comparison is meant to show.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def letterbox(image: np.ndarray, width: int, height: int):
    h, w = image.shape[:2]
    r = min(height / h, width / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    if (new_w, new_h) != (w, h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (width - new_w) / 2, (height - new_h) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    blob = image[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(blob), r, left, top


def postprocess(output: np.ndarray, conf: float, iou: float, r: float, left: int, top: int):
    pred = output[0]
    if pred.shape[0] > pred.shape[1]:
        pred = pred.T
    scores = pred[4:]
    class_ids = scores.argmax(axis=0)
    best = scores.max(axis=0)
    keep = best > conf
    boxes, best, class_ids = pred[:4, keep].T, best[keep], class_ids[keep]
    xywh = np.column_stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2, boxes[:, 2], boxes[:, 3]])
    indices = cv2.dnn.NMSBoxesBatched(xywh.tolist(), best.tolist(), class_ids.tolist(), conf, iou)
    indices = np.asarray(indices, dtype=int).reshape(-1)
    result = xywh[indices].copy()
    result[:, 0] = (result[:, 0] - left) / r
    result[:, 1] = (result[:, 1] - top) / r
    result[:, 2:] /= r
    return result, best[indices], class_ids[indices]


def summarize(samples_ms):
    a = np.asarray(samples_ms)
    q = lambda p: float(np.percentile(a, p))  # noqa: E731
    return {"count": int(a.size), "mean_ms": float(a.mean()), "min_ms": float(a.min()), "p50_ms": q(50),
            "p90_ms": q(90), "p95_ms": q(95), "p99_ms": q(99), "p999_ms": q(99.9), "max_ms": float(a.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", help="input image (default: synthetic 1280x720 frame)")
    parser.add_argument("-n", "--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--json")
    args = parser.parse_args()

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.inter_op_num_threads = 1
    if args.threads > 0:
        options.intra_op_num_threads = args.threads
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")  # same as takt default
    session = ort.InferenceSession(args.model, options, providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    height, width = inp.shape[2], inp.shape[3]

    if args.image:
        image = cv2.imread(args.image, cv2.IMREAD_COLOR)
        image_desc = args.image
    else:
        image = np.full((720, 1280, 3), 28, np.uint8)
        image_desc = "synthetic 1280x720"

    stages = {"preprocess": [], "inference": [], "postprocess": [], "total": []}
    detections = 0
    for i in range(args.warmup + args.iterations):
        t0 = time.perf_counter()
        blob, r, left, top = letterbox(image, width, height)
        t1 = time.perf_counter()
        output = session.run(None, {inp.name: blob})[0]
        t2 = time.perf_counter()
        boxes, _, _ = postprocess(output, args.conf, args.iou, r, left, top)
        t3 = time.perf_counter()
        if i >= args.warmup:
            stages["preprocess"].append((t1 - t0) * 1e3)
            stages["inference"].append((t2 - t1) * 1e3)
            stages["postprocess"].append((t3 - t2) * 1e3)
            stages["total"].append((t3 - t0) * 1e3)
        detections = len(boxes)

    result = {"implementation": "python", "backend": "onnxruntime-cpu", "image": image_desc, "width": width,
              "height": height, "iterations": args.iterations, "detections": detections,
              "stages": {k: summarize(v) for k, v in stages.items()}}
    print(f"python_baseline | {image_desc} | {width}x{height} | {args.iterations} iterations")
    for name, s in result["stages"].items():
        print(f"  {name:<12} mean {s['mean_ms']:8.3f}  p50 {s['p50_ms']:8.3f}  p99 {s['p99_ms']:8.3f} ms")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()

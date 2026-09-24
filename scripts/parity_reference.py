"""Reference detections from Ultralytics for the parity check.

Runs Ultralytics' own predict() on the *same ONNX file* takt-vision uses, so any difference comes
from preprocessing, decoding, NMS or box scaling - the parts takt reimplements in C++.

    python scripts/parity_reference.py --model models/yolo11n.onnx --image bus.jpg --json ref.json
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model, task="detect")
    result = model.predict(args.image, imgsz=args.imgsz, conf=args.conf, iou=args.iou, max_det=300,
                           agnostic_nms=False, device="cpu", verbose=False)[0]
    boxes = result.boxes
    detections = [
        {"cls": int(c), "score": float(s), "box": [float(v) for v in b]}
        for b, s, c in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())
    ]
    out = {"implementation": "ultralytics", "model": args.model, "image": args.image, "conf": args.conf,
           "iou": args.iou, "imgsz": args.imgsz, "orig_shape": list(result.orig_shape), "detections": detections}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
    print(f"ultralytics: {len(detections)} detections on {args.image}")


if __name__ == "__main__":
    main()

"""Export an Ultralytics YOLO detection model to ONNX for takt-vision.

    python scripts/export_yolo.py --weights yolo11n.pt --imgsz 640 --out models/

Produces a static-shape float32 model with the raw [1, 4 + classes, anchors] head (no NMS in the
graph) - NMS runs in C++ where it can be measured. Weights are downloaded by Ultralytics on first
use and are AGPL-3.0 licensed; they are not part of this repository.
"""

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="yolo11n.pt", help="Ultralytics .pt weights or model name")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--out", default="models", help="output directory")
    args = parser.parse_args()

    from ultralytics import YOLO  # imported late so --help works without ultralytics

    model = YOLO(args.weights)
    exported = Path(model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, dynamic=False,
                                 simplify=True, half=False, nms=False))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / exported.name
    if exported.resolve() != target.resolve():
        shutil.move(str(exported), target)
    print(f"exported {target}")


if __name__ == "__main__":
    main()

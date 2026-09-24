"""Generate a tiny ONNX "YOLO" whose output is a known constant detection head.

Used by tests/test_onnx_backend.cpp to exercise the ONNX Runtime backend and the full pipeline
without real weights. Input: images [1, 3, 64, 64]. Output: output0 [1, 6, 21] (4 box values +
2 classes, 21 anchors). The input is folded in as `mean(images) * 0`, so the graph really depends
on it and ONNX Runtime cannot constant-fold the model away.

    python scripts/make_tiny_model.py build/tiny_yolo.onnx
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ANCHORS = 21
CLASSES = 2


def build_table() -> np.ndarray:
    table = np.zeros((1, 4 + CLASSES, ANCHORS), np.float32)
    table[0, 4:, :] = 0.01  # background scores

    def put(anchor, cx, cy, w, h, cls, score):
        table[0, 0:4, anchor] = (cx, cy, w, h)
        table[0, 4 + cls, anchor] = score

    put(3, 32, 32, 20, 10, cls=1, score=0.9)   # the object
    put(7, 33, 32, 20, 10, cls=1, score=0.8)   # duplicate of it: NMS must remove
    put(10, 10, 10, 8, 8, cls=0, score=0.6)    # a second, separate object
    return table


def main(path: str) -> None:
    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])
    output = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 4 + CLASSES, ANCHORS])
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["mean"], keepdims=0),
        helper.make_node("Mul", ["mean", "zero"], ["zeroed"]),
        helper.make_node("Add", ["zeroed", "table"], ["output0"]),
    ]
    initializers = [
        numpy_helper.from_array(np.array(0.0, np.float32), "zero"),
        numpy_helper.from_array(build_table(), "table"),
    ]
    graph = helper.make_graph(nodes, "tiny_yolo", [images], [output], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)],
                              producer_name="takt-vision tests")
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tiny_yolo.onnx")

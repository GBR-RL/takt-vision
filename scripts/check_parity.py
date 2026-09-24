"""Compare takt-vision detections with the Ultralytics reference.

    python scripts/check_parity.py ref.json takt.jsonl --tol-px 1.0 --tol-score 0.01 --report parity.json

Detections are matched one-to-one (same class, greedy by IoU, IoU > 0.5). The check passes when
every detection has a partner, box corners agree within --tol-px pixels and scores within
--tol-score. A detection without a partner is tolerated only if its score is within --tol-score of
the confidence threshold: such boxes can legitimately flip either way on float differences, and
they are reported as borderline. Exit status is non-zero on failure.
"""

import argparse
import json
import sys
from pathlib import Path


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", help="JSON from parity_reference.py")
    parser.add_argument("takt", help="JSON lines from takt_run --jsonl (last frame is used)")
    parser.add_argument("--tol-px", type=float, default=1.0)
    parser.add_argument("--tol-score", type=float, default=0.01)
    parser.add_argument("--report", help="write a JSON summary here")
    args = parser.parse_args()

    ref_doc = json.loads(Path(args.reference).read_text())
    ref = sorted(ref_doc["detections"], key=lambda d: -d["score"])
    lines = [line for line in Path(args.takt).read_text().splitlines() if line.strip()]
    ours = json.loads(lines[-1])["detections"]
    conf = ref_doc.get("conf", 0.25)

    unmatched = list(range(len(ours)))
    pairs, missing = [], []
    for r in ref:
        best, best_iou = None, 0.5
        for j in unmatched:
            if ours[j]["cls"] == r["cls"]:
                v = iou(r["box"], ours[j]["box"])
                if v > best_iou:
                    best, best_iou = j, v
        if best is None:
            missing.append(r)
        else:
            unmatched.remove(best)
            pairs.append((r, ours[best], best_iou))
    extra = [ours[j] for j in unmatched]

    def borderline(d):
        return d["score"] - conf <= args.tol_score

    max_px = max((abs(a - b) for r, o, _ in pairs for a, b in zip(r["box"], o["box"])), default=0.0)
    max_score = max((abs(r["score"] - o["score"]) for r, o, _ in pairs), default=0.0)
    min_iou = min((v for *_, v in pairs), default=1.0)
    hard_missing = [d for d in missing if not borderline(d)]
    hard_extra = [d for d in extra if not borderline(d)]
    passed = not hard_missing and not hard_extra and max_px <= args.tol_px and max_score <= args.tol_score

    name = Path(ref_doc.get("image", args.reference)).name
    print(f"parity {name}: reference {len(ref)}, takt {len(ours)}, matched {len(pairs)}, "
          f"borderline {len(missing) + len(extra) - len(hard_missing) - len(hard_extra)}")
    print(f"  max box deviation {max_px:.3f} px (tol {args.tol_px}), max score deviation {max_score:.4f} "
          f"(tol {args.tol_score}), min IoU {min_iou:.4f}")
    for d in hard_missing:
        print(f"  MISSING in takt: cls {d['cls']} score {d['score']:.3f} box {d['box']}")
    for d in hard_extra:
        print(f"  EXTRA in takt:   cls {d['cls']} score {d['score']:.3f} box {d['box']}")
    print("  PASS" if passed else "  FAIL")

    if args.report:
        Path(args.report).write_text(json.dumps({
            "image": name, "passed": passed, "reference": len(ref), "takt": len(ours), "matched": len(pairs),
            "max_box_deviation_px": max_px, "max_score_deviation": max_score, "min_iou": min_iou,
            "missing": hard_missing, "extra": hard_extra, "tol_px": args.tol_px, "tol_score": args.tol_score,
        }, indent=1) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

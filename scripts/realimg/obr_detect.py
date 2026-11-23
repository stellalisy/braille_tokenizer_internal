"""Optical braille recognition on real photos: image -> braille cell string.

Runs the DotNeuralNet YOLOv8 cell detector (snoop2head/yolov8m-braille) on the real-image sets of the
"Learning to Read Braille" benchmark (Kartik0806/BrailleBench, real-braille-images/) and writes one
JSON line per image with the detected cells as Unicode braille. Lines and word spaces are inferred from
box geometry with fixed rules (no tuning on the eval images):
  - boxes are grouped into lines when their y centres differ by less than half the mean box height
    (the detector repo's own rule);
  - inside a line, a space is inserted when the centre-to-centre gap exceeds 1.6 x the line's median gap
    (an empty cell roughly doubles the pitch);
  - lines are joined with a space.
Run with the venv-obr interpreter (ultralytics 8.0.58, torch 2.1.2), not the training venv.
"""
import argparse
import json
import os

import numpy as np
from ultralytics import YOLO


def cell_char(name: str) -> str:
    """Detector class name '100100' (dot i+1 = bit i) -> Unicode braille cell."""
    return chr(0x2800 + sum(1 << i for i, b in enumerate(name) if b == "1"))


def boxes_to_text(xywh: np.ndarray, cls: np.ndarray, names: dict, space_mult: float = 1.6) -> tuple[str, int]:
    if len(xywh) == 0:
        return "", 0
    order = xywh[:, 1].argsort()
    xywh, cls = xywh[order], cls[order]
    thr = np.mean(xywh[:, 3]) / 2
    cuts = np.where(np.diff(xywh[:, 1]) > thr)[0] + 1
    lines = []
    for idx in np.split(np.arange(len(xywh)), cuts):
        idx = idx[xywh[idx, 0].argsort()]
        xs = xywh[idx, 0]
        gaps = np.diff(xs)
        med = np.median(gaps) if len(gaps) else 0
        s = cell_char(names[int(cls[idx[0]])])
        for k in range(1, len(idx)):
            if med > 0 and gaps[k - 1] > space_mult * med:
                s += " "
            s += cell_char(names[int(cls[idx[k]])])
        lines.append(s)
    return " ".join(lines), len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="BrailleBench/real-braille-images")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.15, help="detector repo default")
    a = ap.parse_args()
    model = YOLO(a.weights)
    rows = [dict(json.loads(l), set="benchmark", base="benchmark") for l in open(f"{a.root}/benchmark/labels.jsonl")]
    rows += [dict(r, set="book", base="book-dataset") for r in json.load(open(f"{a.root}/book-dataset/labels.json"))]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf8") as f:
        for r in rows:
            res = model.predict(f"{a.root}/{r['base']}/{r['image_path']}", conf=a.conf, verbose=False)[0]
            b = res.boxes
            text, n_lines = boxes_to_text(b.xywh.cpu().numpy(), b.cls.cpu().numpy(), model.names)
            f.write(json.dumps({"set": r["set"], "id": r["id"], "image": r["image_path"], "gt": r["ground_truth"],
                                "grade": r.get("grade"), "obr": text, "n_cells": int(len(b)), "n_lines": n_lines,
                                "mean_conf": float(b.conf.mean()) if len(b) else None}, ensure_ascii=False) + "\n")
    print("wrote", len(rows), "rows to", a.out)


if __name__ == "__main__":
    main()

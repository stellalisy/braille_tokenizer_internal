"""Measure the optical recognizer's cell error channel against gold cell labels (Angelina dataset).

The 401-image English test set has print labels only, so the recognizer's error process is measured
here instead: the DotNeuralNet YOLOv8 detector (same weights and threshold as obr_detect.py) runs on
Angelina images that carry a gold box and dot pattern for every cell. Detected boxes are matched to
gold boxes greedily by IoU (>= 0.3). Then:
  matched, same pattern       -> correct
  matched, different pattern  -> omit (one dot lost) / add (one dot gained) / neighbour (one dot moved
                                 to an adjacent position) / mirror (columns swapped) / multi (other)
  unmatched detection         -> insert (counted as a repeat when its pattern equals the nearest gold
                                 cell's)
  unmatched gold cell         -> delete
Rates are per gold cell; dot-level omission and addition counts are kept by dot position.

Subsets: uploaded/test (v1.1, added after the detector was published, so the closest to unseen data)
and books/val, handwritten/val (possibly in the detector's training data; reported separately).
For two-sided pages only the front is labelled, so some insertions may be back-side dots.
Run with the venv-obr interpreter.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
from ultralytics import YOLO


def bits(name):
    return sum(1 << i for i, b in enumerate(name) if b == "1")


POS = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1), 5: (1, 2)}


def mirror(m):
    return ((m & 0b111) << 3) | ((m & 0b111000) >> 3)


def classify(gold, pred):
    if pred == gold:
        return "correct"
    lost, gained = gold & ~pred, pred & ~gold
    nl, ng = bin(lost).count("1"), bin(gained).count("1")
    if nl == 1 and ng == 0:
        return "omit"
    if nl == 0 and ng == 1:
        return "add"
    if nl == 1 and ng == 1:
        (c1, r1), (c2, r2) = POS[lost.bit_length() - 1], POS[gained.bit_length() - 1]
        if abs(c1 - c2) + abs(r1 - r2) == 1:
            return "neighbour"
    if mirror(gold) == pred:
        return "mirror"
    return "multi"


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="AngelinaDataset")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.15)
    a = ap.parse_args()
    model = YOLO(a.weights)
    subsets = {"uploaded_test": "uploaded/test.txt", "books_val": "books/val.txt",
               "handwritten_val": "handwritten/val.txt"}
    res = {}
    for name, lst in subsets.items():
        base = os.path.join(a.root, os.path.dirname(lst))
        ev, dots_lost, dots_gained, pairs_c = Counter(), Counter(), Counter(), Counter()
        per_img = []
        n_gold = n_img = 0
        for line in open(os.path.join(a.root, lst)):
            rel = line.strip().replace("\\", "/")
            if not rel:
                continue
            img = os.path.join(base, rel)
            csv = img[: -len(".jpg")] + ".csv"
            if not (os.path.exists(img) and os.path.exists(csv)):
                ev["missing_file"] += 1
                continue
            gold = []
            for row in open(csv):
                p = row.strip().split(";")
                if len(p) == 5:
                    gold.append(([float(x) for x in p[:4]], int(p[4])))
            r = model.predict(img, conf=a.conf, verbose=False)[0]
            pb = r.boxes.xyxyn.cpu().numpy()
            pc = [bits(model.names[int(c)]) for c in r.boxes.cls.cpu().numpy()]
            pairs = sorted(((iou(g[0], pb[j]), i, j) for i, g in enumerate(gold) for j in range(len(pb))
                            if iou(g[0], pb[j]) >= 0.3), reverse=True)
            ug, up = set(), set()
            for _, i, j in pairs:
                if i in ug or j in up:
                    continue
                ug.add(i)
                up.add(j)
                gm, pm = gold[i][1], pc[j]
                ev[classify(gm, pm)] += 1
                if gm != pm:
                    pairs_c[(chr(0x2800 + gm), chr(0x2800 + pm))] += 1
                for d in range(6):
                    if gm >> d & 1 and not pm >> d & 1:
                        dots_lost[d + 1] += 1
                    if pm >> d & 1 and not gm >> d & 1:
                        dots_gained[d + 1] += 1
            gc = np.array([[(g[0][0] + g[0][2]) / 2, (g[0][1] + g[0][3]) / 2] for g in gold]) if gold else None
            for j in range(len(pb)):
                if j in up:
                    continue
                ev["insert"] += 1
                if gc is not None:
                    c = np.array([(pb[j][0] + pb[j][2]) / 2, (pb[j][1] + pb[j][3]) / 2])
                    k = int(np.argmin(((gc - c) ** 2).sum(1)))
                    if gold[k][1] == pc[j]:
                        ev["insert_repeat"] += 1
            ev["delete"] += len(gold) - len(ug)
            # deleted gold cells whose row (same y band) has no detection at all: a missed line
            rows_hit = [((g[0][1] + g[0][3]) / 2, i in ug) for i, g in enumerate(gold)]
            h = np.median([g[0][3] - g[0][1] for g in gold]) if gold else 0
            missed_line = 0
            for y, hit in rows_hit:
                if not hit and not any(h2 for y2, h2 in rows_hit if abs(y2 - y) < h / 2):
                    missed_line += 1
            ev["delete_whole_line"] += missed_line
            per_img.append((len(gold), len(gold) - len(ug)))
            n_gold += len(gold)
            n_img += 1
        rates = {k: round(100 * v / max(1, n_gold), 2) for k, v in ev.items()}
        errs = {k: ev[k] for k in ("omit", "add", "neighbour", "mirror", "multi", "insert", "delete")}
        tot = sum(errs.values())
        res[name] = {"images": n_img, "gold_cells": n_gold, "counts": dict(ev), "per_100_gold_cells": rates,
                     "error_mix": {k: round(v / max(1, tot), 3) for k, v in errs.items()},
                     "top_substitutions": [[g, p_, n] for (g, p_), n in pairs_c.most_common(10)],
                     "delete_rate_by_image": sorted(round(d / max(1, n), 3) for n, d in per_img),
                     "dots_lost": dict(sorted(dots_lost.items())), "dots_gained": dict(sorted(dots_gained.items()))}
        print(name, json.dumps(res[name]), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"subsets": res, "args": vars(a)}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

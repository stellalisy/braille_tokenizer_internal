"""Score the rule-based back-translator on the same clean and noisy test sets
that scripts/train_backtranslate.py uses.

    python scripts/eval_liblouis.py --data data --out runs/liblouis
"""
import argparse
import json
import os
import random

from rapidfuzz.distance import Levenshtein

from brltok import noise, translate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/liblouis")
    ap.add_argument("--eval-noise", default="0.01,0.02,0.05,0.1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    test = [json.loads(l) for l in open(f"{a.data}/test.jsonl", encoding="utf8")]
    refs = [r["text"] for r in test]
    norm = lambda s: " ".join(s.split())
    res = {}
    for p in [0.0] + [float(x) for x in a.eval_noise.split(",") if x]:
        rng = random.Random(4242)   # same seed as the model evaluation
        hyps = [norm(translate.back(noise.flip_dots(r["g2"], p, rng))) for r in test]
        edits = sum(Levenshtein.distance(h, norm(r)) for h, r in zip(hyps, refs))
        res[str(p) if p else "0"] = {
            "cer": round(edits / sum(len(norm(r)) for r in refs), 4),
            "exact": round(sum(h == norm(r) for h, r in zip(hyps, refs)) / len(refs), 4)}
    out = {"liblouis": translate.version(), "test": res}
    json.dump(out, open(f"{a.out}/result.json", "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()

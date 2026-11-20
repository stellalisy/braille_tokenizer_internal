"""Rescore saved CommonsenseQA outputs with brltok.scoring.choice_correct (CPU only).

Choices come from the clean print records, so a pipeline answer that copies a noise-corrupted
choice text does not match; a letter answer still does. Writes <run>/csqa_choice.json next to each
result.json and prints a table of released EM vs choice accuracy.

    python scripts/rescore_csqa.py exp/phaseA/qwen3-1.7b exp/phaseA/llama3.1-8b-instruct ...
"""
import glob
import json
import os
import sys

from brltok.scoring import choice_correct


def main(runs, inputs="data/phaseA/commonsenseqa"):
    base = [json.loads(l) for l in open(f"{inputs}/print.jsonl", encoding="utf8")]
    for run in runs:
        files = sorted(glob.glob(f"{run}/commonsenseqa_*_details.jsonl") +
                       glob.glob(f"{run}/*/commonsenseqa_*_details.jsonl"))
        out = {}
        for f in files:
            cond = os.path.basename(f)[len("commonsenseqa_"):-len("_details.jsonl")]
            rows = [json.loads(l) for l in open(f, encoding="utf8")]
            assert len(rows) <= len(base)
            acc = [choice_correct(r["response"], b["choices"], b["golden_answers"])
                   for r, b in zip(rows, base)]
            em = [r["metrics"]["em"] for r in rows]
            out[cond] = {"n": len(rows), "released_em": round(100 * sum(em) / len(em), 2),
                         "choice_acc": round(100 * sum(acc) / len(acc), 2)}
        if out:
            json.dump(out, open(f"{run}/csqa_choice.json", "w"), indent=1)
            print(f"\n{run}")
            for c, v in sorted(out.items()):
                print(f"  {c:16s} released EM {v['released_em']:6.2f}   choice acc {v['choice_acc']:6.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])

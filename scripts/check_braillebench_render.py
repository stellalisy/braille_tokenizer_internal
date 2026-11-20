"""Check that our liblouis build renders BrailleBench questions the same way the release does.

For each dataset and grade, re-translate the English `question` with liblouis and compare to the
released `braille_question` in Unicode (cell for cell, ASCII spaces normalized). Also reports how
often liblouis back-translation of the released braille recovers the English question exactly
(the pipeline baseline's input quality).

    python scripts/check_braillebench_render.py --bb $BRL_SCRATCH/ext/braillebench --out exp/phaseA/render_check
"""
import argparse
import json
import os
import subprocess

from brltok import translate
from brltok.cells import normalize


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf8")]


def norm_ws(s):
    return " ".join(s.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bb", required=True, help="BrailleBench checkout")
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default="gsm8k,commonsenseqa")
    ap.add_argument("--show", type=int, default=5, help="mismatch examples to keep per file")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for ds in a.datasets.split(","):
        for g in (1, 2):
            recs = load(f"{a.bb}/data/braille/{ds}/grade{g}/unicode.jsonl")
            same = back_exact = 0
            examples = []
            for r in recs:
                src = r.get("clean_question") or r["question"]
                ours = norm_ws(translate.forward_string(src, g))
                theirs = norm_ws(normalize(r["braille_question"]))
                if ours == theirs:
                    same += 1
                elif len(examples) < a.show:
                    examples.append({"question": src, "ours": ours, "theirs": theirs})
                if norm_ws(translate.back(theirs, g)) == norm_ws(src):
                    back_exact += 1
            n = len(recs)
            res[f"{ds}/grade{g}"] = {"n": n, "render_match": round(same / n, 4),
                                    "back_exact": round(back_exact / n, 4), "mismatches": examples}
            print(ds, g, n, res[f"{ds}/grade{g}"]["render_match"], res[f"{ds}/grade{g}"]["back_exact"], flush=True)
    bb_commit = subprocess.check_output(["git", "-C", a.bb, "rev-parse", "HEAD"], text=True).strip()
    out = {"liblouis": translate.version(), "braillebench_commit": bb_commit, "result": res}
    json.dump(out, open(f"{a.out}/result.json", "w"), indent=1, ensure_ascii=False)
    json.dump({"command": "python scripts/check_braillebench_render.py --bb " + a.bb + " --out " + a.out,
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "liblouis": translate.version(), "braillebench_commit": bb_commit},
              open(f"{a.out}/config.json", "w"), indent=1)


if __name__ == "__main__":
    main()

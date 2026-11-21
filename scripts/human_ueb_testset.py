"""Phase A step 6: a human-authored English braille test set, and liblouis scored on it.

Source: the examples from "The Rules of Unified English Braille" (ICEB, 2013 edition, with
2019 updates) as transcribed into liblouis's test suite (tests/braille-specs/en-ueb.yaml, Grade 2),
plus the Grade 1 harness from Torch Trust (en-ueb-g1_harness.yaml). The braille was written by
the rulebook's authors, not by liblouis. liblouis is developed against these examples, so the rule
baseline is in-sample here; the set is a check on learned models, not a fair test of liblouis.

Entries with typeform (emphasis) or mode options are skipped, since our inputs carry no emphasis.

    python scripts/human_ueb_testset.py --specs $BRL_SCRATCH/liblouis-3.32.0/tests/braille-specs \
        --out data/human/ueb_rulebook.jsonl --result exp/phaseA/human_ueb/result.json
"""
import argparse
import json
import os
import subprocess
import sys

import yaml
from rapidfuzz.distance import Levenshtein

from brltok.cells import from_ascii, normalize
from brltok.translate import back, forward_string, version

SOURCES = [("en-ueb.yaml", 2, "ICEB Rules of UEB examples via liblouis tests"),
           ("en-ueb-g1_harness.yaml", 1, "Torch Trust Grade 1 harness via liblouis tests")]
LOWER_BRF = str.maketrans({"`": "@", "{": "[", "|": "\\", "}": "]", "~": "^"})


def to_unicode(braille, display):
    if "unicode" in display:
        return normalize(braille)
    return normalize(from_ascii(braille.translate(LOWER_BRF)))


def load(path, grade, source):
    out, skipped = [], 0
    for doc in yaml.safe_load_all(open(path, encoding="utf8")):
        if not doc or "tests" not in doc:
            continue
        display = doc.get("display", "")
        for i, t in enumerate(doc["tests"]):
            opts = t[2] if len(t) > 2 and isinstance(t[2], dict) else {}
            if set(opts) - {"xfail"} or not isinstance(t[0], str) or not isinstance(t[1], str):
                skipped += 1
                continue
            try:
                br = to_unicode(t[1], display)
            except ValueError:
                skipped += 1
                continue
            out.append({"id": f"{os.path.basename(path)}:{i}", "source": source, "grade": grade,
                        "print": t[0], "braille": br, "liblouis_xfail": bool(opts.get("xfail"))})
    return out, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--result", required=True)
    a = ap.parse_args()
    recs, skipped = [], {}
    for fname, grade, source in SOURCES:
        r, s = load(os.path.join(a.specs, fname), grade, source)
        recs += r
        skipped[fname] = s
    res = {}
    for grade in (1, 2):
        rs = [r for r in recs if r["grade"] == grade]
        fwd = sum(normalize(forward_string(r["print"], grade)) == r["braille"] for r in rs)
        em, edits, chars = 0, 0, 0
        for r in rs:
            hyp = back(r["braille"], grade)
            r["liblouis_back"] = hyp
            em += hyp == r["print"]
            edits += Levenshtein.distance(hyp, r["print"])
            chars += len(r["print"])
        res[f"g{grade}"] = {"n": len(rs), "n_xfail": sum(r["liblouis_xfail"] for r in rs),
                            "mean_print_chars": round(chars / len(rs), 1),
                            "forward_match": round(100 * fwd / len(rs), 2),
                            "back_exact": round(100 * em / len(rs), 2),
                            "back_cer": round(100 * edits / chars, 2)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.makedirs(os.path.dirname(a.result), exist_ok=True)
    json.dump({"result": res, "skipped": skipped, "command": "python " + " ".join(sys.argv),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "liblouis": version()}, open(a.result, "w"), indent=1)
    print(json.dumps(res, indent=1), skipped)


if __name__ == "__main__":
    main()

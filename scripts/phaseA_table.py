"""Phase A tables (markdown) from exp/phaseA/<run>/result.json.

Gap closed by the pipeline = (pipeline - stock braille) / (print - stock braille), with stock
braille = Braille ASCII at the same grade and noise rate, as in the outline's headline metric.

    python scripts/phaseA_table.py exp/phaseA/qwen3-1.7b exp/phaseA/qwen3-1.7b-thinking ...
"""
import json
import os
import sys

RATES = ["p0", "p2", "p5", "p10"]
DS = [("gsm8k", "GSM8K"), ("commonsenseqa", "CSQA")]


def gc(pipe, br, pr):
    return None if pr == br else 100 * (pipe - br) / (pr - br)


def f(x):
    return "" if x is None else f"{x:.1f}"


def load(run):
    """A run's result.json, or the merged result.json files of its shard subfolders."""
    if os.path.exists(f"{run}/result.json"):
        return json.load(open(f"{run}/result.json"))
    merged = {}
    for sub in sorted(os.listdir(run)):
        p = f"{run}/{sub}/result.json"
        if os.path.exists(p):
            for k, v in json.load(open(p)).items():
                assert k not in merged, f"{k} in two shards of {run}"
                merged[k] = v
    return merged


def with_choice(run, r):
    """Use choice accuracy as the CommonsenseQA primary metric (scripts/rescore_csqa.py output)."""
    for p in [f"{run}/csqa_choice.json"]:
        if os.path.exists(p):
            for cond, v in json.load(open(p)).items():
                k = f"commonsenseqa/{cond}"
                if k in r:
                    r[k] = dict(r[k], released_em=v["released_em"], primary=v["choice_acc"])
    return r


def main(runs):
    res = {os.path.basename(r.rstrip("/")): with_choice(r, load(r)) for r in runs}
    print("### Clean input\n")
    print("| Run | Dataset | Print | Print (preprocessed) | ASCII G1 | ASCII G2 | Unicode G1 | Unicode G2 | Pipeline G1 | Pipeline G2 | Gap closed G1 | Gap closed G2 |")
    print("|" + "---|" * 12)
    for name, r in res.items():
        for ds, label in DS:
            g = lambda c: r[f"{ds}/{c}"]["primary"]
            pr = g("print")
            row = [name, label, f(pr), f(g("print_clean")), f(g("g1_ascii_p0")), f(g("g2_ascii_p0")),
                   f(g("g1_unicode_p0")), f(g("g2_unicode_p0")), f(g("pipe_g1_p0")), f(g("pipe_g2_p0")),
                   f(gc(g("pipe_g1_p0"), g("g1_ascii_p0"), pr)), f(gc(g("pipe_g2_p0"), g("g2_ascii_p0"), pr))]
            print("| " + " | ".join(row) + " |")
    print("\n### Dot noise (flip rate 0, 2, 5, 10%)\n")
    print("| Run | Dataset | Grade | Input | 0% | 2% | 5% | 10% |")
    print("|---|---|---|---|---|---|---|---|")
    for name, r in res.items():
        for ds, label in DS:
            for gr in ("g1", "g2"):
                for kind, lab in (("ascii", "ASCII"), ("unicode", "Unicode"), ("pipe", "Pipeline")):
                    key = (lambda p: f"{ds}/pipe_{gr}_{p}") if kind == "pipe" else (lambda p: f"{ds}/{gr}_{kind}_{p}")
                    vals = [f(r[key(p)]["primary"]) for p in RATES]
                    print(f"| {name} | {label} | {gr.upper()} | {lab} | " + " | ".join(vals) + " |")
    print("\n### Truncated generations (of n), clean input\n")
    print("| Run | Dataset | Print | ASCII G1 | ASCII G2 | Unicode G1 | Unicode G2 | Pipeline G2 | n |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, r in res.items():
        for ds, label in DS:
            t = lambda c: str(r[f"{ds}/{c}"]["truncated"])
            print(f"| {name} | {label} | " + " | ".join(t(c) for c in
                  ("print", "g1_ascii_p0", "g2_ascii_p0", "g1_unicode_p0", "g2_unicode_p0", "pipe_g2_p0"))
                  + f" | {r[f'{ds}/print']['n']} |")


if __name__ == "__main__":
    main(sys.argv[1:])

"""Collect runs/*/result.json into one markdown table.

    python scripts/summarize.py --runs runs
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    a = ap.parse_args()
    rows, levels = [], ["0", "0.01", "0.02", "0.05", "0.1"]
    for path in sorted(glob.glob(f"{a.runs}/*/result.json")):
        r = json.load(open(path))
        name = os.path.basename(os.path.dirname(path))
        t = r["test"]
        if "args" in r:
            meta = [str(r["vocab"]), f"{r['params'] / 1e6:.2f}M", str(r["args"]["train_noise"] or "none"),
                    f"{r['train_seconds'] / 60:.0f} min"]
        else:
            meta = ["", "", "", ""]
        rows.append([name] + meta + [f"{100 * t['0']['exact']:.1f}"] +
                    [f"{100 * t[l]['cer']:.2f}" if l in t else "" for l in levels])
    head = ["run", "vocab", "params", "train noise", "train time", "exact %, clean"] + \
           [f"CER % at {float(l):.0%} flips" if l != "0" else "CER %, clean" for l in levels]
    print("| " + " | ".join(head) + " |")
    print("|" + " --- |" * len(head))
    for r in rows:
        print("| " + " | ".join(r) + " |")


if __name__ == "__main__":
    main()

"""Acceptance tests of design note revision 2 for one Stage A run, before the language model.

Runs, in order, as subprocesses (each writes its own result file under the run directory):
  1. copy probe (scripts/analysis/vocab_copy_diag.py), excluding any string in the non-word training data
  2. test-set words with dot flips at 0, 2, 5, 10% (scripts/vocab_eval_words.py)
  3. test-set words with the writing and ocr noise profiles at 5 and 10%
  4. real photos (scripts/realimg/realimg_score.py; the other grade's slot uses the Stage A run)

    python scripts/vocab_eval_suite.py --run exp/vocab/v2/g1_local_nonw_mix
"""
import argparse
import json
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--exclude", default=None, help="non-word jsonl (default data/vocab/nonwords_g{g}.jsonl)")
    a = ap.parse_args()
    g = json.load(open(f"{a.run}/train_log.json"))["args"]["grade"]
    exclude = a.exclude or f"data/vocab/nonwords_g{g}.jsonl"
    py = sys.executable
    other = "exp/vocab/g2_a_s0" if g == 1 else "exp/vocab/g1_a_s0"
    runs = {1: a.run, 2: other} if g == 1 else {1: other, 2: a.run}
    steps = [
        [py, "scripts/analysis/vocab_copy_diag.py", "--run", a.run, "--exclude", exclude, "--tag", "_v2"],
        [py, "scripts/vocab_eval_words.py", "--run", a.run],
        [py, "scripts/vocab_eval_words.py", "--run", a.run, "--noise-profile", "writing", "--rates", "5,10"],
        [py, "scripts/vocab_eval_words.py", "--run", a.run, "--noise-profile", "ocr", "--rates", "5,10"],
        [py, "scripts/realimg/realimg_score.py", "--obr", "data/realimg/obr_v1.jsonl", "--run1", runs[1],
         "--run2", runs[2], "--out", f"{a.run}/realimg_score.json"],
    ]
    for cmd in steps:
        print("##", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    print("SUITE DONE", flush=True)


if __name__ == "__main__":
    main()

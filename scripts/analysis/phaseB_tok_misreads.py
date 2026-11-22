"""Misreads by braille tokenizer: thinking-trace option restatement and question transcription.

For each grade and tokenizer (E2 init, seed 0) on CommonsenseQA:
  acc_think       accuracy with thinking on
  opt_misread     share of options (among items whose print trace names all five) that the braille
                  trace never names verbatim
  item_misread    share of those items with at least one option not named
  q_exact         question transcribed exactly ("Braille: ... Print:", greedy, to newline)
  q_le1           question transcribed with at most one word error (runaway outputs count as errors)
Run from the repo root: python scripts/analysis/phaseB_tok_misreads.py
"""
import json
import os
import re
import sys

sys.path.insert(0, ".")
from brltok.scoring import choice_correct

T = "exp/phaseB/train"


def strip(c):
    return re.sub(r"^[A-E]\)\s*", "", c)


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def think(d):
    m = re.search(r"<think>(.*?)</think>", d["response"], re.S)
    return norm(m.group(1) if m else d["response"])


def found(t, phrase):
    return re.search(r"\b" + re.escape(norm(phrase).strip()) + r"\b", t) is not None


for g in (1, 2):
    items = {r["id"]: r for r in map(json.loads, open(f"data/phaseA/commonsenseqa/g{g}_unicode_p0.jsonl"))}
    pp = f"{T}/g{g}_t3_e2_s0/eval_thinking_print_all/commonsenseqa_g{g}_p0_details.jsonl"
    if not os.path.exists(pp):
        print(f"G{g}: no print-template thinking run yet")
        continue
    P = {d["id"]: d for d in map(json.loads, open(pp))}
    sel = [i for i in P if all(found(think(P[i]), strip(o)) for o in items[i]["choices"])]
    pacc = 100 * sum(P[i]["metrics"]["choice"] for i in P) / len(P)
    print(f"\nGrade {g}: print (braille template, thinking) {pacc:.1f}; items whose print trace names all five: {len(sel)}")
    print("  tok  acc_think  opt_misread  item_misread  acc_drop_on_misread_items  q_exact  q_le1")
    for t in ("t1", "t2", "t3"):
        run = f"{T}/g{g}_{t}_e2_s0"
        bp = f"{run}/eval_thinking/commonsenseqa_g{g}_p0_details.jsonl"
        row = [f"  {t}"]
        if os.path.exists(bp):
            B = {d["id"]: d for d in map(json.loads, open(bp))}
            acc = 100 * sum(B[i]["metrics"]["choice"] for i in B) / len(B)
            n_opt = miss_opt = miss_item = 0
            mis_items = []
            for i in sel:
                tb = think(B[i])
                m = sum(not found(tb, strip(o)) for o in items[i]["choices"])
                n_opt += 5
                miss_opt += m
                if m:
                    miss_item += 1
                    mis_items.append(i)
            drop = (100 * sum(B[i]["metrics"]["choice"] - P[i]["metrics"]["choice"] for i in mis_items) / max(1, len(mis_items)))
            row += [f"{acc:9.1f}", f"{100 * miss_opt / n_opt:11.1f}", f"{100 * miss_item / len(sel):12.1f}", f"{drop:25.1f}"]
        else:
            row += ["        -"] * 4
        tp = f"{run}/transcribe/commonsenseqa_details.jsonl"
        if os.path.exists(tp):
            q = [r for r in map(json.loads, open(tp)) if r["part"] == "question"]
            row += [f"{100 * sum(r['exact'] for r in q) / len(q):8.1f}", f"{100 * sum(r['errors'] <= 1 for r in q) / len(q):6.1f}"]
        else:
            row += ["       -", "     -"]
        print(" ".join(row))

"""Which options get misread? Grade 1 T3 E2 seed 0, thinking traces, CommonsenseQA.

An option is misread when the print-template trace names it verbatim and the braille trace does not.
Compares misread and correctly read options by training frequency of their rarest word, number of print
tokens per word, and number of braille units per word. Also reports the tokenizer vocabularies.
Run from the repo root with the training venv: python scripts/analysis/phaseB_misread_words.py
"""
import json
import re
import sys
from collections import Counter

sys.path.insert(0, ".")
from transformers import AutoTokenizer

from brltok.tokenizers import RuleUnitTokenizer

R = "exp/phaseB/train/g1_t3_e2_s0"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
items = {r["id"]: r for r in map(json.loads, open("data/phaseA/commonsenseqa/g1_unicode_p0.jsonl"))}
B = {d["id"]: d for d in map(json.loads, open(f"{R}/eval_thinking/commonsenseqa_g1_p0_details.jsonl"))}
P = {d["id"]: d for d in map(json.loads, open(f"{R}/eval_thinking_print_all/commonsenseqa_g1_p0_details.jsonl"))}

for g in (1, 2):
    for t in ("t2", "t3"):
        try:
            d = json.load(open(f"data/phaseB/tok/g{g}_{t}.json"))
            v = d.get("vocab") or d.get("units") or d.get("tokens")
            print(f"g{g}_{t}: vocab {len(v) if v else '?'}; keys {list(d)[:6]}")
        except Exception as e:
            print(f"g{g}_{t}: {e}")
btok = RuleUnitTokenizer.load("data/phaseB/tok/g1_t3.json")

wc = Counter()
for r in map(json.loads, open("data/phaseB/train.jsonl")):
    wc.update(re.findall(r"[a-z]+", r["text"].lower()))


def strip(c):
    return re.sub(r"^[A-E]\)\s*", "", c)


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def think(d):
    m = re.search(r"<think>(.*?)</think>", d["response"], re.S)
    return norm(m.group(1) if m else d["response"])


def found(t, phrase):
    return re.search(r"\b" + re.escape(norm(phrase).strip()) + r"\b", t) is not None


rows = []
for i in B:
    tp, tb = think(P[i]), think(B[i])
    for o, bo in zip(items[i]["choices"], items[i]["braille_choices"]):
        o = strip(o)
        if not found(tp, o):
            continue
        ws = re.findall(r"[a-z]+", o.lower())
        if not ws:
            continue
        bwords = bo.replace("⠀", " ").split()[1:]   # drop the "A)" label
        units = [len(btok.encode(w)) if hasattr(btok, "encode") else 0 for w in bwords]
        rows.append({"misread": not found(tb, o), "minfreq": min(wc[w] for w in ws),
                     "ptok": max(len(tok(" " + w, add_special_tokens=False)["input_ids"]) for w in ws),
                     "units": max(units) if units else 0, "text": o})

n = len(rows)
m = [r for r in rows if r["misread"]]
print(f"\noptions {n}, misread {len(m)} ({100 * len(m) / n:.1f}%)")


def rate(sel):
    return f"{100 * sum(r['misread'] for r in sel) / max(1, len(sel)):5.1f}% of {len(sel)}"


print("misread rate by training frequency of rarest word")
for lo, hi in ((0, 0), (1, 9), (10, 99), (100, 999), (1000, 10 ** 12)):
    print(f"  {lo:>5}-{hi:<12} {rate([r for r in rows if lo <= r['minfreq'] <= hi])}")
print("misread rate by print tokens in longest word")
for k in (1, 2, 3, 4):
    print(f"  {k}{'+' if k == 4 else ' '} {rate([r for r in rows if (r['ptok'] >= 4 if k == 4 else r['ptok'] == k)])}")
print("misread rate by braille units in longest word")
for lo, hi in ((1, 3), (4, 6), (7, 9), (10, 99)):
    print(f"  {lo}-{hi} {rate([r for r in rows if lo <= r['units'] <= hi])}")
print("\nsample misread options with their braille trace counterpart unknown; texts:")
print(", ".join(r["text"] for r in m[:60]))

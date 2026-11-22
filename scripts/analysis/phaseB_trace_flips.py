"""Noise floor for item flips, and whether misreading the options predicts a lost item."""
import json
import re
import sys
from collections import Counter

sys.path.insert(0, ".")
from brltok.scoring import choice_correct

R = "exp/phaseB/train/g1_t3_e2_s0"
items = {r["id"]: r for r in map(json.loads, open("data/phaseA/commonsenseqa/g1_unicode_p0.jsonl"))}


def load(p):
    out = {}
    for d in map(json.loads, open(p)):
        it = items[d["id"]]
        d["c"] = d["metrics"].get("choice", choice_correct(d["response"], it["choices"], it["golden_answers"]))
        out[d["id"]] = d
    return out


B = load(f"{R}/eval_thinking/commonsenseqa_g1_p0_details.jsonl")
P = load(f"{R}/eval_thinking_print_all/commonsenseqa_g1_p0_details.jsonl")
PA = load("exp/phaseA/qwen3-1.7b-thinking-32k/print_pipe/commonsenseqa_print_details.jsonl")
PIPE = load("exp/phaseA/qwen3-1.7b-thinking-32k/print_pipe/commonsenseqa_pipe_g1_p0_details.jsonl")
ids = sorted(B)


def flips(X, Y, name):
    t = Counter((int(X[i]["c"]), int(Y[i]["c"])) for i in ids)
    acc = lambda D: 100 * sum(D[i]["c"] for i in ids) / len(ids)
    print(f"{name:42s} acc {acc(X):.1f} -> {acc(Y):.1f}  lost {t[(1, 0)]:4d} won {t[(0, 1)]:4d}  changed {t[(1, 0)] + t[(0, 1)]}")


print("thinking on, CommonsenseQA, item flips")
flips(PA, P, "print EN-EN -> print in braille template")
flips(PA, PIPE, "print EN-EN -> pipeline G1")
flips(P, B, "print in braille template -> braille")


def strip(c):
    return re.sub(r"^[A-E]\)\s*", "", c)


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def think(d):
    m = re.search(r"<think>(.*?)</think>", d["response"], re.S)
    return norm(m.group(1) if m else d["response"])


def found(t, phrase):
    return re.search(r"\b" + re.escape(norm(phrase).strip()) + r"\b", t) is not None


# Restrict to items where the print trace restates all five options, so absence in braille is a misreading.
sel = [i for i in ids if all(found(think(P[i]), strip(o)) for o in items[i]["choices"])]
print(f"\nitems whose print trace names all five options: {len(sel)}")
groups = {"braille names all five": [], "braille misses the gold option": [], "braille misses only other options": []}
for i in sel:
    t = think(B[i])
    gold = items[i]["golden_answers"][0]
    miss = [strip(o) for o in items[i]["choices"] if not found(t, strip(o))]
    if not miss:
        groups["braille names all five"].append(i)
    elif any(norm(m).strip() == norm(gold).strip() for m in miss):
        groups["braille misses the gold option"].append(i)
    else:
        groups["braille misses only other options"].append(i)
for k, g in groups.items():
    if g:
        pa = 100 * sum(P[i]["c"] for i in g) / len(g)
        ba = 100 * sum(B[i]["c"] for i in g) / len(g)
        print(f"  {k:36s} n={len(g):4d}  print {pa:.1f}  braille {ba:.1f}  diff {ba - pa:+.1f}")

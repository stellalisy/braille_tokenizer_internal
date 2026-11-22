"""Compare thinking traces, braille vs print (same template), Grade 1 T3 E2 seed 0, CommonsenseQA."""
import json
import random
import re
import sys
from collections import Counter

sys.path.insert(0, ".")
R = "exp/phaseB/train/g1_t3_e2_s0"
items = {r["id"]: r for r in map(json.loads, open("data/phaseA/commonsenseqa/g1_unicode_p0.jsonl"))}
B = {d["id"]: d for d in map(json.loads, open(f"{R}/eval_thinking/commonsenseqa_g1_p0_details.jsonl"))}
P = {d["id"]: d for d in map(json.loads, open(f"{R}/eval_thinking_print_all/commonsenseqa_g1_p0_details.jsonl"))}
NT = {d["id"]: d for d in map(json.loads, open(f"{R}/eval/commonsenseqa_g1_p0_details.jsonl"))}
ids = sorted(B)


def strip(c):
    return re.sub(r"^[A-E]\)\s*", "", c)


def think(d):
    m = re.search(r"<think>(.*?)</think>", d["response"], re.S)
    return (m.group(1) if m else d["response"]).lower()


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def mentioned(text, phrase):
    return re.search(r"\b" + re.escape(norm(phrase).strip()) + r"\b", norm(text)) is not None


c = lambda D, i: D[i]["metrics"]["choice"]
print("accuracy: braille think %.1f  print think %.1f  braille no-think %.1f" % tuple(
    100 * sum(c(D, i) for i in ids) / len(ids) for D in (B, P, NT)))
print("print x braille (thinking):", dict(Counter((int(c(P, i)), int(c(B, i))) for i in ids)))
print("braille no-think x think:", dict(Counter((int(c(NT, i)), int(c(B, i))) for i in ids)))

# how often each option text appears in the trace
for name, D in (("braille", B), ("print", P)):
    tot = hit = 0
    allopt = 0
    for i in ids:
        t = think(D[i])
        opts = [strip(x) for x in items[i]["choices"]]
        h = [mentioned(t, o) for o in opts]
        tot += len(h)
        hit += sum(h)
        allopt += all(h)
    print(f"{name}: option text appears in trace {100 * hit / tot:.1f}% of options; all five in {100 * allopt / len(ids):.1f}% of items")

lost = [i for i in ids if c(P, i) and not c(B, i)]
won = [i for i in ids if c(B, i) and not c(P, i)]
print("lost", len(lost), "won", len(won))
for name, sel in (("lost", lost), ("both right", [i for i in ids if c(P, i) and c(B, i)])):
    tot = hit = 0
    for i in sel:
        t = think(B[i])
        for o in (strip(x) for x in items[i]["choices"]):
            tot += 1
            hit += mentioned(t, o)
    print(f"braille trace option mention, {name}: {100 * hit / tot:.1f}%")

# question restatement: share of question content words found in trace
STOP = set("a an the of to in on at for is are was were be you your would what where who why how which it its that this with and or do does did can could if when they he she his her their them".split())
def qcov(i, D):
    ws = [w for w in norm(items[i]["question"]).split() if w not in STOP and len(w) > 2]
    t = norm(think(D[i]))
    return sum(re.search(r"\b" + w + r"\b", t) is not None for w in ws) / max(1, len(ws))
for name, sel in (("lost", lost), ("all", ids)):
    print(f"question content words in trace, {name}: braille {100 * sum(qcov(i, B) for i in sel) / len(sel):.1f}% print {100 * sum(qcov(i, P) for i in sel) / len(sel):.1f}%")

random.seed(0)
out = open("/tmp/trace_examples.txt", "w")
for i in random.sample(lost, min(25, len(lost))):
    it = items[i]
    out.write("=" * 100 + f"\n{i}  Q: {it['question']}\n   options: {[strip(x) for x in it['choices']]}  gold: {it['golden_answers'][0]}\n")
    out.write(f"--- BRAILLE (wrong): {B[i]['predicted']}\n{think(B[i])[:1500]}\n")
    out.write(f"--- PRINT (right): {P[i]['predicted']}\n{think(P[i])[:800]}\n")
print("examples -> /tmp/trace_examples.txt")

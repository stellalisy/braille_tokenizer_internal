"""Phase B training data: print segments from several corpora, translated to UEB Grade 1 and 2.

Sources (training splits only): Gutenberg sentences already in data/train.jsonl, WikiText-103 train,
Dolly 15k, GSM8K train (questions whole, answers by sentence with calculator marks removed), and
CommonsenseQA train questions. Each record keeps the braille and the liblouis rule units for both
grades, and whether braille and print have the same word count (needed by the distillation loss).

Check B1: no GSM8K test or CommonsenseQA validation question appears in the data, exactly or
after normalization (lowercase, punctuation and whitespace removed).

    python scripts/phaseB_build_data.py --out data/phaseB --workers 32
"""
import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from multiprocessing import Pool

import nltk

from brltok.align import units_from
from brltok.translate import forward, version

CAPS = {"wikitext": 150_000, "dolly": 60_000}
SEED = 0
N_DEV = 2000


def norm(s):
    return re.sub(r"[\W_]+", "", s.lower())


def sentences(text):
    return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]


def keep(s):
    n = len(s.split())
    return 3 <= n <= 80 and len(s) <= 500


def hf(repo, *args, **kw):
    from datasets import load_dataset
    return load_dataset(repo, *args, **kw)


def collect():
    segs = []   # (source, text)
    for l in open("data/train.jsonl", encoding="utf8"):
        segs.append(("gutenberg", json.loads(l)["text"]))
    wt = hf("Salesforce/wikitext", name="wikitext-103-raw-v1", split="train")
    rng = random.Random(SEED)
    wsents = []
    for t in wt["text"]:
        t = t.strip()
        if not t or t.startswith("="):
            continue
        t = re.sub(r" @(.)@ ", r"\1", t)                      # "1 @,@ 000" -> "1,000"
        t = re.sub(r" ([.,;:!?')\]])", r"\1", t).replace("( ", "(").replace(" n't", "n't").replace(" 's", "'s")
        wsents += [s for s in sentences(t) if keep(s)]
        if len(wsents) > 4 * CAPS["wikitext"]:
            break
    segs += [("wikitext", s) for s in rng.sample(wsents, CAPS["wikitext"])]
    dsents = []
    for r in hf("databricks/databricks-dolly-15k", split="train"):
        for f in ("instruction", "context", "response"):
            dsents += [s for s in sentences(r[f]) if keep(s)]
    segs += [("dolly", s) for s in rng.sample(dsents, min(CAPS["dolly"], len(dsents)))]
    for r in hf("openai/gsm8k", "main", split="train"):
        segs.append(("gsm8k_q", r["question"].strip()))
        ans = re.sub(r"<<[^>]*>>", "", r["answer"])
        ans, _, final = ans.partition("####")
        for line in ans.splitlines():
            segs += [("gsm8k_a", s) for s in sentences(line) if len(s.split()) >= 2]
        segs.append(("gsm8k_a", f"The answer is {final.strip()}."))
    for r in hf("tau/commonsense_qa", split="train"):
        segs.append(("csqa_q", r["question"].strip()))
    seen, out = set(), []
    for src, t in segs:
        t = " ".join(t.split())
        if t and t not in seen:
            seen.add(t)
            out.append((src, t))
    return out


def translate(item):
    src, text = item
    rec = {"source": src, "text": text}
    for g in (1, 2):
        try:
            tr = forward(text, g)
        except Exception as e:      # noqa: BLE001  liblouis errors are rare; record and skip
            return {"source": src, "text": text, "error": str(e)}
        if any(0x2840 <= ord(ch) <= 0x28FF for ch in tr.braille):
            # liblouis writes characters it cannot translate as 8-dot cells; 6-dot tokenizers skip them
            return {"source": src, "text": text, "error": "8-dot cell"}
        rec[f"g{g}"] = tr.braille
        rec[f"units_g{g}"] = [[u.cells, u.text] for u in units_from(tr)]
        rec[f"aligned_g{g}"] = len(tr.braille.split()) == len(text.split())
    return rec


def contamination(recs):
    tests = [r["question"] for r in hf("openai/gsm8k", "main", split="test")]
    tests += [r["question"] for r in hf("tau/commonsense_qa", split="validation")]
    exact = {t.strip() for t in tests}
    normed = {norm(t) for t in tests}
    hits_exact = sum(r["text"] in exact for r in recs)
    hits_norm = sum(norm(r["text"]) in normed for r in recs)
    return {"n_test_questions": len(tests), "exact": hits_exact, "normalized": hits_norm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    segs = collect()
    print("segments", len(segs), flush=True)
    with Pool(a.workers) as p:
        recs = p.map(translate, segs, chunksize=500)
    errors = [r for r in recs if "error" in r]
    recs = [r for r in recs if "error" not in r]
    random.Random(SEED).shuffle(recs)
    dev, train = recs[:N_DEV], recs[N_DEV:]
    os.makedirs(a.out, exist_ok=True)
    for name, rs in (("train", train), ("dev", dev)):
        with open(f"{a.out}/{name}.jsonl", "w", encoding="utf8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cont = contamination(recs)
    counts = {}
    for r in train:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    manifest = {
        "n_train": len(train), "n_dev": len(dev), "n_errors": len(errors), "n_8dot_dropped": sum(e["error"] == "8-dot cell" for e in errors), "by_source": counts,
        "aligned_g1": round(sum(r["aligned_g1"] for r in train) / len(train), 4),
        "aligned_g2": round(sum(r["aligned_g2"] for r in train) / len(train), 4),
        "contamination": cont, "seed": SEED, "caps": CAPS, "liblouis": version(),
        "command": "python " + " ".join(sys.argv),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "train_sha256": hashlib.sha256(open(f"{a.out}/train.jsonl", "rb").read()).hexdigest(),
    }
    json.dump(manifest, open(f"{a.out}/manifest.json", "w"), indent=1)
    print(json.dumps(manifest, indent=1))
    assert cont["exact"] == 0 and cont["normalized"] == 0, "test items in training data"


if __name__ == "__main__":
    main()

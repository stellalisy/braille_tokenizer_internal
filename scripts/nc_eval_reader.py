"""E0 of docs/design_noisy_channel.md: the character reader against the revision 2 token reader.

Both readers decode by beam search (k=16) and are scored on the same items:
  copy probe   typo and random strings read in isolation (built as in scripts/analysis/vocab_copy_diag.py,
               seed 0, excluding known words and every non-word training string)
  test words   words of the Phase A test spans (questions and choices of CommonsenseQA and GSM8K) clean
               and at 10% flip, writing and ocr noise (data/v2inputs/{ds}/g{g}_unicode_{p0,p10,w10,o10});
               spans whose braille and print word counts differ are skipped
Reported: top-1 exact accuracy and oracle@16 (right spelling anywhere in the 16).

    python scripts/nc_eval_reader.py --char exp/nc/g1/reader --token exp/vocab/v2/g1_scratch_local_nonw_mix
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys

import torch

sys.path.insert(0, ".")


def probe_sets(g, held, n=1000, seed=0):
    rng = random.Random(seed)
    known, freq = set(), {}
    for line in open("data/phaseB/train.jsonl", encoding="utf8"):
        for w in re.findall(r"[a-z]+", json.loads(line)["text"].lower()):
            known.add(w)
            freq[w] = freq.get(w, 0) + 1
    known.update(json.loads(l)["text"].lower() for l in open(f"data/vocab/words_g{g}.jsonl", encoding="utf8"))
    avoid = set()
    for line in open(f"data/vocab/nonwords_g{g}.jsonl", encoding="utf8"):
        avoid.update(w.lower() for w in json.loads(line)["text"].split())
    ranked = [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:20000] if len(w) >= 4 and w.isalpha()]
    common = [w for w in ranked[1::2] if w not in held]

    def typo(w):
        for _ in range(50):
            i = rng.randrange(len(w))
            op = rng.choice("sids")
            c = chr(97 + rng.randrange(26))
            if op == "s":
                t = w[:i] + c + w[i + 1:]
            elif op == "i":
                t = w[:i] + c + w[i:]
            elif op == "d":
                t = w[:i] + w[i + 1:]
            else:
                if i == len(w) - 1:
                    continue
                t = w[:i] + w[i + 1] + w[i] + w[i + 2:]
            if t != w and t not in known and t not in avoid and len(t) >= 3:
                return t
        return None

    ty = []
    while len(ty) < n:
        t = typo(rng.choice(common))
        if t:
            ty.append(t)
    rd = []
    while len(rd) < n:
        t = "".join(chr(97 + rng.randrange(26)) for _ in range(rng.randint(4, 9)))
        if t not in known and t not in avoid:
            rd.append(t)
    return {"typo": ty, "random": rd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", required=True, help="nc_train.py reader run")
    ap.add_argument("--token", required=True, help="revision 2 token reader run")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--inputs", default="data/v2inputs")
    ap.add_argument("--conds", default="p0,p10,w10,o10")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from brltok import translate
    from brltok.cells import normalize
    from brltok.seq2seq import CharVocab, cell_input, load
    from brltok.vocab_encoder import load_encoder, span_words

    reader, meta = load(f"{a.char}/model.pt")
    cv = CharVocab(meta["chars"])
    g = meta["grade"]
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    m = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=torch.bfloat16)
    emb = m.get_input_embeddings().weight.detach().float()
    del m
    enc = load_encoder(f"{a.token}/encoder.pt", emb)

    def read_char(words):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            nb = reader.nbest([cell_input(w) for w in words], k=a.k)
        return [[cv.decode(s) for s, _ in lst] for lst in nb]

    def read_token(words):
        out = []
        for j in range(0, len(words), 128):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                res = enc.nbest(words[j: j + 128], k=a.k)
            out.extend([[tok.decode(ids).strip() for ids, _ in r[0]] for r in res])
            torch.cuda.empty_cache()
        return out

    readers = {"char": read_char, "token": read_token}
    report = {}
    held = set(json.load(open(f"{a.char}/heldout_words.json")))
    for name, strs in probe_sets(g, held).items():
        words = [normalize(translate.forward_string(s, g)) for s in strs]
        keep = [(w, s) for w, s in zip(words, strs) if len(span_words(w)) == 1]
        for rn, fn in readers.items():
            got = fn([w for w, _ in keep])
            report[f"copy_{name}/{rn}"] = {
                "n": len(keep), "top1": round(100 * sum(c[0] == s if c else False for c, (_, s) in zip(got, keep)) / len(keep), 2),
                f"oracle{a.k}": round(100 * sum(s in c for c, (_, s) in zip(got, keep)) / len(keep), 2)}
            print(f"copy_{name}/{rn}", report[f"copy_{name}/{rn}"], flush=True)
    for cond in a.conds.split(","):
        words, golds = [], []
        for ds in ("commonsenseqa", "gsm8k"):
            for line in open(f"{a.inputs}/{ds}/g{g}_unicode_{cond}.jsonl", encoding="utf8"):
                r = json.loads(line)
                pairs = [(r["braille_question"], r.get("clean_question", r["question"]))]
                pairs += list(zip(r.get("braille_choices", []), r.get("clean_choices", r.get("choices", []))))
                for b, t in pairs:
                    b = normalize(b)
                    if len(span_words(b)) == len(t.split()):
                        words.extend(b[s:e] for s, e in span_words(b))
                        golds.extend(t.split())
        for rn, fn in readers.items():
            got = fn(words)
            report[f"test_{cond}/{rn}"] = {
                "n": len(words), "top1": round(100 * sum(c[0] == w if c else False for c, w in zip(got, golds)) / len(words), 2),
                f"oracle{a.k}": round(100 * sum(w in c for c, w in zip(got, golds)) / len(words), 2)}
            print(f"test_{cond}/{rn}", report[f"test_{cond}/{rn}"], flush=True)
    os.makedirs(f"{a.char}/eval", exist_ok=True)
    json.dump({"report": report, "args": vars(a), "command": "python " + " ".join(sys.argv),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "liblouis": translate.version(), "slurm_job": os.environ.get("SLURM_JOB_ID")},
              open(f"{a.char}/eval/e0.json", "w"), indent=1)


if __name__ == "__main__":
    main()

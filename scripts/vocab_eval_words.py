"""Word-level reading accuracy of the print-vocabulary encoder, before any language model.

Spans are the Phase A test inputs (questions and choices of CommonsenseQA and GSM8K, clean and with
2, 5, 10% dot flips; data/phaseA/{ds}/g{g}_unicode_p{P}.jsonl). A span is scored when its braille and
print have the same number of words. A word is read correctly when the greedy decoded tokens give
exactly the print word. The liblouis back-translation of the same braille span is scored the same
way (word strings, when its word count matches).

Words are split into seen (in the training sentences or the training word list), unseen, and
non-alphabetic (numbers, symbols). Writes <run>/words/result.json and misread samples.

    python scripts/vocab_eval_words.py --run exp/vocab/g1_a_s0

With --noise-profile writing|ocr the clean inputs are corrupted here with brltok.noise.braille_noise
at each rate (seeded) instead of reading the dot-flip files; outputs get a _<profile> suffix. Because
insertions and deletions can change liblouis's word count, "pipeline_strict" counts every word of a
span as wrong when the back-translation does not split into the right number of words (the plain
"pipeline" figure leaves such spans out, which flatters liblouis).
"""
import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

import torch

sys.path.insert(0, ".")


def key(w):
    letters = re.findall(r"[a-z]+(?:'[a-z]+)?", w.lower())
    return " ".join(letters) if letters else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--encoder", default=None, help="default <run>/encoder.pt")
    ap.add_argument("--inputs", default="data/phaseA")
    ap.add_argument("--data", default="data/phaseB")
    ap.add_argument("--vocab", default="data/vocab")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    ap.add_argument("--rates", default="0,2,5,10")
    ap.add_argument("--hard", action="store_true")
    ap.add_argument("--noise-profile", default=None, choices=["writing", "ocr"])
    a = ap.parse_args()
    out = f"{a.run}/words"
    sfx = f"_{a.noise_profile}" if a.noise_profile else ""
    os.makedirs(out, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from brltok import translate
    from brltok.cells import normalize
    from brltok.vocab_encoder import load_encoder, span_words, word_targets

    log = json.load(open(f"{a.run}/train_log.json"))
    g = log["args"]["grade"]
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    m = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=torch.bfloat16)
    emb = m.get_input_embeddings().weight.detach().float()
    del m
    enc = load_encoder(a.encoder or f"{a.run}/encoder.pt", emb)

    seen = set()
    for line in open(f"{a.data}/train.jsonl", encoding="utf8"):
        seen.update(re.findall(r"[a-z]+(?:'[a-z]+)?", json.loads(line)["text"].lower()))
    held = set(json.load(open(f"{a.run}/heldout_words.json")))
    for line in open(f"{a.vocab}/words_g{g}.jsonl", encoding="utf8"):
        w = json.loads(line)["text"].lower()
        if w not in held:
            seen.add(w)

    def cat(w):
        k = key(w)
        if k is None:
            return "nonalpha"
        return "seen" if all(p in seen for p in k.split()) else "unseen"

    results, samples = {}, defaultdict(list)
    for ds in ("commonsenseqa", "gsm8k"):
        for P in [int(x) for x in a.rates.split(",")]:
            if a.noise_profile:
                import random as _r
                from brltok.noise import braille_noise
                nr = _r.Random(1000 + P)
                recs = [json.loads(l) for l in open(f"{a.inputs}/{ds}/g{g}_unicode_p0.jsonl", encoding="utf8")]
                for r in recs:
                    r["braille_question"] = braille_noise(normalize(r["braille_question"]), P / 100, nr, a.noise_profile)
                    if r.get("braille_choices"):
                        r["braille_choices"] = [braille_noise(normalize(c), P / 100, nr, a.noise_profile)
                                                for c in r["braille_choices"]]
            else:
                recs = [json.loads(l) for l in open(f"{a.inputs}/{ds}/g{g}_unicode_p{P}.jsonl", encoding="utf8")]
            spans, texts = [], []
            n_spans = 0
            for r in recs:
                pairs = [(r["braille_question"], r["question"])]
                pairs += list(zip(r.get("braille_choices", []), r.get("choices", [])))
                for b, t in pairs:
                    n_spans += 1
                    b = normalize(b)
                    if len(span_words(b)) == len(t.split()):
                        spans.append(b)
                        texts.append(t)
            stats = defaultdict(lambda: [0, 0])
            conf = {"right": [], "wrong": []}
            pipe = defaultdict(lambda: [0, 0])
            strict = defaultdict(lambda: [0, 0])
            for k in range(0, len(spans), 64):
                dec = enc.decode(spans[k: k + 64], hard=a.hard)
                for b, t, (vec, ids, cf, lens) in zip(spans[k: k + 64], texts[k: k + 64], dec):
                    words = t.split()
                    gold = word_targets(tok, t)
                    back = translate.back(b, g).split()
                    back_ok = len(back) == len(words)
                    c = 0
                    for w, gw, n, bw in zip(words, gold, lens, back if back_ok else [None] * len(words)):
                        pred = ids[c: c + n]
                        pc = cf[c: c + n]
                        c += n
                        ok = tok.decode(pred).strip() == w
                        for s in ("all", cat(w), f"ptok{min(len(gw), 4)}"):
                            stats[s][0] += ok
                            stats[s][1] += 1
                        conf["right" if ok else "wrong"].append(min(pc) if pc else 0.0)
                        for s in ("all", cat(w)):
                            strict[s][0] += back_ok and bw == w
                            strict[s][1] += 1
                        if back_ok:
                            for s in ("all", cat(w)):
                                pipe[s][0] += bw == w
                                pipe[s][1] += 1
                            pipe["enc_on_same_words"][0] += ok
                            pipe["enc_on_same_words"][1] += 1
                        if not ok and len(samples[f"{ds}_p{P}"]) < 200:
                            samples[f"{ds}_p{P}"].append({"word": w, "read": tok.decode(pred).strip(), "cat": cat(w),
                                                          "back": bw if back_ok else None})
            res = {s: {"acc": round(100 * v[0] / v[1], 2), "n": v[1]} for s, v in sorted(stats.items())}
            res["pipeline"] = {s: {"acc": round(100 * v[0] / v[1], 2), "n": v[1]} for s, v in sorted(pipe.items())}
            res["pipeline_strict"] = {s: {"acc": round(100 * v[0] / v[1], 2), "n": v[1]} for s, v in sorted(strict.items())}
            res["confidence"] = {k2: round(sum(v) / max(1, len(v)), 3) for k2, v in conf.items()}
            res["spans_scored"] = f"{len(spans)}/{n_spans}"
            results[f"{ds}/p{P}"] = res
            print(f"{ds:14s} p{P:<2d} all {res['all']['acc']:6.2f}  seen {res.get('seen', {}).get('acc', '-')}  "
                  f"unseen {res.get('unseen', {}).get('acc', '-')} (n={res.get('unseen', {}).get('n', 0)})  "
                  f"nonalpha {res.get('nonalpha', {}).get('acc', '-')}  | pipeline {res['pipeline'].get('all', {}).get('acc', '-')} "
                  f"vs enc {res['pipeline'].get('enc_on_same_words', {}).get('acc', '-')}  "
                  f"pipeline strict {res['pipeline_strict']['all']['acc']}  conf r/w {res['confidence']}", flush=True)
    json.dump({"results": results, "run": a.run, "hard": a.hard, "command": "python " + " ".join(sys.argv),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "liblouis": translate.version(), "slurm_job": os.environ.get("SLURM_JOB_ID")},
              open(f"{out}/result{'_hard' if a.hard else ''}{sfx}.json", "w"), indent=1)
    json.dump(samples, open(f"{out}/misreads{sfx}.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

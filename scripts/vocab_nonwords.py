"""Non-word training data for the print-vocabulary encoder (design note revision 2).

Copying must be required by the data: every item's label is its exact spelling, and none of the
strings is a known word. Four kinds, written to data/vocab/nonwords_g{g}.jsonl as {"braille", "text",
"kind"}:
  sent_typo   : a training sentence with one or two words replaced by a one-letter edit of themselves
  sent_random : a training sentence with one word replaced by a random letter string
  word_typo   : a one-letter edit of a common word, alone
  word_random : a random letter string (4 to 9 letters), alone
Typo bases are restricted to words at even frequency ranks of the training corpus (or outside its top
20,000), so the copy probe (scripts/analysis/vocab_copy_diag.py), which draws typo bases from odd
ranks, never sees a training base. Held-out dev words are never used. 30% of isolated items are
capitalized.

    python scripts/vocab_nonwords.py --grade 1 --n 200000
"""
import argparse
import json
import random
import re
import sys
from collections import Counter

sys.path.insert(0, ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grade", type=int, required=True)
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--heldout", required=True, help="heldout_words.json of the Stage A run")
    a = ap.parse_args()
    from brltok import translate
    from brltok.cells import normalize
    from brltok.vocab_encoder import span_words

    g = a.grade
    rng = random.Random(a.seed)
    held = set(json.load(open(a.heldout)))
    sents, freq, known = [], Counter(), set()
    for line in open("data/phaseB/train.jsonl", encoding="utf8"):
        t = json.loads(line)["text"]
        sents.append(t)
        ws = re.findall(r"[a-z]+", t.lower())
        freq.update(ws)
        known.update(ws)
    for line in open(f"data/vocab/words_g{g}.jsonl", encoding="utf8"):
        known.add(json.loads(line)["text"].lower())
    ranked = [w for w, _ in freq.most_common(20000) if len(w) >= 4 and w.isalpha()]
    odd = set(ranked[1::2])
    bases = [w for w in ranked[0::2] if w not in held]

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
                if i >= len(w) - 1:
                    continue
                t = w[:i] + w[i + 1] + w[i] + w[i + 2:]
            if t != w and t.lower() not in known and len(t) >= 3:
                return t
        return None

    def rand_str():
        while True:
            t = "".join(chr(97 + rng.randrange(26)) for _ in range(rng.randint(4, 9)))
            if t not in known:
                return t

    def cap(t):
        return t[0].upper() + t[1:] if rng.random() < 0.3 else t

    kinds = ["sent_typo"] * 40 + ["sent_random"] * 15 + ["word_typo"] * 30 + ["word_random"] * 15
    out = open(f"data/vocab/nonwords_g{g}.jsonl", "w", encoding="utf8")
    counts = Counter()
    while sum(counts.values()) < a.n:
        k = rng.choice(kinds)
        if k.startswith("word"):
            t = typo(rng.choice(bases)) if k == "word_typo" else rand_str()
            if not t:
                continue
            text = cap(t)
        else:
            ws = rng.choice(sents).split()
            cand = [i for i, w in enumerate(ws) if w.isalpha() and len(w) >= 4]
            if not cand:
                continue
            picks = rng.sample(cand, min(len(cand), 1 if k == "sent_random" else rng.randint(1, 2)))
            ok = True
            for i in picks:
                w = ws[i]
                if k == "sent_random":
                    new = rand_str()
                else:
                    lw = w.lower()
                    if lw in odd or lw in held:
                        ok = False
                        break
                    new = typo(lw)
                    if not new:
                        ok = False
                        break
                ws[i] = new[0].upper() + new[1:] if w[0].isupper() else new
            if not ok:
                continue
            text = " ".join(ws)
        b = normalize(translate.forward_string(text, g))
        if len(span_words(b)) != len(text.split()):
            continue
        out.write(json.dumps({"braille": b, "text": text, "kind": k}, ensure_ascii=False) + "\n")
        counts[k] += 1
    out.close()
    json.dump({"counts": counts, "seed": a.seed, "n": a.n, "liblouis": translate.version()},
              open(f"data/vocab/nonwords_g{g}.manifest.json", "w"), indent=1)
    print(dict(counts))


if __name__ == "__main__":
    main()

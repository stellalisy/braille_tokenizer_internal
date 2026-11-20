"""Build the synthetic parallel corpus used by the tokenizers and toy models.

Source text is the NLTK Gutenberg sample (public-domain novels). Each line of
the output JSONL holds the print sentence, its UEB Grade 1 and Grade 2 braille,
and the Grade 2 rule units with their print expansions.

    python scripts/build_data.py --out data --test-sentences sents.json
"""
import argparse
import json
import os
import random
import re

from brltok import translate
from brltok.align import units_from

SKIP = ("bible", "shakespeare", "milton", "blake", "whitman")


def sentences(min_words, max_words):
    import nltk
    from nltk.corpus import gutenberg

    out = []
    for f in gutenberg.fileids():
        if f.startswith(SKIP):
            continue
        raw = re.sub(r"\s+", " ", gutenberg.raw(f))
        for s in nltk.sent_tokenize(raw):
            s = s.strip()
            if (min_words <= len(s.split()) <= max_words and s.isascii()
                    and not any(c in s for c in "_[]*")):
                out.append(s)
    return out


def record(text):
    g2 = translate.forward(text, 2)
    us = units_from(g2)
    return {
        "text": text,
        "g1": translate.forward_string(text, 1),
        "g2": g2.braille,
        "units": [[u.cells, u.text] for u in us],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--test-sentences", help="JSON list of held-out test sentences")
    ap.add_argument("--max-train", type=int, default=60000)
    ap.add_argument("--dev", type=int, default=500)
    ap.add_argument("--min-words", type=int, default=5)
    ap.add_argument("--max-words", type=int, default=35)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    pool = sorted(set(sentences(a.min_words, a.max_words)))
    rng.shuffle(pool)
    if a.test_sentences:
        test = json.load(open(a.test_sentences))
        held = set(test)
        pool = [s for s in pool if s not in held]
    else:
        test, pool = pool[:500], pool[500:]
    dev, train = pool[:a.dev], pool[a.dev:a.dev + a.max_train]
    for name, split in (("train", train), ("dev", dev), ("test", test)):
        with open(os.path.join(a.out, f"{name}.jsonl"), "w", encoding="utf8") as f:
            for s in split:
                f.write(json.dumps(record(s), ensure_ascii=False) + "\n")
        print(name, len(split))
    print("liblouis", translate.version())


if __name__ == "__main__":
    main()

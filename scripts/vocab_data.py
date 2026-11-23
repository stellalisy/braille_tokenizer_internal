"""Data for the print-vocabulary encoder (docs/design_print_vocab_encoder.md).

Writes to --out (default data/vocab):
  words_g{1,2}.jsonl   single-word pairs {"text", "braille"} from a general English frequency list
                       (wordfreq, top --n-words alphabetic words; 30% also capitalized). Independent
                       of the test sets.
  allowed_ids.json     print token ids the encoder may output: every token whose text is printable
                       ASCII, plus any token in the training targets (sentences and word list).
  vocab_words.json     lowercase word types seen in training (corpus sentences and word list), for
                       reporting seen and unseen test words.
  manifest.json        sizes, versions, command.

    python scripts/vocab_data.py
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys

sys.path.insert(0, ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/vocab")
    ap.add_argument("--train", default="data/phaseB/train.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    ap.add_argument("--n-words", type=int, default=150000)
    ap.add_argument("--cap-share", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import wordfreq
    from transformers import AutoTokenizer

    from brltok import translate

    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    rng = random.Random(a.seed)
    words = [w for w in wordfreq.top_n_list("en", a.n_words * 2) if re.fullmatch(r"[a-z]+(?:'[a-z]+)?", w)][: a.n_words]
    forms = []
    for w in words:
        forms.append(w)
        if rng.random() < a.cap_share:
            forms.append(w[0].upper() + w[1:])
    counts = {}
    for g in (1, 2):
        n = 0
        with open(f"{a.out}/words_g{g}.jsonl", "w", encoding="utf8") as f:
            for w in forms:
                b = translate.forward_string(w, g)
                if not b or " " in b.strip():
                    continue
                f.write(json.dumps({"text": w, "braille": b.strip()}, ensure_ascii=False) + "\n")
                n += 1
        counts[f"words_g{g}"] = n
        print(f"g{g}: {n} word pairs", flush=True)

    allowed = set()
    for i in range(len(tok)):
        s = tok.decode([i])
        if s and all(32 <= ord(c) < 127 for c in s):
            allowed.add(i)
    n_ascii = len(allowed)
    seen = set(w.lower() for w in words)
    for line in open(a.train, encoding="utf8"):
        r = json.loads(line)
        for w in r["text"].split():
            allowed.update(tok(" " + w, add_special_tokens=False)["input_ids"])
            seen.update(re.findall(r"[a-z]+(?:'[a-z]+)?", w.lower()))
    for w in forms:
        allowed.update(tok(" " + w, add_special_tokens=False)["input_ids"])
    allowed -= set(tok.all_special_ids)
    json.dump(sorted(allowed), open(f"{a.out}/allowed_ids.json", "w"))
    json.dump(sorted(seen), open(f"{a.out}/vocab_words.json", "w"))
    man = dict(counts, n_wordfreq=len(words), wordfreq=wordfreq.__version__ if hasattr(wordfreq, "__version__") else "3.1.1",
               allowed_ids=len(allowed), allowed_ascii=n_ascii, seen_word_types=len(seen),
               liblouis=translate.version(), seed=a.seed, command="python " + " ".join(sys.argv),
               git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    json.dump(man, open(f"{a.out}/manifest.json", "w"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()

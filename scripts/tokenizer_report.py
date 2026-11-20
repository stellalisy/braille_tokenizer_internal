"""Train T2 and T3 on the training split and report length, vocabulary and coverage.

    python scripts/tokenizer_report.py --data data --out runs/tokenizers
"""
import argparse
import json
import os

from brltok.tokenizers import UNK, BrailleBPE, CellTokenizer, RuleUnitTokenizer


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf8")]


def gold_segments(rec):
    """Rule segmentation of one sentence as a list of words, each a list of units."""
    out, cur = [], []
    for cells, _ in rec["units"]:
        if cells == " ":
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(cells)
    if cur:
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/tokenizers")
    ap.add_argument("--bpe-sizes", default="1000,4000,8000,16000")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    train, test = load(f"{a.data}/train.jsonl"), load(f"{a.data}/test.jsonl")
    lines = [r["g2"] for r in train]
    test_lines = [r["g2"] for r in test]
    n_words = sum(len(l.split()) for l in test_lines)
    n_chars = sum(len(r["text"]) for r in test)
    n_cells = sum(len(l.replace(" ", "")) for l in test_lines)

    toks = {"T1 cell": CellTokenizer()}
    for v in map(int, a.bpe_sizes.split(",")):
        t = BrailleBPE.train(lines, v)
        t.save(f"{a.out}/bpe{v}.json")
        toks[f"T2 BPE {v}"] = t
    rule = RuleUnitTokenizer.build([r["units"] for r in train])
    rule.save(f"{a.out}/rule.json")
    toks["T3 rule units"] = rule
    toks["T3 rule units, no lexicon"] = RuleUnitTokenizer(
        rule.vocab, rule.lexicon, rule.inventory, rule.expansions, use_lexicon=False)

    rows = []
    for name, t in toks.items():
        n_tok = n_unk = 0
        for l in test_lines:
            ids = t.encode(l)
            n_tok += len(ids)
            n_unk += sum(i == t.unk_id for i in ids)
        # T1 counts a space as a token; report it both ways so lengths compare
        n_nospace = n_tok - (n_words - len(test_lines)) if name == "T1 cell" else n_tok
        rows.append({"tokenizer": name, "vocab": t.vocab_size, "test_tokens": n_nospace,
                     "tokens_per_word": round(n_nospace / n_words, 3),
                     "chars_per_token": round(n_chars / n_nospace, 2), "unk": n_unk})

    # how often does braille-only segmentation reproduce the rule segmentation?
    agree = {}
    for name in ("T3 rule units", "T3 rule units, no lexicon"):
        t = toks[name]
        w_ok = w_all = s_ok = 0
        for r in test:
            gold = gold_segments(r)
            pred = [t.segment(w) for w in r["g2"].split()]
            hits = sum(g == p for g, p in zip(gold, pred))
            w_ok += hits
            w_all += len(gold)
            s_ok += hits == len(gold) and len(gold) == len(pred)
        agree[name] = {"word_agreement": round(w_ok / w_all, 4),
                       "sentence_agreement": round(s_ok / len(test), 4)}

    seen = sum(w in rule.lexicon for l in test_lines for w in l.split())
    report = {"test_sentences": len(test), "test_words": n_words, "test_cells": n_cells,
              "print_chars": n_chars, "rows": rows, "rule_segmentation": agree,
              "test_words_in_lexicon": round(seen / n_words, 4),
              "lexicon_size": len(rule.lexicon), "rule_multi_cell_units":
              sum(len(u) > 1 for u in rule.inventory)}
    json.dump(report, open(f"{a.out}/report.json", "w"), indent=1, ensure_ascii=False)
    for r in rows:
        print(r)
    print(agree)
    print({k: report[k] for k in ("test_words_in_lexicon", "lexicon_size", "rule_multi_cell_units")})


if __name__ == "__main__":
    main()

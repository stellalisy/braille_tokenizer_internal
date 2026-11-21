"""Phase B tokenizers and print expansions (gate B2).

Trains T2 (BPE, 8k) and T3 (rule units) on data/phaseB/train.jsonl for each grade, keeps T1
(cells), and counts every token's print expansions over the training data (brltok.expansion).

    python scripts/phaseB_tokenizers.py --data data/phaseB
"""
import argparse
import json
import os
import subprocess
import sys

from brltok.expansion import corpus_expansions
from brltok.tokenizers import BrailleBPE, CellTokenizer, RuleUnitTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/phaseB")
    ap.add_argument("--bpe-vocab", type=int, default=8000)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(f"{a.data}/train.jsonl", encoding="utf8")]
    os.makedirs(f"{a.data}/tok", exist_ok=True)
    report = {}
    for g in (1, 2):
        toks = {"t1": CellTokenizer(),
                "t2": BrailleBPE.train((r[f"g{g}"] for r in recs), vocab_size=a.bpe_vocab),
                "t3": RuleUnitTokenizer.build(r[f"units_g{g}"] for r in recs)}
        for name, tok in toks.items():
            if name != "t1":
                tok.save(f"{a.data}/tok/g{g}_{name}.json")
            counts, stats = corpus_expansions(tok, recs, g)
            with open(f"{a.data}/tok/g{g}_{name}_expansions.jsonl", "w", encoding="utf8") as f:
                for i, c in sorted(counts.items()):
                    f.write(json.dumps({"id": i, "exp": [[t, sw, n] for (t, sw), n in c.most_common()]},
                                       ensure_ascii=False) + "\n")
            used = len(counts)
            with_exp = sum(any(t.strip() for (t, _), _n in c.items()) for c in counts.values())
            report[f"g{g}_{name}"] = {
                "vocab": tok.vocab_size, "types_seen": used, "types_with_expansion": with_exp,
                "token_occurrences": stats["tokens"],
                "nonempty_share": round(stats["nonempty"] / stats["tokens"], 4),
                "tokens_per_word": round(stats["tokens"] / sum(len(r["text"].split()) for r in recs), 3)}
            print(f"g{g} {name}", report[f"g{g}_{name}"], flush=True)
    report["command"] = "python " + " ".join(sys.argv)
    report["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    json.dump(report, open(f"{a.data}/tok/report.json", "w"), indent=1)


if __name__ == "__main__":
    main()

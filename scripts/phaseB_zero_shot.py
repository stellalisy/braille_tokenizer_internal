"""Gate B3: can a frozen LM read braille through a new input layer before any training?

For each dev sentence the frozen model sees print "Braille: ", the braille input, print
"\nPrint:", and is scored on " " + the print sentence (mean NLL per target token). Inputs:
stock tokenizer on Braille ASCII and on Unicode braille, T1/T2/T3 with E1 (mean init) and
E2 (print-aligned init), and a print-to-print copy as the floor. No parameter is trained.

    python scripts/phaseB_zero_shot.py --data data/phaseB --n 200 --out exp/phaseB/zero_shot
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter

import torch

from brltok.bridge import BrailleInput, freeze, mean_init, param_hash, print_aligned_init
from brltok.cells import to_ascii
from brltok.tokenizers import BrailleBPE, CellTokenizer, RuleUnitTokenizer

PREFIX, SUFFIX = "Braille: ", "\nPrint:"


def load_counts(path):
    out = {}
    for l in open(path, encoding="utf8"):
        d = json.loads(l)
        out[d["id"]] = Counter({(t, sw): n for t, sw, n in d["exp"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/phaseB")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = freeze(AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=torch.float32))
    h0 = param_hash(model)
    base_emb = model.get_input_embeddings().weight.detach()
    dev = [json.loads(l) for l in open(f"{a.data}/dev.jsonl", encoding="utf8")][: a.n]

    @torch.no_grad()
    def nll(inputs_embeds, target):
        tgt = tok(target, add_special_tokens=False)["input_ids"]
        te = model.get_input_embeddings()(torch.tensor([tgt]))
        x = torch.cat([inputs_embeds, te], dim=1)
        logits = model(inputs_embeds=x).logits[0, -len(tgt) - 1:-1]
        return torch.nn.functional.cross_entropy(logits.float(), torch.tensor(tgt), reduction="sum").item(), len(tgt)

    def text_input(s):
        ids = tok(PREFIX + s + SUFFIX, add_special_tokens=False)["input_ids"]
        return model.get_input_embeddings()(torch.tensor([ids]))

    results, per_item = {}, {}

    def run(name, make):
        t0, tot, n, items = time.time(), 0.0, 0, []
        for r in dev:
            s, k = nll(make(r), " " + r["text"])
            tot, n = tot + s, n + k
            items.append(s / k)
        results[name] = {"nll": round(tot / n, 4), "seconds": round(time.time() - t0, 1)}
        per_item[name] = items
        print(name, results[name], flush=True)

    run("print_copy", lambda r: text_input(r["text"]))
    for g in (1, 2):
        run(f"g{g}_stock_ascii", lambda r: text_input(to_ascii(r[f"g{g}"])))
        run(f"g{g}_stock_unicode", lambda r: text_input(r[f"g{g}"]))
        toks = {"t1": CellTokenizer(), "t2": BrailleBPE.load(f"{a.data}/tok/g{g}_t2.json"),
                "t3": RuleUnitTokenizer.load(f"{a.data}/tok/g{g}_t3.json")}
        for tn, bt in toks.items():
            counts = load_counts(f"{a.data}/tok/g{g}_{tn}_expansions.jsonl")
            e2, fb = print_aligned_init(counts, tok, base_emb, bt.vocab_size)
            results[f"g{g}_{tn}_e2_fallback"] = fb
            for en, table in (("e1", mean_init(base_emb, bt.vocab_size)), ("e2", e2)):
                emb = torch.nn.Embedding.from_pretrained(table, freeze=True)
                bi = BrailleInput(emb)
                run(f"g{g}_{tn}_{en}",
                    lambda r, bi=bi, bt=bt: bi.splice(model, tok, PREFIX, bt.encode(r[f"g{g}"]), SUFFIX)[0])
    gate = {}
    for g in (1, 2):
        e2_beats_e1 = all(results[f"g{g}_{t}_e2"]["nll"] < results[f"g{g}_{t}_e1"]["nll"] for t in ("t1", "t2", "t3"))
        best = min(("t1", "t2", "t3"), key=lambda t: results[f"g{g}_{t}_e2"]["nll"])
        best_beats_stock = results[f"g{g}_{best}_e2"]["nll"] < results[f"g{g}_stock_ascii"]["nll"]
        wins = sum(x < y for x, y in zip(per_item[f"g{g}_{best}_e2"], per_item[f"g{g}_stock_ascii"]))
        gate[f"g{g}"] = {"e2_beats_e1_all": e2_beats_e1, "best_e2": best, "best_e2_beats_stock_ascii": best_beats_stock,
                         "best_e2_wins_vs_stock_items": f"{wins}/{len(dev)}", "pass": e2_beats_e1 and best_beats_stock}
    assert param_hash(model) == h0
    os.makedirs(a.out, exist_ok=True)
    json.dump({"results": results, "gate_b3": gate, "n": len(dev), "model": a.model, "revision": a.revision,
               "base_hash": h0, "command": "python " + " ".join(sys.argv),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()},
              open(f"{a.out}/result.json", "w"), indent=1)
    print(json.dumps(gate, indent=1))


if __name__ == "__main__":
    main()

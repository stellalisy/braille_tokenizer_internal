"""Phase B diagnostic: the model's distribution over CommonsenseQA options, braille vs print.

Both the braille-input prompt (trained table for the braille spans) and the same prompt in print are
followed by the answer prefix "The answer is:", and the next-token probabilities of " A" to " E" are
read and renormalized over the five letters. Per item we save both distributions and the gold letter;
the summary reports agreement, flips by print margin, and where the braille pick ranks under print.

    python scripts/phaseB_choiceprobs.py --run exp/phaseB/train/g1_t3_e2_s0
"""
import argparse
import json
import os
import subprocess
import sys

import torch

LETTERS = "ABCDE"
PREFIX = "The answer is:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None, help="default <run>/choiceprobs")
    ap.add_argument("--inputs", default="data/phaseA")
    ap.add_argument("--data", default="data/phaseB")
    ap.add_argument("--bb", default=os.path.expandvars("$BRL_SCRATCH/ext/braillebench"))
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    out = a.out or f"{a.run}/choiceprobs"
    os.makedirs(out, exist_ok=True)
    sys.path.insert(0, os.path.join(a.bb, "src"))
    import evaluate_braille as bb
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from brltok.scoring import gold_choices
    from brltok.splice import Braille, build, qa_segments
    from brltok.tokenizers import BrailleBPE, CellTokenizer, RuleUnitTokenizer

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=torch.bfloat16).to(dev).eval()
    base = model.get_input_embeddings()
    log = json.load(open(f"{a.run}/train_log.json"))
    grade, tname = log["args"]["grade"], log["args"]["tok"]
    btok = {"t1": lambda: CellTokenizer(),
            "t2": lambda: BrailleBPE.load(f"{a.data}/tok/g{grade}_t2.json"),
            "t3": lambda: RuleUnitTokenizer.load(f"{a.data}/tok/g{grade}_t3.json")}[tname]()
    table = torch.load(f"{a.run}/table.pt").to(torch.bfloat16).to(dev)
    letter_ids = [tok(" " + c, add_special_tokens=False)["input_ids"] for c in LETTERS]
    assert all(len(x) == 1 for x in letter_ids), letter_ids
    letter_ids = [x[0] for x in letter_ids]
    pre = tok(PREFIX, add_special_tokens=False)["input_ids"]

    def chat(p):
        return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    recs = [json.loads(l) for l in open(f"{a.inputs}/commonsenseqa/g{grade}_unicode_p0.jsonl", encoding="utf8")][: a.limit]
    rows = []
    for r in recs:
        ch = [Braille(b, p) for b, p in zip(r["braille_choices"], r["choices"])]
        segs = qa_segments(bb, "commonsenseqa", grade, Braille(r["braille_question"], r["question"]), ch, chat=chat)
        ex = build(segs, tok, btok)
        with torch.no_grad():
            x = torch.cat([base(torch.tensor(ids, device=dev)) if k == "t" else table[torch.tensor(ids, device=dev)]
                           for k, ids in ex.parts] + [base(torch.tensor(pre, device=dev))])[None]
            ls = model(inputs_embeds=x).logits[0, -1, letter_ids].float()
            lt = model(input_ids=torch.tensor([ex.teacher + pre], device=dev)).logits[0, -1, letter_ids].float()
        gold = sorted(gold_choices(r["golden_answers"], r["choices"]))
        rows.append({"id": r["id"], "gold": gold,
                     "p_braille": [round(v, 5) for v in torch.softmax(ls, -1).tolist()],
                     "p_print": [round(v, 5) for v in torch.softmax(lt, -1).tolist()]})
    with open(f"{out}/details.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    def top(p):
        return max(range(5), key=lambda i: p[i])

    n = len(rows)
    S = {"n": n}
    S["acc_print"] = 100 * sum(LETTERS[top(r["p_print"])] in r["gold"] for r in rows) / n
    S["acc_braille"] = 100 * sum(LETTERS[top(r["p_braille"])] in r["gold"] for r in rows) / n
    S["agree"] = 100 * sum(top(r["p_print"]) == top(r["p_braille"]) for r in rows) / n
    # where does the braille pick rank under print, when they disagree
    ranks = [sorted(range(5), key=lambda i: -r["p_print"][i]).index(top(r["p_braille"]))
             for r in rows if top(r["p_print"]) != top(r["p_braille"])]
    S["disagree_rank_under_print"] = {k + 1: ranks.count(k) for k in range(1, 5)}
    # flips by print confidence (probability of print's top choice)
    bins = [(0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.01)]
    S["agree_by_print_conf"] = {}
    for lo, hi in bins:
        sel = [r for r in rows if lo <= max(r["p_print"]) < hi]
        if sel:
            S["agree_by_print_conf"][f"{lo}-{hi}"] = {
                "n": len(sel), "agree": round(100 * sum(top(r["p_print"]) == top(r["p_braille"]) for r in sel) / len(sel), 1),
                "print_acc": round(100 * sum(LETTERS[top(r["p_print"])] in r["gold"] for r in sel) / len(sel), 1),
                "braille_acc": round(100 * sum(LETTERS[top(r["p_braille"])] in r["gold"] for r in sel) / len(sel), 1)}
    # confidence of braille when it disagrees with a confident print
    conf_dis = [max(r["p_braille"]) for r in rows if max(r["p_print"]) >= 0.9 and top(r["p_print"]) != top(r["p_braille"])]
    S["braille_conf_when_disagreeing_with_confident_print"] = {
        "n": len(conf_dis), "mean": round(sum(conf_dis) / max(1, len(conf_dis)), 3),
        "share_over_0.9": round(sum(c >= 0.9 for c in conf_dis) / max(1, len(conf_dis)), 3)}
    S["mean_max_prob"] = {"print": round(sum(max(r["p_print"]) for r in rows) / n, 3),
                          "braille": round(sum(max(r["p_braille"]) for r in rows) / n, 3)}
    S["mean_prob_on_print_choice"] = round(sum(r["p_braille"][top(r["p_print"])] for r in rows) / n, 3)
    S.update(run=a.run, command="python " + " ".join(sys.argv),
             git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
             slurm_job=os.environ.get("SLURM_JOB_ID"))
    json.dump(S, open(f"{out}/summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()

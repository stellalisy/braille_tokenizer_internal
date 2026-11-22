"""Phase B diagnostic: how close braille-input hidden states are to print, layer by layer.

For each test item the student is the braille-input prompt (trained table for the braille spans) and
the teacher is the same prompt with the print text in those spans, as in training. At every layer we
compare (a) the hidden state at the last token of each braille word with the teacher's state at the
last token of the same print word, and (b) the state at the last prompt position, from which the
answer is generated. Reported: normalized squared error |hs-ht|^2/|ht|^2 and cosine, by dataset.

    python scripts/phaseB_repsim.py --run exp/phaseB/train/g1_t3_e2_s0 --n 300
"""
import argparse
import json
import os
import subprocess
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None, help="default <run>/repsim.json")
    ap.add_argument("--inputs", default="data/phaseA")
    ap.add_argument("--data", default="data/phaseB")
    ap.add_argument("--datasets", default="commonsenseqa,gsm8k")
    ap.add_argument("--bb", default=os.path.expandvars("$BRL_SCRATCH/ext/braillebench"))
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.bb, "src"))
    import evaluate_braille as bb
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    def chat(p):
        return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    res = {}
    for ds in a.datasets.split(","):
        recs = [json.loads(l) for l in open(f"{a.inputs}/{ds}/g{grade}_unicode_p0.jsonl", encoding="utf8")][: a.n]
        acc = {k: None for k in ("w_mse", "w_cos", "l_mse", "l_cos")}
        nw = nl = 0
        for r in recs:
            ch = [Braille(b, p) for b, p in zip(r.get("braille_choices", []), r.get("choices", []))]
            segs = qa_segments(bb, ds, grade, Braille(r["braille_question"], r["question"]), ch, chat=chat)
            ex = build(segs, tok, btok)
            with torch.no_grad():
                x = torch.cat([base(torch.tensor(ids, device=dev)) if k == "t" else table[torch.tensor(ids, device=dev)]
                               for k, ids in ex.parts])[None]
                hs = model(inputs_embeds=x, output_hidden_states=True).hidden_states
                ht = model(input_ids=torch.tensor([ex.teacher], device=dev), output_hidden_states=True).hidden_states
            hs = torch.stack(hs)[:, 0].float()   # [L+1, T, d]
            ht = torch.stack(ht)[:, 0].float()
            vals = {}
            if ex.pairs:
                sp = torch.tensor([s for s, _ in ex.pairs], device=dev)
                tp = torch.tensor([t for _, t in ex.pairs], device=dev)
                s, t = hs[:, sp], ht[:, tp]
                vals["w_mse"] = ((s - t).pow(2).sum(-1) / t.pow(2).sum(-1).clamp(min=1e-6)).sum(1)
                vals["w_cos"] = torch.nn.functional.cosine_similarity(s, t, dim=-1).sum(1)
                nw += len(ex.pairs)
            s, t = hs[:, -1], ht[:, -1]
            vals["l_mse"] = (s - t).pow(2).sum(-1) / t.pow(2).sum(-1).clamp(min=1e-6)
            vals["l_cos"] = torch.nn.functional.cosine_similarity(s, t, dim=-1)
            nl += 1
            for k, v in vals.items():
                acc[k] = v if acc[k] is None else acc[k] + v
        out = {k: [round(x, 4) for x in (v / (nw if k.startswith("w") else nl)).tolist()] for k, v in acc.items()}
        out.update(n_items=nl, n_words=nw)
        res[ds] = out
        L = len(out["w_cos"])
        print(ds, "layer  word_nmse word_cos  last_nmse last_cos", flush=True)
        for i in range(L):
            print(f"  {i:2d}  {out['w_mse'][i]:.3f}  {out['w_cos'][i]:.3f}   {out['l_mse'][i]:.3f}  {out['l_cos'][i]:.3f}")
    res.update(run=a.run, command="python " + " ".join(sys.argv),
               git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               slurm_job=os.environ.get("SLURM_JOB_ID"))
    json.dump(res, open(a.out or f"{a.run}/repsim.json", "w"), indent=1)


if __name__ == "__main__":
    main()

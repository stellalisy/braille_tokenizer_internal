"""Phase B diagnostic: how well a trained braille input reads, measured by transcription.

The frozen model sees "Braille: " + braille span + "\\nPrint:" (the transcription format used in
training) and decodes greedily to the first newline. Word error rate (WER) against the print text
is reported per dataset and part (question, choices), with per-item outputs for error analysis.
Items are the clean Phase A test inputs (g{grade}_unicode_p0).

    python scripts/phaseB_transcribe.py --run exp/phaseB/train/g1_t3_e2_s0
"""
import argparse
import json
import os
import re
import subprocess
import sys

PREFIX, SUFFIX = "Braille: ", "\nPrint:"


def words(s):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", s.lower())


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(b)]


def strip_label(c):
    return re.sub(r"^[A-E]\)\s*", "", c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None, help="default <run>/transcribe")
    ap.add_argument("--inputs", default="data/phaseA")
    ap.add_argument("--data", default="data/phaseB")
    ap.add_argument("--datasets", default="commonsenseqa,gsm8k")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = a.out or f"{a.run}/transcribe"
    os.makedirs(out, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM
    from vllm import LLM, SamplingParams

    from brltok.splice import Braille, build
    from brltok.tokenizers import BrailleBPE, CellTokenizer, RuleUnitTokenizer

    hf = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=torch.bfloat16)
    base = hf.get_input_embeddings().weight.detach().clone()
    del hf
    llm = LLM(model=a.model, revision=a.revision, dtype="bfloat16", seed=a.seed, max_model_len=4096,
              enable_prompt_embeds=True, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=400, stop=["\n"], seed=a.seed)

    log = json.load(open(f"{a.run}/train_log.json"))
    grade, tname = log["args"]["grade"], log["args"]["tok"]
    btok = {"t1": lambda: CellTokenizer(),
            "t2": lambda: BrailleBPE.load(f"{a.data}/tok/g{grade}_t2.json"),
            "t3": lambda: RuleUnitTokenizer.load(f"{a.data}/tok/g{grade}_t3.json")}[tname]()
    table = torch.load(f"{a.run}/table.pt").to(torch.bfloat16)

    def embed(braille):
        ex = build([PREFIX, Braille(braille), SUFFIX], tok, btok)
        return torch.cat([base[torch.tensor(ids)] if k == "t" else table[torch.tensor(ids)] for k, ids in ex.parts])

    summary = {}
    for ds in a.datasets.split(","):
        recs = [json.loads(l) for l in open(f"{a.inputs}/{ds}/g{grade}_unicode_p0.jsonl", encoding="utf8")][: a.limit]
        jobs = []   # (item index, part, braille, reference print)
        for i, r in enumerate(recs):
            jobs.append((i, "question", r["braille_question"], r["question"]))
            for b, p in zip(r.get("braille_choices", []), r.get("choices", [])):
                jobs.append((i, "choice", b, p))
        outs = llm.generate([{"prompt_embeds": embed(b)} for _, _, b, _ in jobs], sp, use_tqdm=False)
        rows = []
        for (i, part, b, ref), o in zip(jobs, outs):
            hyp = o.outputs[0].text.strip()
            rw, hw = words(ref), words(hyp)
            if part == "choice":   # the "A)" label is transcribed too; score the answer text only
                rw, hw = words(strip_label(ref)), words(strip_label(hyp))
            rows.append({"item": recs[i].get("id"), "part": part, "ref": ref, "hyp": hyp,
                         "errors": edit_distance(rw, hw), "ref_words": len(rw), "exact": rw == hw})
        with open(f"{out}/{ds}_details.jsonl", "w", encoding="utf8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        for part in ("question", "choice"):
            rs = [r for r in rows if r["part"] == part]
            if not rs:
                continue
            wer = 100 * sum(r["errors"] for r in rs) / max(1, sum(r["ref_words"] for r in rs))
            ex = 100 * sum(r["exact"] for r in rs) / len(rs)
            summary[f"{ds}/{part}"] = {"n": len(rs), "wer": round(wer, 2), "exact": round(ex, 2)}
            print(f"{ds}/{part:9s} n={len(rs)} WER={wer:.2f} exact={ex:.1f}", flush=True)
        if ds == "commonsenseqa":   # items whose five choices all transcribe exactly
            by = {}
            for r in rows:
                if r["part"] == "choice":
                    by.setdefault(r["item"], []).append(r["exact"])
            summary[f"{ds}/all_choices_exact"] = round(100 * sum(all(v) for v in by.values()) / len(by), 2)
            print("all five choices exact:", summary[f"{ds}/all_choices_exact"], flush=True)

    json.dump({"summary": summary, "run": a.run, "command": "python " + " ".join(sys.argv),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "decoding": {"temperature": 0.0, "max_tokens": 400, "stop": "\\n"},
               "slurm_job": os.environ.get("SLURM_JOB_ID")}, open(f"{out}/result.json", "w"), indent=1)


if __name__ == "__main__":
    main()

"""Phase B evaluation: the frozen model reading braille through a trained input layer.

Each item gets BrailleBench's braille-input prompt (G1-EN or G2-EN, Unicode note) in the chat
template with thinking off (--thinking turns it on); the question and choices are braille spans embedded with the trained
table, the rest is print embedded with the base model's own table. vLLM generates from these
prompt embeddings with greedy decoding, and extraction and scoring are those of Phase A
(scripts/phaseA_eval.py): BrailleBench extraction and math_verify for GSM8K, choice scoring for
CommonsenseQA. Braille comes from the Phase A input files (Unicode, same items and noise).

--print-embeds runs the Phase A print condition through the same prompt-embedding path, as a
check that the path itself costs nothing.

    python scripts/phaseB_eval.py --run exp/phaseB/train/g1_t3_e2_s0 --conditions g1_p0
    python scripts/phaseB_eval.py --print-embeds --out exp/phaseB/eval_print_embeds --limit 200
"""
import argparse
import json
import os
import subprocess
import sys
import time

PRIMARY = {"gsm8k": "math_verify", "commonsenseqa": "choice"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="training run dir with table.pt and train_log.json")
    ap.add_argument("--print-embeds", action="store_true")
    ap.add_argument("--out", default=None, help="default <run>/eval")
    ap.add_argument("--conditions", default="", help="comma list like g1_p0,g1_p5; grade must match the run")
    ap.add_argument("--datasets", default="gsm8k,commonsenseqa")
    ap.add_argument("--inputs", default="data/phaseA")
    ap.add_argument("--data", default="data/phaseB")
    ap.add_argument("--bb", default=os.path.expandvars("$BRL_SCRATCH/ext/braillebench"))
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=10240)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--encoder", default=None,
                    help="print-vocabulary encoder checkpoint (brltok.vocab_encoder); --run then names its run dir")
    ap.add_argument("--hard", action="store_true", help="with --encoder: argmax token embeddings instead of soft mixes")
    ap.add_argument("--readings", default=None,
                    help="with --encoder: directory of <ds>_<cond>.jsonl print ids per span (scripts/vocab_rescore.py), "
                         "embedded with the base table in place of the reader's output")
    ap.add_argument("--thinking", action="store_true",
                    help="Qwen3 thinking mode; use with --max-tokens 32768 --max-model-len 36864 as in Phase A")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--print-part", choices=["question", "choices", "all"], default=None,
                    help="diagnostic: give this part of the prompt in print, the rest in braille")
    ap.add_argument("--qa-dev", default="data/phaseB/qa/dev.jsonl",
                    help="items for condition qadev (held-out training questions, for selection)")
    a = ap.parse_args()
    out = a.out or f"{a.run}/eval"
    os.makedirs(out, exist_ok=True)

    sys.path.insert(0, os.path.join(a.bb, "src"))
    import evaluate_braille as bb
    import torch
    from transformers import AutoModelForCausalLM
    from vllm import LLM, SamplingParams
    import vllm

    from brltok.scoring import choice_correct
    from brltok.splice import Braille, build, qa_segments
    from brltok.tokenizers import BrailleBPE, CellTokenizer, RuleUnitTokenizer

    hf = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=torch.bfloat16)
    base = hf.get_input_embeddings().weight.detach().clone()
    del hf
    if a.encoder:
        from brltok.vocab_encoder import build_vocab, load_encoder
        enc = load_encoder(a.encoder, base.float(), device="cuda")
        base_bf = base
    llm = LLM(model=a.model, revision=a.revision, dtype="bfloat16", seed=a.seed, max_model_len=a.max_model_len,
              enable_prompt_embeds=True, gpu_memory_utilization=0.80 if a.encoder else 0.85)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, seed=a.seed)

    def chat(p):
        return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=a.thinking)

    if a.print_embeds:
        jobs = [(ds, "print") for ds in a.datasets.split(",")]
        table = btok = None
        grade = None
    elif a.encoder:
        log = json.load(open(f"{a.run}/train_log.json"))
        grade = log["args"]["grade"]
        table = btok = None
        jobs = [(ds, c) for ds in a.datasets.split(",") for c in a.conditions.split(",")]
    else:
        log = json.load(open(f"{a.run}/train_log.json"))
        grade, tname = log["args"]["grade"], log["args"]["tok"]
        btok = {"t1": lambda: CellTokenizer(),
                "t2": lambda: BrailleBPE.load(f"{a.data}/tok/g{grade}_t2.json"),
                "t3": lambda: RuleUnitTokenizer.load(f"{a.data}/tok/g{grade}_t3.json")}[tname]()
        table = torch.load(f"{a.run}/table.pt").to(torch.bfloat16)
        assert table.shape == (btok.vocab_size, base.shape[1])
        jobs = [(ds, c) for ds in a.datasets.split(",") for c in a.conditions.split(",")]

    def prompt_embeds(ds, r):
        if a.print_embeds:
            p = chat(bb.make_braille_prompt(ds, r, bb.CONFIGS["EN-EN"], None))
            return base[tok(p, add_special_tokens=False)["input_ids"]], p
        ch = [Braille(c) for c in r.get("braille_choices", [])] if ds == "commonsenseqa" else []
        q = Braille(r["braille_question"])
        if a.print_part in ("question", "all"):
            q = r["question"]
        if a.print_part in ("choices", "all") and ch:
            ch = list(r["choices"])
        segs = qa_segments(bb, ds, grade, q, ch, chat=chat)
        shown = "".join(s if isinstance(s, str) else "[" + s.braille + "]" for s in segs)
        if a.encoder:
            vx = build_vocab(segs, tok)
            if "_reading" in r:
                assert len(r["_reading"]) == len(vx.spans)
                dec = [(base_bf[torch.tensor(ids, dtype=torch.long)],) for ids in r["_reading"]]
            else:
                dec = enc.decode(vx.spans, hard=a.hard) if vx.spans else []
            parts = [base_bf[torch.tensor(v)] if k == "t" else dec[v][0].to("cpu", torch.bfloat16) for k, v in vx.parts]
            return torch.cat(parts), shown
        ex = build(segs, tok, btok)
        parts = [base[torch.tensor(ids)] if k == "t" else table[torch.tensor(ids)] for k, ids in ex.parts]
        shown = "".join(s if isinstance(s, str) else "[" + s.braille + "]" for s in segs)
        return torch.cat(parts), shown

    results = json.load(open(f"{out}/result.json")) if os.path.exists(f"{out}/result.json") else {}
    base_choices = {}
    for ds, cond in jobs:
        key = f"{ds}/{cond}"
        if key in results and a.limit is None:
            continue
        if cond == "qadev":
            recs = [json.loads(l) for l in open(a.qa_dev, encoding="utf8")]
            recs = [dict(r, braille_question=r[f"g{grade}_question"], braille_choices=r.get(f"g{grade}_choices", []))
                    for r in recs if r["ds"] == ds][: a.limit]
        elif cond == "print":
            path = f"{a.inputs}/{ds}/print.jsonl"
        else:
            g, p = cond.split("_")
            assert int(g[1]) == grade, f"condition {cond} does not match grade {grade}"
            path = f"{a.inputs}/{ds}/{g}_unicode_{p}.jsonl"
        if cond != "qadev":
            recs = [json.loads(l) for l in open(path, encoding="utf8")][: a.limit]
        if a.readings:
            rd = [json.loads(l)["spans"] for l in open(f"{a.readings}/{ds}_{cond}.jsonl", encoding="utf8")]
            for r, x in zip(recs, rd):
                r["_reading"] = x
        if cond == "qadev":
            base_choices = {i: r.get("choices") for i, r in enumerate(recs)}
        elif ds == "commonsenseqa":
            base_choices = {i: json.loads(l)["choices"] for i, l in
                            enumerate(open(f"{a.inputs}/{ds}/print.jsonl", encoding="utf8"))}
        detail = f"{out}/{ds}_{cond}_details.jsonl"
        done = sum(1 for _ in open(detail, encoding="utf8")) if os.path.exists(detail) else 0
        t0 = time.time()
        for start in range(done, len(recs), a.chunk):
            idx = range(start, min(start + a.chunk, len(recs)))
            built = [prompt_embeds(ds, recs[i]) for i in idx]
            outs = llm.generate([{"prompt_embeds": e} for e, _ in built], sp, use_tqdm=False)
            with open(detail, "a", encoding="utf8") as f:
                for i, (e, shown), o in zip(idx, built, outs):
                    r, resp = recs[i], o.outputs[0].text
                    gold = bb.get_gold_answers(r, ds)
                    pred = bb.extract_answer(resp)
                    m = bb.compute_metrics(pred, gold, ds)
                    if ds == "commonsenseqa":
                        m["choice"] = choice_correct(resp, base_choices[i], gold)
                    f.write(json.dumps({"id": r.get("id"), "prompt": shown, "gold": gold, "predicted": pred,
                                        "metrics": m, "response": resp, "finish": o.outputs[0].finish_reason,
                                        "prompt_tokens": int(e.shape[0]),
                                        "output_tokens": len(o.outputs[0].token_ids)}, ensure_ascii=False) + "\n")
        rows = [json.loads(l) for l in open(detail, encoding="utf8")]
        assert len(rows) == len(recs)
        acc = {}
        for row in rows:
            for k, v in row["metrics"].items():
                acc.setdefault(k, []).append(v)
        summ = {k: round(100 * sum(v) / len(v), 2) for k, v in acc.items()}
        summ.update(n=len(rows), primary=summ[PRIMARY[ds]], truncated=sum(r["finish"] == "length" for r in rows),
                    mean_prompt_tokens=round(sum(r["prompt_tokens"] for r in rows) / len(rows), 1),
                    mean_output_tokens=round(sum(r["output_tokens"] for r in rows) / len(rows), 1),
                    gen_seconds=round(time.time() - t0, 1))
        results[key] = summ
        print(f"{key:28s} {summ['primary']:6.2f} trunc={summ['truncated']} ptok={summ['mean_prompt_tokens']}", flush=True)
        json.dump(results, open(f"{out}/result.json", "w"), indent=1)

    json.dump({"command": "python " + " ".join(sys.argv), "run": a.run, "print_embeds": a.print_embeds,
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "encoder": a.encoder, "hard": a.hard, "readings": a.readings,
               "model": a.model, "revision": a.revision, "thinking": a.thinking, "seed": a.seed,
               "decoding": {"temperature": 0.0, "max_tokens": a.max_tokens}, "vllm": vllm.__version__,
               "slurm_job": os.environ.get("SLURM_JOB_ID")}, open(f"{out}/config.json", "w"), indent=1)


if __name__ == "__main__":
    main()

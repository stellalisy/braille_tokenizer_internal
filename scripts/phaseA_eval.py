"""Phase A evaluation: stock model on print, braille (ASCII/Unicode, clean/noisy) and pipeline inputs.

Prompts, answer extraction and scoring are imported unchanged from the BrailleBench release, so
every condition uses the same prompt template and the same extraction. Decoding is greedy with
1,024 new tokens, as in BrailleBench. Inputs come from scripts/phaseA_build_inputs.py.

    python scripts/phaseA_eval.py --model Qwen/Qwen3-1.7B --out exp/phaseA/qwen3-1.7b
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

PRIMARY = {"gsm8k": "math_verify", "commonsenseqa": "em"}


def condition_config(cond):
    """Map a condition name to (BrailleBench config name, braille format)."""
    if cond in ("print", "print_clean") or cond.startswith("pipe_"):
        return "EN-EN", None
    g, fmt, _ = cond.split("_")          # e.g. g2_ascii_p5
    return f"G{g[1]}-EN", fmt


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--inputs", default="data/phaseA")
    ap.add_argument("--bb", default=os.path.expandvars("$BRL_SCRATCH/ext/braillebench"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default="gsm8k,commonsenseqa")
    ap.add_argument("--conditions", default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--thinking", action="store_true", help="Qwen3 thinking mode (off by default)")
    ap.add_argument("--no-chat", action="store_true", help="raw prompt, no chat template (diagnostic)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(a.bb, "src"))
    import evaluate_braille as bb          # BrailleBench prompts, extraction, metrics
    from vllm import LLM, SamplingParams
    import vllm

    os.makedirs(a.out, exist_ok=True)
    llm = LLM(model=a.model, revision=a.revision, dtype="bfloat16", seed=a.seed,
              max_model_len=a.max_model_len, enable_prefix_caching=True,
              gpu_memory_utilization=0.85)   # default (1.0) left no room for activations: OOM
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, seed=a.seed)

    def chat(prompt):
        if a.no_chat:
            return prompt + "\n\n"
        kw = {"enable_thinking": a.thinking} if "qwen3" in a.model.lower() else {}
        return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, **kw)

    results, input_hashes = {}, {}
    if os.path.exists(f"{a.out}/result.json"):        # resume: keep finished conditions
        results = json.load(open(f"{a.out}/result.json"))
        print("resuming; done:", sorted(results), flush=True)
    for ds in a.datasets.split(","):
        conds = sorted(f[:-6] for f in os.listdir(f"{a.inputs}/{ds}") if f.endswith(".jsonl"))
        if a.conditions != "all":
            conds = [c for c in conds if c in a.conditions.split(",")]
        for cond in conds:
            path = f"{a.inputs}/{ds}/{cond}.jsonl"
            input_hashes[f"{ds}/{cond}"] = sha256(path)
            if f"{ds}/{cond}" in results and a.limit is None:
                continue
            recs = [json.loads(l) for l in open(path, encoding="utf8")][: a.limit]
            cfg_name, fmt = condition_config(cond)
            prompts = [bb.make_braille_prompt(ds, r, bb.CONFIGS[cfg_name], fmt) for r in recs]
            t0 = time.time()
            outs = llm.generate([chat(p) for p in prompts], sp, use_tqdm=False)
            gen_s = time.time() - t0
            acc, n_trunc, n_prompt_tok = {}, 0, 0
            with open(f"{a.out}/{ds}_{cond}_details.jsonl", "w", encoding="utf8") as f:
                for r, p, o in zip(recs, prompts, outs):
                    resp = o.outputs[0].text
                    n_trunc += o.outputs[0].finish_reason == "length"
                    n_prompt_tok += len(o.prompt_token_ids)
                    gold = bb.get_gold_answers(r, ds)
                    pred = bb.extract_answer(resp)
                    m = bb.compute_metrics(pred, gold, ds)
                    for k, v in m.items():
                        acc.setdefault(k, []).append(v)
                    f.write(json.dumps({"id": r.get("id"), "prompt": p, "gold": gold, "predicted": pred,
                                        "metrics": m, "response": resp,
                                        "finish": o.outputs[0].finish_reason}, ensure_ascii=False) + "\n")
            summ = {k: round(100 * sum(v) / len(v), 2) for k, v in acc.items()}
            summ.update(n=len(recs), primary=summ[PRIMARY[ds]], truncated=n_trunc,
                        mean_prompt_tokens=round(n_prompt_tok / len(recs), 1), gen_seconds=round(gen_s, 1),
                        bb_config=cfg_name, format=fmt)
            results[f"{ds}/{cond}"] = summ
            print(f"{ds:14s} {cond:18s} {summ['primary']:6.2f}  trunc={n_trunc} "
                  f"ptok={summ['mean_prompt_tokens']} {gen_s:.0f}s", flush=True)
            json.dump(results, open(f"{a.out}/result.json", "w"), indent=1)

    import louis
    manifest = json.load(open(f"{a.inputs}/manifest.json"))
    json.dump({
        "command": "python " + " ".join(sys.argv),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model": a.model, "revision": a.revision, "thinking": a.thinking, "chat_template": not a.no_chat, "seed": a.seed,
        "decoding": {"temperature": 0.0, "max_tokens": a.max_tokens},
        "vllm": vllm.__version__, "liblouis": louis.version(),
        "braillebench_commit": subprocess.check_output(["git", "-C", a.bb, "rev-parse", "HEAD"], text=True).strip(),
        "inputs_manifest": manifest.get("git_commit"), "input_sha256": input_hashes,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }, open(f"{a.out}/config.json", "w"), indent=1)


if __name__ == "__main__":
    main()

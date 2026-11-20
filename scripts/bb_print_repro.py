"""Reproduce BrailleBench's print (EN-EN) numbers for Llama 3.1 8B.

BrailleBench's print baselines came from its July `src/evaluate.py` (commit 20c449c), run on AWS
Bedrock (`us.meta.llama3-1-8b-instruct-v1:0`), not from the released braille harness. That client
sends a hand-built Llama 3 prompt (no system block, one newline after each header) with
temperature max(0, 0.01). This script runs variants of that setup and scores each output with
both the July scorer and the released scorer.

    python scripts/bb_print_repro.py --out exp/phaseA/bb_print_repro
"""
import argparse
import json
import os
import subprocess
import sys
import types

BEDROCK = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}"
           "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--revision", default="0e9e39f249a16976918f6564b8830bc894c89659")
    ap.add_argument("--inputs", default="data/phaseA")
    ap.add_argument("--jul", default=os.path.expandvars("$BRL_SCRATCH/ext/braillebench_jul"))
    ap.add_argument("--bb", default=os.path.expandvars("$BRL_SCRATCH/ext/braillebench"))
    ap.add_argument("--datasets", default="commonsenseqa,gsm8k")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.modules["model_client"] = types.SimpleNamespace(MODELS={}, invoke_with_retry=None)
    sys.path.insert(0, a.jul)
    import evaluate_jul as J                     # July prompts, extraction, metrics
    sys.path.insert(0, os.path.join(a.bb, "src"))
    import evaluate_braille as R                 # released extraction, metrics
    from vllm import LLM, SamplingParams, TokensPrompt

    os.makedirs(a.out, exist_ok=True)
    llm = LLM(model=a.model, revision=a.revision, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()

    def as_tokens(text, extra_bos=False):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        return TokensPrompt(prompt_token_ids=([tok.bos_token_id] if extra_bos else []) + ids)

    variants = {
        # name: (prompt builder, temperature, seeds)
        "chat_greedy": (lambda p: as_tokens(tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)), 0.0, [0]),
        "bedrock_greedy": (lambda p: as_tokens(BEDROCK.format(prompt=p)), 0.0, [0]),
        "bedrock_t0.01": (lambda p: as_tokens(BEDROCK.format(prompt=p)), 0.01,
                          [int(s) for s in a.seeds.split(",")]),
        "bedrock_doublebos_greedy": (lambda p: as_tokens(BEDROCK.format(prompt=p), extra_bos=True), 0.0, [0]),
    }
    results = {}
    for ds in a.datasets.split(","):
        recs = [json.loads(l) for l in open(f"{a.inputs}/{ds}/print.jsonl", encoding="utf8")]
        jul_recs = [{"question": r["question"], "choices": " ".join(r.get("choices") or []),
                     "answer": R.get_gold_answers(r, ds)} for r in recs]
        prompts = [J.make_prompt(ds, jr) for jr in jul_recs]
        for name, (build, temp, seeds) in variants.items():
            for seed in seeds:
                sp = SamplingParams(temperature=temp, max_tokens=1024, seed=seed)
                outs = llm.generate([build(p) for p in prompts], sp, use_tqdm=False)
                acc = {"jul": [], "released": []}
                with open(f"{a.out}/{ds}_{name}_s{seed}.jsonl", "w", encoding="utf8") as f:
                    for jr, p, o in zip(jul_recs, prompts, outs):
                        resp = o.outputs[0].text
                        gold = jr["answer"]
                        key = "math_verify" if ds == "gsm8k" else "em"
                        mj = J.compute_metrics(J.extract_answer(resp), gold, ds)[key]
                        mr = R.compute_metrics(R.extract_answer(resp), gold, ds)[key]
                        acc["jul"].append(mj)
                        acc["released"].append(mr)
                        f.write(json.dumps({"prompt": p, "gold": gold, "response": resp,
                                            "jul": mj, "released": mr}, ensure_ascii=False) + "\n")
                k = f"{ds}/{name}/s{seed}"
                results[k] = {s: round(100 * sum(v) / len(v), 2) for s, v in acc.items()}
                print(k, results[k], flush=True)
                json.dump(results, open(f"{a.out}/result.json", "w"), indent=1)
    json.dump({"command": "python " + " ".join(sys.argv), "model": a.model, "revision": a.revision,
               "braillebench_jul_commit": "20c449c",
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()},
              open(f"{a.out}/config.json", "w"), indent=1)


if __name__ == "__main__":
    main()

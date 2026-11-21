"""Self-distillation targets for Phase B: the frozen model's own print answers on training questions.

GSM8K train and CommonsenseQA train items get BrailleBench's print prompt (EN-EN), the chat
template with thinking off, and greedy decoding, exactly as in the Phase A print condition. The
response becomes the target the braille input layer is trained to reproduce from the braille
prompt, so training pushes braille toward print behavior and not toward the gold answers.
Every response is kept, correct or not; responses cut off at the token limit are dropped.
Questions and choices are translated to UEB Grade 1 and 2 with liblouis. A fixed 250 items per
dataset are held out as QA-dev for tuning; the test sets are never touched.

Two steps, because the vLLM environment has no `datasets`:
    python scripts/phaseB_qa_targets.py prepare --out data/phaseB/qa     # training venv, CPU
    python scripts/phaseB_qa_targets.py generate --out data/phaseB/qa    # vLLM venv, GPU
"""
import argparse
import json
import os
import random
import subprocess
import sys

SEED = 0
N_DEV = 250


def load_items():
    from datasets import load_dataset
    items = []
    for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="train")):
        items.append({"ds": "gsm8k", "id": f"gsm8k_train_{i}", "question": r["question"].strip(),
                      "golden_answers": [r["answer"].split("####")[-1].strip()]})
    for r in load_dataset("tau/commonsense_qa", split="train"):
        ch = [f"{l}) {t}" for l, t in zip(r["choices"]["label"], r["choices"]["text"])]
        gold = r["choices"]["text"][r["choices"]["label"].index(r["answerKey"])]
        items.append({"ds": "commonsenseqa", "id": r["id"], "question": r["question"].strip(),
                      "choices": ch, "golden_answers": [gold]})
    return items


def prepare(a):
    from brltok.translate import forward
    items = load_items()[: a.limit]
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/items.jsonl", "w", encoding="utf8") as f:
        for it in items:
            for g in (1, 2):
                it[f"g{g}_question"] = forward(it["question"], g).braille
                if "choices" in it:
                    it[f"g{g}_choices"] = [forward(c, g).braille for c in it["choices"]]
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print("items", len(items))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "generate"])
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    ap.add_argument("--bb", default=os.path.expandvars("$BRL_SCRATCH/ext/braillebench"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.step == "prepare":
        return prepare(a)
    sys.path.insert(0, os.path.join(a.bb, "src"))
    import evaluate_braille as bb
    from vllm import LLM, SamplingParams

    from brltok.scoring import choice_correct
    from brltok.translate import version

    items = [json.loads(l) for l in open(f"{a.out}/items.jsonl", encoding="utf8")][: a.limit]
    llm = LLM(model=a.model, revision=a.revision, dtype="bfloat16", seed=SEED, max_model_len=a.max_tokens + 2048,
              enable_prefix_caching=True, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, seed=SEED)

    def chat(p):
        return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    prompts = [chat(bb.make_braille_prompt(it["ds"], it, bb.CONFIGS["EN-EN"], None)) for it in items]
    outs = llm.generate(prompts, sp)
    kept, dropped = [], 0
    for it, o in zip(items, outs):
        if o.outputs[0].finish_reason == "length":
            dropped += 1
            continue
        resp = o.outputs[0].text
        if it["ds"] == "gsm8k":
            correct = bool(bb.compute_metrics(bb.extract_answer(resp), it["golden_answers"], "gsm8k")["math_verify"])
        else:
            correct = bool(choice_correct(resp, it["choices"], it["golden_answers"]))
        kept.append(dict(it, response=resp, correct=correct, response_tokens=len(o.outputs[0].token_ids)))
    rng = random.Random(SEED)
    dev_ids = set()
    for ds in ("gsm8k", "commonsenseqa"):
        ids = [r["id"] for r in kept if r["ds"] == ds]
        dev_ids |= set(rng.sample(ids, min(N_DEV, len(ids))))
    os.makedirs(a.out, exist_ok=True)
    for name, rows in (("train", [r for r in kept if r["id"] not in dev_ids]),
                       ("dev", [r for r in kept if r["id"] in dev_ids])):
        with open(f"{a.out}/{name}.jsonl", "w", encoding="utf8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    acc = {ds: round(100 * sum(r["correct"] for r in kept if r["ds"] == ds) /
                     max(1, sum(r["ds"] == ds for r in kept)), 2) for ds in ("gsm8k", "commonsenseqa")}
    manifest = {"n_items": len(items), "n_kept": len(kept), "dropped_length": dropped, "n_dev": len(dev_ids),
                "target_accuracy": acc, "seed": SEED, "model": a.model, "revision": a.revision,
                "thinking": False, "max_tokens": a.max_tokens, "liblouis": version(),
                "command": "python " + " ".join(sys.argv),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    json.dump(manifest, open(f"{a.out}/manifest.json", "w"), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()

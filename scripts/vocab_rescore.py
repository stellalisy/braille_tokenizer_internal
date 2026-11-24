"""N-best rescoring reference for the print-vocabulary reader (design note revision 2, through the model).

The reader proposes its k best spellings per word (VocabEncoder.nbest, beam search). Every word whose
top reading has reader probability below tau is rescored: the frozen model scores each candidate by
the log probability of the item's whole text (spans joined by newlines, words by spaces) with that word
replaced and every other word at its top reading, and the pick maximizes
    model log prob + lam * reader log prob.
Words are rescored independently of each other. This is a reference for what the soft mix could pass
on, not a proposed method: it needs k model passes per uncertain word.

  --dev   tunes lam and tau on dev sentences (data/phaseB/dev.jsonl, those whose braille and print
          word counts match), each corrupted at --dev-rate with every profile (flip, writing, ocr).
          Reports greedy, beam top-1, oracle@k and rescored word accuracy. Writes <run>/rescore/dev.json.
  --conditions g1_p10,...   writes the chosen print ids per item and span to
          <run>/rescore/<ds>_<cond>.jsonl, read by scripts/phaseB_eval.py --readings. lam and tau come
          from <run>/rescore/dev.json unless given.

    python scripts/vocab_rescore.py --run exp/vocab/v2/g1_scratch_local_nonw_mix --dev
    python scripts/vocab_rescore.py --run exp/vocab/v2/g1_scratch_local_nonw_mix --conditions g1_p10,g1_w10,g1_o10
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys

LAMS = [0.0, 0.5, 1.0, 2.0, 4.0]
TAUS = [0.5, 0.9, 0.99]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--encoder", default=None, help="default <run>/encoder.pt")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--n-dev", type=int, default=400)
    ap.add_argument("--dev-rate", type=float, default=0.10)
    ap.add_argument("--conditions", default="")
    ap.add_argument("--datasets", default="gsm8k,commonsenseqa")
    ap.add_argument("--inputs", default="data/v2inputs")
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--bb", default=os.path.expandvars("$BRL_SCRATCH/ext/braillebench"))
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = f"{a.run}/rescore"
    os.makedirs(out, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM
    from vllm import LLM, SamplingParams

    from brltok.cells import normalize
    from brltok.noise import braille_noise, flip_dots
    from brltok.vocab_encoder import load_encoder, span_words

    grade = json.load(open(f"{a.run}/train_log.json"))["args"]["grade"]
    hf = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=torch.bfloat16)
    base = hf.get_input_embeddings().weight.detach().clone()
    del hf
    enc = load_encoder(a.encoder or f"{a.run}/encoder.pt", base.float(), device="cuda")
    llm = LLM(model=a.model, revision=a.revision, dtype="bfloat16", seed=a.seed, max_model_len=4096,
              gpu_memory_utilization=0.35)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
    prefix = tok("Text:\n", add_special_tokens=False)["input_ids"]

    def read(items):
        """items: list of lists of spans -> per item, per span, per word: [(ids, reader logp, string)]."""
        flat = [s for it in items for s in it]
        cands, j = [], 0
        while j < len(flat):          # batches of about 128 words: beam logits are [words * k, vocab]
            n, e = 0, j
            while e < len(flat) and (e == j or n + len(span_words(flat[e])) <= 128):
                n += len(span_words(flat[e]))
                e += 1
            cands.extend(enc.nbest(flat[j:e], k=a.k))
            torch.cuda.empty_cache()      # vLLM shares the GPU
            j = e
        res, c = [], 0
        for it in items:
            # a word whose beam never reached END within max_tokens gets one empty reading
            res.append([[[(ids, lp, tok.decode(ids).strip()) for ids, lp in w] or [([], float("-inf"), "")]
                         for w in cands[c + n]]
                        for n in range(len(it))])
            c += len(it)
        return res

    def text_of(words_by_span):
        return "\n".join(" ".join(ws) for ws in words_by_span)

    def score_texts(texts):
        uniq = sorted(set(texts))
        prompts = [{"prompt_token_ids": prefix + tok(t, add_special_tokens=False)["input_ids"]} for t in uniq]
        res = {}
        for j in range(0, len(prompts), 4096):
            outs = llm.generate(prompts[j: j + 4096], sp, use_tqdm=False)
            for t, p, o in zip(uniq[j: j + 4096], prompts[j: j + 4096], outs):
                ids = p["prompt_token_ids"]
                res[t] = sum(o.prompt_logprobs[q][ids[q]].logprob for q in range(len(prefix), len(ids)))
        return res

    def rescore(cands, max_tau):
        """Score every candidate of every word below max_tau. Returns per item a list of
        (span, word, [(model logp, reader logp)]) for the rescored words."""
        jobs, texts = [], []
        for ii, item in enumerate(cands):
            top = [[w[0][2] for w in span] for span in item]
            for si, span in enumerate(item):
                for wi, w in enumerate(span):
                    if not w[0][1] < math.log(max_tau) or len(w) < 2:
                        continue
                    vts = []
                    for _, _, s in w:
                        v = [list(x) for x in top]
                        v[si][wi] = s
                        vts.append(text_of(v))
                    jobs.append((ii, si, wi, vts))
                    texts.extend(vts)
        sc = score_texts(texts)
        per = [[] for _ in cands]
        for ii, si, wi, vts in jobs:
            per[ii].append((si, wi, [(sc[t], c[1]) for t, c in zip(vts, cands[ii][si][wi])]))
        return per

    def choose(cands, per, lam, tau):
        """Chosen candidate index per item, span, word."""
        pick = [[[0] * len(span) for span in item] for item in cands]
        for ii, lst in enumerate(per):
            for si, wi, sc in lst:
                if not cands[ii][si][wi][0][1] < math.log(tau):
                    continue
                pick[ii][si][wi] = max(range(len(sc)), key=lambda c: sc[c][0] + lam * sc[c][1])
        return pick

    meta = {"command": "python " + " ".join(sys.argv),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "k": a.k, "model": a.model, "revision": a.revision, "slurm_job": os.environ.get("SLURM_JOB_ID")}

    if a.dev:
        rows = []
        for line in open("data/phaseB/dev.jsonl", encoding="utf8"):
            r = json.loads(line)
            b = normalize(r[f"g{grade}"])
            if len(span_words(b)) == len(r["text"].split()):
                rows.append((b, r["text"].split()))
            if len(rows) >= a.n_dev:
                break
        report = {}
        for prof in ("flip", "writing", "ocr"):
            items, golds = [], []
            for i, (b, gw) in enumerate(rows):
                rng = random.Random(f"rescore-dev|{prof}|{i}")
                nb = flip_dots(b, a.dev_rate, rng) if prof == "flip" else braille_noise(b, a.dev_rate, rng, prof)
                items.append([nb])
                golds.append(gw)
            cands = read(items)
            greedy = [x for j in range(0, len(items), 16) for x in enc.decode([it[0] for it in items[j: j + 16]])]
            n = sum(len(g) for g in golds)
            gacc = sum(tok.decode(ids).strip() == w for (_, idl, _, lens), gw in zip(greedy, golds)
                       for ids, w in zip(_split(idl, lens), gw))
            top1 = sum(c[0][2] == w for item, gw in zip(cands, golds) for c, w in zip(item[0], gw))
            orac = sum(any(x[2] == w for x in c) for item, gw in zip(cands, golds) for c, w in zip(item[0], gw))
            per = rescore(cands, max(TAUS))
            grid = {}
            for lam in LAMS:
                for tau in TAUS:
                    pick = choose(cands, per, lam, tau)
                    ok = sum(item[0][wi][p][2] == w for item, pk, gw in zip(cands, pick, golds)
                             for wi, (p, w) in enumerate(zip(pk[0], gw)))
                    grid[f"lam{lam}_tau{tau}"] = round(100 * ok / n, 2)
            report[prof] = {"words": n, "greedy": round(100 * gacc / n, 2), "beam_top1": round(100 * top1 / n, 2),
                            f"oracle_at_{a.k}": round(100 * orac / n, 2), "rescored": grid,
                            "rescored_words": sum(len(p) for p in per)}
            print(prof, json.dumps(report[prof]), flush=True)
        mean = {key: sum(report[p]["rescored"][key] for p in report) / len(report) for key in report["flip"]["rescored"]}
        best = max(mean, key=mean.get)
        lam, tau = float(best.split("_")[0][3:]), float(best.split("_")[1][3:])
        json.dump({"profiles": report, "mean": mean, "best": {"lam": lam, "tau": tau}, "rate": a.dev_rate,
                   "n_sentences": len(rows), **meta}, open(f"{out}/dev.json", "w"), indent=1)
        print("best", best, round(mean[best], 2), flush=True)

    if a.conditions:
        sys.path.insert(0, os.path.join(a.bb, "src"))
        import evaluate_braille as bb
        from brltok.splice import Braille, qa_segments
        from brltok.vocab_encoder import build_vocab

        best = json.load(open(f"{out}/dev.json"))["best"] if os.path.exists(f"{out}/dev.json") else {}
        lam = a.lam if a.lam is not None else best["lam"]
        tau = a.tau if a.tau is not None else best["tau"]

        def chat(p):
            return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)

        for ds in a.datasets.split(","):
            for cond in a.conditions.split(","):
                g, p = cond.split("_")
                assert int(g[1]) == grade
                recs = [json.loads(l) for l in open(f"{a.inputs}/{ds}/{g}_unicode_{p}.jsonl", encoding="utf8")]
                items = []
                for r in recs:
                    ch = [Braille(c) for c in r.get("braille_choices", [])] if ds == "commonsenseqa" else []
                    segs = qa_segments(bb, ds, grade, Braille(r["braille_question"]), ch, chat=chat)
                    items.append(build_vocab(segs, tok).spans)
                cands = read(items)
                per = rescore(cands, tau)
                pick = choose(cands, per, lam, tau)
                changed = 0
                with open(f"{out}/{ds}_{cond}.jsonl", "w", encoding="utf8") as f:
                    for item, pk in zip(cands, pick):
                        spans = []
                        for span, ps in zip(item, pk):
                            ids = []
                            for w, q in zip(span, ps):
                                ids.extend(w[q][0])
                                changed += q != 0
                            spans.append(ids)
                        f.write(json.dumps({"spans": spans}) + "\n")
                nw = sum(len(s) for it in cands for s in it)
                print(f"{ds}/{cond}: {len(recs)} items, {nw} words, {sum(len(x) for x in per)} rescored, "
                      f"{changed} changed from the reader's top reading", flush=True)
                json.dump({"lam": lam, "tau": tau, "words": nw, "rescored": sum(len(x) for x in per),
                           "changed": changed, **meta}, open(f"{out}/{ds}_{cond}.meta.json", "w"), indent=1)


def _split(ids, lens):
    out, c = [], 0
    for n in lens:
        out.append(ids[c: c + n])
        c += n
    return out


if __name__ == "__main__":
    main()

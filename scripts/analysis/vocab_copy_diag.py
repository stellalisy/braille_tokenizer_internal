"""Can the print-vocabulary encoder copy letter strings it has never seen as words?

Probe sets (all braille from liblouis forward translation, clean):
  real-heldout : real words held out from Stage A training (rare but well-formed English)
  typo         : a common training word with one random letter edit (substitute, insert, delete or swap
                 neighbours) that is not itself a known word; the encoder should spell the typo
  random       : random lowercase letter strings, length 4 to 9
Each set is read isolated (the word alone) and in context (replacing one word of a dev sentence).
For each read we record: exact copy, "snapped" (output is the original word for typos, or any known word),
and the teacher-forced log probability of the gold spelling vs the greedy output.

    python scripts/analysis/vocab_copy_diag.py --run exp/vocab/g1_a_s0
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from brltok import translate
    from brltok.cells import normalize
    from brltok.vocab_encoder import load_encoder, span_words, word_targets

    g = json.load(open(f"{a.run}/train_log.json"))["args"]["grade"]
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    m = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=torch.bfloat16)
    emb = m.get_input_embeddings().weight.detach().float()
    del m
    enc = load_encoder(f"{a.run}/encoder.pt", emb)
    enc.eval()
    rng = random.Random(a.seed)

    held = set(json.load(open(f"{a.run}/heldout_words.json")))
    known, freq = set(), Counter()
    for line in open("data/phaseB/train.jsonl", encoding="utf8"):
        ws = re.findall(r"[a-z]+", json.loads(line)["text"].lower())
        known.update(ws)
        freq.update(ws)
    vocab_words = [json.loads(l)["text"] for l in open(f"data/vocab/words_g{g}.jsonl", encoding="utf8")]
    known.update(w.lower() for w in vocab_words)
    common = [w for w, c in freq.most_common(20000) if len(w) >= 4 and w.isalpha() and w not in held]

    def typo(w):
        for _ in range(50):
            i = rng.randrange(len(w))
            op = rng.choice("sids")
            c = chr(97 + rng.randrange(26))
            if op == "s":
                t = w[:i] + c + w[i + 1:]
            elif op == "i":
                t = w[:i] + c + w[i:]
            elif op == "d":
                t = w[:i] + w[i + 1:]
            else:
                if i == len(w) - 1:
                    continue
                t = w[:i] + w[i + 1] + w[i] + w[i + 2:]
            if t != w and t not in known and len(t) >= 3:
                return t
        return None

    sets = {}
    hw = sorted(w for w in held if w.isalpha() and w.islower())
    sets["real-heldout"] = [(w, w) for w in rng.sample(hw, min(a.n, len(hw)))]
    ty = []
    while len(ty) < a.n:
        w = rng.choice(common)
        t = typo(w)
        if t:
            ty.append((t, w))
    sets["typo"] = ty
    rd = []
    while len(rd) < a.n:
        t = "".join(chr(97 + rng.randrange(26)) for _ in range(rng.randint(4, 9)))
        if t not in known:
            rd.append((t, None))
    sets["random"] = rd

    dev = []
    for line in open("data/phaseB/dev.jsonl", encoding="utf8"):
        t = json.loads(line)["text"]
        ws = t.split()
        if 6 <= len(ws) <= 25:
            dev.append(ws)
        if len(dev) >= 3000:
            break

    def braille(text):
        return normalize(translate.forward_string(text, g))

    @torch.no_grad()
    def read(spans):
        out = []
        for k in range(0, len(spans), 128):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                dec = enc.decode(spans[k: k + 128])
            for _, ids, conf, lens in dec:
                ws, c = [], 0
                for n in lens:
                    ws.append((ids[c: c + n], min(conf[c: c + n]) if n else 0.0))
                    c += n
                out.append(ws)
        return out

    @torch.no_grad()
    def seq_logp(spans, word_index, word_ids_per_span):
        """log p(given token ids) for one word per span, teacher-forced; word_ids_per_span gives all words."""
        res = []
        for k in range(0, len(spans), 128):
            sp = spans[k: k + 128]
            wi = [w for x in word_ids_per_span[k: k + 128] for w in x]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                H = enc.encode(sp)
                mem, mpad, words = enc.word_memory(H, sp)
                prev, lab = enc.targets(wi)
                lg = enc.logits(mem, mpad, prev).float()
            lp = F.log_softmax(lg, -1)
            tl = torch.where(lab < 0, torch.zeros_like(lab), lab)
            tok_lp = lp.gather(-1, tl[..., None])[..., 0] * (lab >= 0)
            wl = tok_lp.sum(-1)
            # pick the probed word of each span
            pos, c = [], 0
            for j, s in enumerate(sp):
                pos.append(c + word_index[k + j])
                c += len(span_words(s))
            res.extend(wl[pos].tolist())
        return res

    report = {}
    examples = defaultdict(list)
    for name, items in sets.items():
        for mode in ("isolated", "context"):
            spans, idx, texts_words, golds = [], [], [], []
            for t, orig in items:
                if mode == "isolated":
                    words = [t]
                    j = 0
                else:
                    ws = list(rng.choice(dev))
                    cand = [i for i, w in enumerate(ws) if w.isalpha() and w.islower() and len(w) >= 3]
                    if not cand:
                        continue
                    j = rng.choice(cand)
                    ws[j] = t
                    words = ws
                b = braille(" ".join(words))
                if len(span_words(b)) != len(words):
                    continue
                spans.append(b)
                idx.append(j)
                texts_words.append(words)
                golds.append((t, orig))
            reads = read(spans)
            got, conf = [], []
            for r, j in zip(reads, idx):
                got.append(tok.decode(r[j][0]).strip())
                conf.append(r[j][1])
            gold_ids = [[word_targets(tok, " ".join(ws))[i] for i in range(len(ws))] for ws in texts_words]
            pred_ids = []
            for gi, r, j in zip(gold_ids, reads, idx):
                x = list(gi)
                x[j] = r[j][0] if r[j][0] else gi[j]
                pred_ids.append(x)
            lp_gold = seq_logp(spans, idx, gold_ids)
            lp_pred = seq_logp(spans, idx, pred_ids)
            n = len(got)
            exact = sum(g_ == t for g_, (t, _) in zip(got, golds))
            snapped_orig = sum(orig is not None and orig != t and g_.lower() == orig for g_, (t, orig) in zip(got, golds))
            known_word = sum(g_ != t and g_.lower() in known for g_, (t, _) in zip(got, golds))
            wrong = [(lg_, lp_) for g_, (t, _), lg_, lp_ in zip(got, golds, lp_gold, lp_pred) if g_ != t]
            report[f"{name}/{mode}"] = {
                "n": n, "exact": round(100 * exact / n, 1),
                "snapped_to_original": round(100 * snapped_orig / n, 1) if name == "typo" else None,
                "wrong_is_known_word": round(100 * known_word / n, 1),
                "mean_conf_right": round(sum(c for c, g_, (t, _) in zip(conf, got, golds) if g_ == t) / max(1, exact), 3),
                "mean_conf_wrong": round(sum(c for c, g_, (t, _) in zip(conf, got, golds) if g_ != t) / max(1, n - exact), 3),
                "wrong_mean_logp_gold": round(sum(x for x, _ in wrong) / max(1, len(wrong)), 2),
                "wrong_mean_logp_pred": round(sum(y for _, y in wrong) / max(1, len(wrong)), 2),
            }
            for g_, (t, orig) in list(zip(got, golds))[:400]:
                if g_ != t and len(examples[f"{name}/{mode}"]) < 15:
                    examples[f"{name}/{mode}"].append(f"{t}{'(' + orig + ')' if orig and orig != t else ''}->{g_}")
            print(name, mode, report[f"{name}/{mode}"], flush=True)
            print("   ", "; ".join(examples[f"{name}/{mode}"]), flush=True)
    json.dump({"report": report, "examples": examples, "args": vars(a)},
              open(f"{a.run}/copy_diag.json", "w"), indent=1)


if __name__ == "__main__":
    main()

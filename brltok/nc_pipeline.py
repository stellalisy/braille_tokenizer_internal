"""Glue for the noisy-channel reader (docs/design_noisy_channel.md): builds candidate lists with all
word-level scores, and wraps the braille prior and any causal LM as scorers for brltok.nc_decode.

    pipe = Pipeline("exp/nc/g1")                     # reader/, translator/, wordprior/, cell_lm/
    items = pipe.build([[span, ...], ...], {"mix10": Channel("mix", 0.1)})
    use_channel(items, "mix10")
    picks, _ = decode(items, Weights(...), pipe.braille_score, llm_scorer("Qwen/Qwen3-1.7B"))
"""
from __future__ import annotations

import math

import torch

from .channel import logsumexp
from .nc_decode import Cand, Word
from .seq2seq import CharVocab, cell_input, cells_to_str


class Pipeline:
    def __init__(self, root: str, device: str = "cuda", k: int = 16, n_brailings: int = 3, tau: float = 0.99,
                 models=None):
        from . import cell_lm, seq2seq
        if models is None:
            models = (seq2seq.load(f"{root}/reader/model.pt", device), seq2seq.load(f"{root}/translator/model.pt", device),
                      seq2seq.load(f"{root}/wordprior/model.pt", device), cell_lm.load(f"{root}/cell_lm/model.pt", device))
        (self.reader, meta), (self.translator, _), (self.wordprior, _), self.cell_lm = models
        self.cv = CharVocab(meta["chars"])
        self.grade = meta.get("grade")
        self.k, self.nb, self.tau = k, n_brailings, tau

    def _ac(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.reader.device.type == "cuda")

    def build(self, items: list[list[str]], channels: dict) -> list:
        """items: per item, its braille spans (normalized) -> per item, per span, a list of Words.
        Words whose top reading has probability >= tau keep only that reading (rescore=False)."""
        words = [w for it in items for sp in it for w in sp.split()]
        with self._ac():
            nb = self.reader.nbest([cell_input(w) for w in words], k=self.k)
        cand_lists = []
        for w, lst in zip(words, nb):
            cs = [(self.cv.decode(s), lp) for s, lp in lst] or [("", -math.inf)]
            seen, uniq = set(), []
            for y, lp in cs:
                if y not in seen:
                    seen.add(y)
                    uniq.append((y, lp))
            rescore = not math.exp(uniq[0][1]) >= self.tau
            cand_lists.append((w, uniq if rescore else uniq[:1], rescore))
        ys = sorted({y for _, cs, _ in cand_lists for y, _ in cs})
        with self._ac():
            wp = dict(zip(ys, self.wordprior.score([[0]] * len(ys), [self.cv.encode(y) for y in ys])))
            tr = self.translator.nbest([self.cv.encode(y) or [self.cv.unk] for y in ys], k=self.nb)
        brl = {y: [(cells_to_str(c), lp) for c, lp in lst] for y, lst in zip(ys, tr)}
        out, j = [], 0
        for it in items:
            spans = []
            for sp in it:
                ws = []
                for _ in sp.split():
                    b, cs, rescore = cand_lists[j]
                    j += 1
                    cands = []
                    for y, lp in cs:
                        c = Cand(y, lp, wp[y])
                        for key, ch in channels.items():
                            tot = [ch.logp(b, cb) + lt for cb, lt in brl[y]]
                            best = max(range(len(tot)), key=lambda q: tot[q]) if tot else None
                            c.chains[key] = (logsumexp(tot), brl[y][best][0] if best is not None else "")
                        cands.append(c)
                    ws.append(Word(b, cands, rescore))
                spans.append(ws)
            out.append(spans)
        return out

    def braille_score(self, texts):
        return self.cell_lm.score(texts)


def use_channel(items, key: str):
    """Set every candidate's chain score and intended braille from channel `key`."""
    for it in items:
        for sp in it:
            for w in sp:
                for c in w.cands:
                    c.chain, c.c = c.chains[key]


class LLMScorer:
    """log P(text | "Text:\\n") under a causal LM in vLLM, summed over the text's tokens."""

    def __init__(self, model: str, revision=None, mem: float = 0.35, seed: int = 0):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model, revision=revision, dtype="bfloat16", seed=seed, max_model_len=4096,
                       gpu_memory_utilization=mem)
        self.tok = self.llm.get_tokenizer()
        self.sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
        self.prefix = self.tok("Text:\n", add_special_tokens=True)["input_ids"]
        self.cache = {}

    def __call__(self, texts):
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        prompts = [{"prompt_token_ids": self.prefix + self.tok(t, add_special_tokens=False)["input_ids"]} for t in todo]
        for j in range(0, len(prompts), 4096):
            outs = self.llm.generate(prompts[j: j + 4096], self.sp, use_tqdm=False)
            for t, p, o in zip(todo[j: j + 4096], prompts[j: j + 4096], outs):
                ids = p["prompt_token_ids"]
                self.cache[t] = sum(o.prompt_logprobs[q][ids[q]].logprob for q in range(len(self.prefix), len(ids)))
        return [self.cache[t] for t in texts]

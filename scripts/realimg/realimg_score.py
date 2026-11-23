"""Read real-photo braille (OBR cell strings from scripts/realimg/obr_detect.py) with the print-vocabulary
encoder and with liblouis back-translation, and score both against the print label of each image.

Inputs: the OBR jsonl and one Stage A run per grade. Grade 1 images go to the Grade 1 encoder and table,
Grade 2 images to Grade 2. The book set has no grade label; each of its images is assigned the grade whose
liblouis forward translation of the label is closer (cell edit distance) to the detected cells, and that
assignment is reported.

Metrics (after lowercasing and keeping only letters, digits and apostrophes in each word):
  word accuracy  = share of label words matched in a word-level alignment (difflib) of read vs label;
  CER            = character edit distance / label length.
Also reported: the OBR cell error rate against liblouis forward translation of the label (an estimate of
detector noise; real transcriptions can differ from liblouis, so it is an upper bound on detector error),
and the same two readers on that liblouis forward translation (the clean reference).
"""
import argparse
import difflib
import json
import os
import re
import sys

import torch

sys.path.insert(0, ".")
from brltok import translate  # noqa: E402
from brltok.cells import normalize  # noqa: E402
from brltok.vocab_encoder import VocabEncoder, load_encoder  # noqa: E402


def words(s: str) -> list[str]:
    out = []
    for w in s.lower().split():
        w = re.sub(r"[^a-z0-9']", "", w)
        if w:
            out.append(w)
    return out


def edit(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def score(read: str, gold: str) -> tuple[int, int, int, int]:
    r, g = words(read), words(gold)
    m = sum(bl.size for bl in difflib.SequenceMatcher(None, r, g, autojunk=False).get_matching_blocks())
    rc, gc = " ".join(r), " ".join(g)
    return m, len(g), edit(rc, gc), len(gc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obr", required=True)
    ap.add_argument("--run1", required=True)
    ap.add_argument("--run2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    m = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=torch.bfloat16)
    emb = m.get_input_embeddings().weight.detach().float()
    del m
    encs = {}
    for g, run in ((1, a.run1), (2, a.run2)):
        if os.path.exists(f"{run}/encoder.pt"):
            encs[g] = load_encoder(f"{run}/encoder.pt", emb, device="cuda")
            src = "encoder.pt"
        else:  # mid-training checkpoint
            allowed = torch.tensor(json.load(open("data/vocab/allowed_ids.json")))
            st = torch.load(f"{run}/ckpt.pt", map_location="cuda", weights_only=False)
            encs[g] = VocabEncoder(allowed, emb, d=512, enc_layers=4, dec_layers=2).cuda()
            encs[g].load_state_dict(st["model"], strict=False)
            src = f"ckpt step {st['step']}"
        encs[g].eval()
        print(f"grade {g}: {run} ({src})", flush=True)

    def enc_read(g, braille):
        if not braille.strip():
            return ""
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, ids, _, lens = encs[g].decode([braille])[0]
        out, c = [], 0
        for n in lens:
            out.append(tok.decode(ids[c: c + n]).strip())
            c += n
        return " ".join(out)

    def lou_read(g, braille):
        try:
            return translate.back(braille, g) if braille.strip() else ""
        except Exception:
            return ""

    rows = [json.loads(l) for l in open(a.obr, encoding="utf8")]
    tot = {}
    recs = []
    for r in rows:
        obr = normalize(r["obr"])
        fw = {g: normalize(translate.forward_string(r["gt"], g)) for g in (1, 2)}
        g = r["grade"]
        if g is None:
            g = min((1, 2), key=lambda k: edit(obr, fw[k]))
            r["grade_assigned"] = g
        rec = {**r, "grade_used": g, "forward": fw[g], "cell_edits": edit(obr, fw[g]), "cells_ref": len(fw[g])}
        for src, braille in (("obr", obr), ("clean", fw[g])):
            for reader, fn in (("encoder", enc_read), ("liblouis", lou_read)):
                txt = fn(g, braille)
                rec[f"{src}_{reader}"] = txt
                s = score(txt, r["gt"])
                for key in (f"g{g}/{src}/{reader}", f"all/{src}/{reader}"):
                    t = tot.setdefault(key, [0, 0, 0, 0])
                    for i in range(4):
                        t[i] += s[i]
        for key in (f"g{g}/cells", "all/cells"):
            t = tot.setdefault(key, [0, 0, 0, 0])
            t[0] += rec["cell_edits"]
            t[1] += rec["cells_ref"]
            t[2] += 1
        recs.append(rec)
    summ = {}
    for k, t in sorted(tot.items()):
        if k.endswith("/cells"):
            summ[k] = {"cell_error_vs_liblouis_forward": round(100 * t[0] / max(1, t[1]), 1), "images": t[2]}
        else:
            summ[k] = {"word_acc": round(100 * t[0] / max(1, t[1]), 1), "cer": round(100 * t[2] / max(1, t[3]), 1),
                       "label_words": t[1]}
    for k, v in summ.items():
        print(k, v)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summ, "rows": recs, "args": vars(a), "liblouis": translate.version()},
              open(a.out, "w", encoding="utf8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

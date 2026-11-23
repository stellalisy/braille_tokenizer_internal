"""Greedy rollouts from a Stage A checkpoint: held-out words, dev sentences, liblouis back-translation side by side."""
import argparse, json, random, re, sys
from collections import Counter
import torch
sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer
from brltok.cells import normalize
from brltok.noise import flip_dots
from brltok.vocab_encoder import VocabEncoder, span_words
from brltok import translate

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--grade", type=int, required=True)
a = ap.parse_args()
M, R = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
tok = AutoTokenizer.from_pretrained(M, revision=R)
m = AutoModelForCausalLM.from_pretrained(M, revision=R, torch_dtype=torch.bfloat16)
emb = m.get_input_embeddings().weight.detach().float(); del m
allowed = torch.tensor(json.load(open("data/vocab/allowed_ids.json")))
enc = VocabEncoder(allowed, emb, d=512, enc_layers=4, dec_layers=2).cuda()
st = torch.load(f"{a.run}/ckpt.pt", map_location="cuda", weights_only=False)
enc.load_state_dict(st["model"], strict=False); enc.eval()
print("checkpoint step", st["step"])
g = a.grade
def back(b):
    try:
        return translate.back(b, g)
    except Exception as e:
        return f"<err {e}>"

def read(spans, hard=False):
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for k in range(0, len(spans), 128):
            for _, ids, conf, lens in enc.decode(spans[k:k+128], hard):
                words, c = [], 0
                for n in lens:
                    words.append((tok.decode(ids[c:c+n]).strip(), min(conf[c:c+n])))
                    c += n
                out.append(words)
    return out

held = set(json.load(open(f"{a.run}/heldout_words.json")))
words = [json.loads(l) for l in open(f"data/vocab/words_g{g}.jsonl")]
wdv = [(normalize(w["braille"]), w["text"]) for w in words if w["text"].lower() in held][:2000]
rng = random.Random(0)
for p in (0.0, 0.05):
    spans = [flip_dots(b, p, rng) if p else b for b, _ in wdv]
    pred = read(spans)
    ok = [pw[0][0] == t if len(pw) == 1 else False for pw, (_, t) in zip(pred, wdv)]
    lou = [back(s).strip() == t for s, (_, t) in zip(spans, wdv)]
    lou_ci = [back(s).strip().lower() == t.lower() for s, (_, t) in zip(spans, wdv)]
    print(f"\nHELD-OUT WORDS p={p}: greedy exact {100*sum(ok)/len(ok):.1f}%  liblouis exact {100*sum(lou)/len(lou):.1f}% (case-insens {100*sum(lou_ci)/len(lou_ci):.1f}%)")
    if p == 0:
        errs = [(t, " ".join(w for w, _ in pw), s, back(s)) for pw, (b, t), s, k in zip(pred, wdv, spans, ok) if not k]
        rng2 = random.Random(1); rng2.shuffle(errs)
        print("  sample errors (gold | encoder | braille | liblouis):")
        for e in errs[:30]:
            print("   ", " | ".join(e))
        # character edit length of errors
        import difflib
        d = Counter()
        for t, pr, _, _ in errs:
            r = difflib.SequenceMatcher(None, t.lower(), pr.lower()).ratio()
            d["near(>=.8)" if r >= .8 else "mid" if r >= .5 else "far"] += 1
        print("  error similarity:", dict(d))
        rightc = [min(c for _, c in pw) for pw, k in zip(pred, ok) if k]
        wrongc = [min(c for _, c in pw) for pw, k in zip(pred, ok) if not k]
        print(f"  min token conf: right {sum(rightc)/len(rightc):.2f}, wrong {sum(wrongc)/max(1,len(wrongc)):.2f}")

dv = []
for line in open("data/phaseB/dev.jsonl"):
    r = json.loads(line); b = normalize(r[f"g{g}"])
    if len(span_words(b)) == len(r["text"].split()):
        dv.append((b, r["text"]))
    if len(dv) >= 300: break
pred = read([b for b, _ in dv])
ok = n = 0
for pw, (_, t) in zip(pred, dv):
    for (w, _), gt in zip(pw, t.split()):
        ok += w == gt; n += 1
print(f"\nDEV SENTENCES greedy word exact {100*ok/n:.1f}% over {n} words")
for pw, (b, t) in list(zip(pred, dv))[:8]:
    print("  GOLD:", t)
    print("  ENC :", " ".join(w for w, _ in pw))
    print("  LOU :", back(b))
    print()

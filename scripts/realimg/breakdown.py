import json, sys, difflib, collections
sys.path.insert(0, "scripts/realimg")
from realimg_score import score
d = json.load(open(sys.argv[1]))
rows = d["rows"]
bk = collections.OrderedDict((k, [0]*5) for k in ["<10%", "10-25%", "25-50%", ">=50%"])
for r in rows:
    e = r["cell_edits"] / max(1, r["cells_ref"])
    k = "<10%" if e < .1 else "10-25%" if e < .25 else "25-50%" if e < .5 else ">=50%"
    b = bk[k]; b[0] += 1
    me, n, _, _ = score(r["obr_encoder"], r["gt"]); ml, _, _, _ = score(r["obr_liblouis"], r["gt"])
    b[1] += me; b[2] += ml; b[3] += n
print("cell-error bucket | images | label words | encoder word acc | liblouis word acc")
for k, b in bk.items():
    print(f"{k:7s} | {b[0]:3d} | {b[3]:4d} | {100*b[1]/max(1,b[3]):5.1f} | {100*b[2]/max(1,b[3]):5.1f}")
# error ops on cells
ops = collections.Counter()
for r in rows:
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, r["forward"], r["obr"], autojunk=False).get_opcodes():
        if tag == "equal": continue
        a, b = r["forward"][i1:i2], r["obr"][j1:j2]
        if tag == "replace" and i2-i1 == j2-j1:
            for x, y in zip(a, b):
                kind = "space->cell" if x == " " else "cell->space" if y == " " else ("1-dot sub" if bin((ord(x)-0x2800) ^ (ord(y)-0x2800)).count("1") == 1 else "multi-dot sub")
                ops[kind] += 1
        else:
            ops["insert" if tag == "insert" else "delete" if tag == "delete" else "unequal replace"] += max(i2-i1, j2-j1)
print(dict(ops.most_common()))
import random; random.seed(3)
for r in random.sample([r for r in rows if r["cell_edits"]/max(1,r["cells_ref"]) < .25], 10):
    print(repr(r["gt"][:50]), "| ref", r["forward"][:40], "| obr", r["obr"][:40], "| enc:", r["obr_encoder"][:45], "| lou:", r["obr_liblouis"][:45])

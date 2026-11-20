#!/bin/bash -l
#SBATCH -A argon
#SBATCH -p gpu-h200
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -J brl-check
#SBATCH -o /gpfs/scrubbed/stelli/braille/logs/check-%j.out
# Setup steps 6 and 7: tests, data rebuild, liblouis baseline re-check.
set -euo pipefail
source /gpfs/projects/stf/stelli/braille_tokenizer_internal/scripts/env.sh
cd "$BRL_ROOT"
mkdir -p "$BRL_SCRATCH/data"
[ -e data ] || ln -s "$BRL_SCRATCH/data" data

echo "== pytest"
python -m pytest tests -rs -q 2>&1 | tail -15

echo "== build_data"
time python scripts/build_data.py --out data --test-sentences test_sentences.json
wc -l data/*.jsonl
sha256sum data/test.jsonl

echo "== eval_liblouis"
OUT=exp/setup/liblouis_rerun
mkdir -p $OUT
python scripts/eval_liblouis.py --data data --out $OUT
python - <<EOF
import json, subprocess, nltk, louis
cfg = {"command": "python scripts/eval_liblouis.py --data data --out $OUT",
       "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
       "liblouis": louis.version(), "nltk": nltk.__version__,
       "data": "scripts/build_data.py --out data --test-sentences test_sentences.json (seed 0)",
       "slurm_job": "$SLURM_JOB_ID"}
json.dump(cfg, open("$OUT/config.json", "w"), indent=1)
EOF
echo "== diff against shipped runs/liblouis"
python - <<'EOF'
import json
a = json.load(open("runs/liblouis/result.json")); b = json.load(open("exp/setup/liblouis_rerun/result.json"))
print("shipped", a["liblouis"], a["test"]); print("rerun  ", b["liblouis"], b["test"])
print("IDENTICAL" if a == b else "DIFFERENT")
EOF
echo DONE

#!/bin/bash -l
#SBATCH -A argon
#SBATCH -p gpu-h200
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH -J brl-toy-seeds
#SBATCH -o /gpfs/scrubbed/stelli/braille/logs/toy-seeds-%j.out
# Low-priority background: toy E1 (table) vs E3 (dots) under 5% training flips, five seeds each.
set -euo pipefail
source /gpfs/projects/stf/stelli/braille_tokenizer_internal/scripts/env.sh
cd "$BRL_ROOT"
OUT=exp/toy/noise_seeds
mkdir -p $OUT
for embed in table dots; do
  for seed in 0 1 2 3 4; do
    d=$OUT/cell_${embed}_noise_s$seed
    mkdir -p $d
    cmd="python scripts/train_backtranslate.py --tok cell --embed $embed --train-noise 0.05 --steps 4000 --seed $seed --threads 1 --out $d"
    printf '{"command": "%s", "git_commit": "%s", "liblouis": "%s", "data": "scripts/build_data.py --out data --test-sentences test_sentences.json (seed 0)", "slurm_job": "%s"}\n' \
      "$cmd" "$(git rev-parse HEAD)" "$LIBLOUIS_VERSION" "$SLURM_JOB_ID" > $d/config.json
    $cmd > $d/train.log 2>&1 &
  done
done
wait
python - <<'EOF'
import json, glob, statistics as st
rows = {}
for f in sorted(glob.glob("exp/toy/noise_seeds/*/result.json")):
    r = json.load(open(f)); e = r["args"]["embed"]
    rows.setdefault(e, []).append(r)
print(json.dumps({e: [ {k: v for k, v in r.items() if k in ("test", "test_noise", "cer", "results")} for r in rs] for e, rs in rows.items()})[:3000])
EOF
echo DONE

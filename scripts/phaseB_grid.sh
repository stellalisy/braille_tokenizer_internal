#!/bin/bash
# Phase B stage 1 grid: grade x tokenizer x embedding, one seed. Each eval job waits for its run.
#   bash scripts/phaseB_grid.sh [seed] [grades] [toks] [embs] [extra train args...]
set -euo pipefail
cd /gpfs/projects/stf/stelli/braille_tokenizer_internal
SEED=${1:-0}; GRADES=${2:-"1 2"}; TOKS=${3:-"t1 t2 t3"}; EMBS=${4:-"e1 e2 e3"}; shift $(( $# < 4 ? $# : 4 )) || true
for g in $GRADES; do for t in $TOKS; do for e in $EMBS; do
  run=exp/phaseB/train/g${g}_${t}_${e}_s${SEED}
  jt=$(sbatch --parsable -t 02:00:00 -J brl-phaseB scripts/phaseB.sbatch train --grade $g --tok $t --emb $e --seed $SEED --out $run "$@" 2>/dev/null)
  je=$(sbatch --parsable -t 03:00:00 -J brl-phaseB --dependency=afterok:$jt scripts/phaseB.sbatch eval --run $run --conditions g${g}_p0 2>/dev/null)
  echo "$run train=$jt eval=$je"
done; done; done

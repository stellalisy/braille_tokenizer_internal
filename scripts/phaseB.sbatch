#!/bin/bash -l
#SBATCH -A argon
#SBATCH -p gpu-h200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH -J brl-phaseB
#SBATCH -o /gpfs/scrubbed/stelli/braille/logs/phaseB-%j.out
# Phase B GPU jobs. Usage: sbatch scripts/phaseB.sbatch {train|eval|qa|transcribe|repsim|choiceprobs|py} [script args]
#   train -> scripts/phaseB_train.py (training venv)
#   eval  -> scripts/phaseB_eval.py (vLLM venv)
#   qa    -> scripts/phaseB_qa_targets.py generate (vLLM venv)
#   transcribe -> scripts/phaseB_transcribe.py (vLLM venv)
#   repsim -> scripts/phaseB_repsim.py (training venv)
#   py <script> [args] -> any script in the training venv
set -euo pipefail
source /gpfs/projects/stf/stelli/braille_tokenizer_internal/scripts/env.sh
cd /gpfs/projects/stf/stelli/braille_tokenizer_internal
MODE=$1; shift
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "$MODE" = train ]; then
  python scripts/phaseB_train.py "$@"
elif [ "$MODE" = repsim ]; then
  python scripts/phaseB_repsim.py "$@"
elif [ "$MODE" = py ]; then
  python "$@"
elif [ "$MODE" = choiceprobs ]; then
  python scripts/phaseB_choiceprobs.py "$@"
else
  deactivate 2>/dev/null || true
  source "$BRL_SCRATCH/venv-vllm/bin/activate"
  # flashinfer/tvm_ffi probes every PATH dir; drop unreadable ones (see TILICUM_AGENTS_FYI.md)
  PATH=$(echo "$PATH" | tr ":" "\n" | while read -r d; do [ -r "$d" ] && [ -x "$d" ] && echo "$d"; done | paste -sd:); export PATH
  if [ "$MODE" = eval ]; then python scripts/phaseB_eval.py "$@"
  elif [ "$MODE" = transcribe ]; then python scripts/phaseB_transcribe.py "$@"; else python scripts/phaseB_qa_targets.py generate "$@"; fi
fi
echo DONE

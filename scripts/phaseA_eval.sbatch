#!/bin/bash -l
#SBATCH -A argon
#SBATCH -p gpu-h200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH -J brl-phaseA
#SBATCH -o /gpfs/scrubbed/stelli/braille/logs/phaseA-%j.out
# Phase A GPU inference. Usage: sbatch scripts/phaseA_eval.sbatch <hf_model_id> <revision> <run_name> [extra args]
set -euo pipefail
source /gpfs/projects/stf/stelli/braille_tokenizer_internal/scripts/env.sh
deactivate 2>/dev/null || true
source "$BRL_SCRATCH/venv-vllm/bin/activate"
# flashinfer/tvm_ffi probes every PATH dir; drop unreadable ones (see TILICUM_AGENTS_FYI.md)
PATH=$(echo "$PATH" | tr ":" "\n" | while read -r d; do [ -r "$d" ] && [ -x "$d" ] && echo "$d"; done | paste -sd:); export PATH
cd "$BRL_ROOT"
MODEL=$1; REV=$2; NAME=$3; shift 3
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/phaseA_eval.py --model "$MODEL" --revision "$REV" --out "exp/phaseA/$NAME" "$@"
echo DONE

#!/bin/bash -l
#SBATCH -A argon
#SBATCH -p gpu-h200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH -J brl-gpu
#SBATCH -o /gpfs/scrubbed/stelli/braille/logs/gpu-%j.out
# Generic GPU inference job in the vLLM env. Usage: sbatch scripts/gpu_run.sbatch python scripts/x.py ...
set -euo pipefail
source /gpfs/projects/stf/stelli/braille_tokenizer_internal/scripts/env.sh
deactivate 2>/dev/null || true
source "$BRL_SCRATCH/venv-vllm/bin/activate"
PATH=$(echo "$PATH" | tr ":" "\n" | while read -r d; do [ -r "$d" ] && [ -x "$d" ] && echo "$d"; done | paste -sd:); export PATH
cd "$BRL_ROOT"
"$@"
echo DONE

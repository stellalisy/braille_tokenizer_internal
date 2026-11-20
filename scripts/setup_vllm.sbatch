#!/bin/bash -l
#SBATCH -A argon
#SBATCH -p gpu-h200
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -J brl-vllm-setup
#SBATCH -o /gpfs/scrubbed/stelli/braille/logs/vllm-setup-%j.out
# CPU-only: inference env (vLLM + BrailleBench scoring deps) and base model downloads.
set -euo pipefail
source /gpfs/projects/stf/stelli/braille_tokenizer_internal/scripts/env.sh
deactivate 2>/dev/null || true
cd "$BRL_SCRATCH"
rm -rf venv-vllm
~/.local/bin/uv python install 3.12 && ~/.local/bin/uv venv --managed-python --python 3.12 venv-vllm  # managed Python ships Python.h, which Triton needs
source venv-vllm/bin/activate
~/.local/bin/uv pip install vllm "math-verify==0.9.0" "latex2sympy2_extended==1.11.0" \
  "antlr4-python3-runtime==4.13.2" huggingface_hub numpy tokenizers rapidfuzz
python -c "import vllm, torch, transformers, math_verify; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
for m in Qwen/Qwen3-1.7B meta-llama/Llama-3.1-8B-Instruct; do
  hf download "$m" --exclude "original/*" > /dev/null
  python -c "from huggingface_hub import HfApi; i=HfApi().model_info('$m'); print('$m', i.sha)"
done
du -sh "$HF_HOME/hub" 2>/dev/null || true
echo DONE

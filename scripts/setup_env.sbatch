#!/bin/bash -l
#SBATCH -A argon
#SBATCH -p gpu-h200
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -J brl-setup
#SBATCH -o /gpfs/scrubbed/stelli/braille/logs/setup-%j.out
# CPU-only: build liblouis 3.32.0 into a local prefix and create the Python env.
set -euo pipefail
source /gpfs/projects/stf/stelli/braille_tokenizer_internal/scripts/env.sh
cd "$BRL_SCRATCH"

echo "== liblouis $LIBLOUIS_VERSION"
rm -rf "$LIBLOUIS_SRC" "$LIBLOUIS_PREFIX"
tar xzf liblouis-$LIBLOUIS_VERSION.tar.gz
cd "$LIBLOUIS_SRC"
./configure --enable-ucs4 --prefix="$LIBLOUIS_PREFIX" > "$BRL_SCRATCH/logs/liblouis-configure.log" 2>&1
make -j8 > "$BRL_SCRATCH/logs/liblouis-make.log" 2>&1
make install > "$BRL_SCRATCH/logs/liblouis-install.log" 2>&1
"$LIBLOUIS_PREFIX/bin/lou_translate" --version | head -1 || true

echo "== venv"
cd "$BRL_SCRATCH"
rm -rf venv
~/.local/bin/uv venv --python /usr/bin/python3.12 venv
source venv/bin/activate
~/.local/bin/uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.8.*" numpy tokenizers rapidfuzz nltk pytest \
  transformers peft datasets accelerate
python - <<'EOF'
import louis, torch, transformers, peft, datasets, accelerate
print("louis", louis.version())
print("torch", torch.__version__, "transformers", transformers.__version__,
      "peft", peft.__version__, "datasets", datasets.__version__, "accelerate", accelerate.__version__)
EOF
python -c "import nltk; nltk.download('gutenberg', download_dir='$NLTK_DATA'); nltk.download('punkt_tab', download_dir='$NLTK_DATA')"
echo DONE

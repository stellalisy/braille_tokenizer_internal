# Source this in every job: `source /gpfs/projects/stf/stelli/braille_tokenizer_internal/scripts/env.sh`
# Rebuildable artifacts (liblouis build, venv, HF cache) live on scrubbed storage.
export BRL_ROOT=/gpfs/projects/stf/stelli/braille_tokenizer_internal
export BRL_SCRATCH=/gpfs/scrubbed/stelli/braille
export LIBLOUIS_VERSION=3.32.0
export LIBLOUIS_PREFIX=$BRL_SCRATCH/liblouis-local
export LIBLOUIS_SRC=$BRL_SCRATCH/liblouis-$LIBLOUIS_VERSION

export LD_LIBRARY_PATH=$LIBLOUIS_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LOUIS_TABLEPATH=$LIBLOUIS_PREFIX/share/liblouis/tables
export PYTHONPATH=$LIBLOUIS_SRC/python:$BRL_ROOT${PYTHONPATH:+:$PYTHONPATH}

export NLTK_DATA=$BRL_SCRATCH/nltk_data
if [ -f "$BRL_SCRATCH/venv/bin/activate" ]; then
  source "$BRL_SCRATCH/venv/bin/activate"
fi

# brltok

Braille tokenizers and a toy back-translation testbed for the braille tokenizer side project.
This covers milestone 2 of the research outline, the liblouis re-measurement from milestone 1, and from-scratch toy answers to research questions 1, 2, and 4.
Nothing here touches a pretrained LLM yet.

## What is in the package

| File | Contents |
| --- | --- |
| `brltok/cells.py` | Unicode cells, dot bitmasks, Braille ASCII conversion |
| `brltok/translate.py` | liblouis wrapper for UEB Grade 1 and 2, with position maps |
| `brltok/align.py` | Rule units, the cell groups liblouis emits together, with their print expansions |
| `brltok/tokenizers.py` | T1 `CellTokenizer`, T2 `BrailleBPE`, T3 `RuleUnitTokenizer` |
| `brltok/noise.py` | Dot flips, cell drops, cell insertions |
| `brltok/embed.py` | E1 `TableEmbedding`, E3 `DotEmbedding` (PyTorch) |
| `brltok/ctc.py` | Small Transformer encoder with a CTC head for braille to print |
| `scripts/` | Data building, tokenizer report, training, liblouis baseline, summary table |
| `tests/` | 16 tests, run with `python -m pytest tests` |
| `runs/` | Tokenizer files, logs, and `result.json` for every run reported below |

## Setup

```bash
pip install tokenizers torch rapidfuzz nltk pytest numpy
```

Tokenizing needs no liblouis. Building data, extracting rule units, and the rule-based baseline need liblouis 3.32 with its Python bindings.

```bash
curl -LO https://github.com/liblouis/liblouis/releases/download/v3.32.0/liblouis-3.32.0.tar.gz
tar xzf liblouis-3.32.0.tar.gz && cd liblouis-3.32.0
./configure --enable-ucs4 --prefix=$HOME/liblouis-local && make -j && make install
export LD_LIBRARY_PATH=$HOME/liblouis-local/lib:$LD_LIBRARY_PATH
export LOUIS_TABLEPATH=$HOME/liblouis-local/share/liblouis/tables
export PYTHONPATH=$PWD/python:$PYTHONPATH      # the bindings live in liblouis-3.32.0/python
```

## Reproduce

```bash
python -c "import nltk; nltk.download('gutenberg'); nltk.download('punkt_tab')"
python scripts/build_data.py --out data --test-sentences test_sentences.json
python scripts/tokenizer_report.py --data data --out runs/tokenizers
python scripts/eval_liblouis.py --out runs/liblouis
python scripts/train_backtranslate.py --tok cell --embed table --steps 4000 --out runs/cell_table
python scripts/train_backtranslate.py --tok cell --embed dots  --steps 4000 --out runs/cell_dots
python scripts/train_backtranslate.py --tok bpe --bpe-size 1000 --steps 4000 --out runs/bpe1000_table
python scripts/train_backtranslate.py --tok bpe --bpe-size 8000 --steps 4000 --out runs/bpe8000_table
python scripts/train_backtranslate.py --tok rule --steps 4000 --out runs/rule_table
python scripts/train_backtranslate.py --tok cell --embed table --train-noise 0.05 --steps 4000 --out runs/cell_table_noise
python scripts/train_backtranslate.py --tok cell --embed dots  --train-noise 0.05 --steps 4000 --out runs/cell_dots_noise
python scripts/summarize.py --runs runs
```

Each training run takes 6 to 11 minutes on one CPU core.

## Using the tokenizers

```python
from brltok.tokenizers import CellTokenizer, BrailleBPE, RuleUnitTokenizer

s = "⠠⠮ ⠅ ⠷ ⠡⠝⠂ ⠼⠁⠃ ⠷ ⠮⠍⠲"          # The knowledge of children, 12 of them.
CellTokenizer().encode(s)                                   # one id per cell
BrailleBPE.load("runs/tokenizers/bpe8000.json").tokens(s)
# ['▁⠠⠮</w>', '▁⠅</w>', '▁⠷</w>', '▁⠡⠝⠂</w>', '▁⠼⠁', '⠃</w>', '▁⠷</w>', '▁⠮⠍⠲</w>']
RuleUnitTokenizer.load("runs/tokenizers/rule.json").tokens(s)
# ['▁⠠⠮▁', '▁⠅▁', '▁⠷▁', '▁⠡⠝', '⠂▁', '▁⠼⠁', '⠃▁', '▁⠷▁', '▁⠮', '⠍', '⠲▁']
```

## Setup of the experiments

- Text is 36,841 training, 500 dev, and 500 test sentences of 5 to 35 words from the NLTK Gutenberg novels, translated to UEB with liblouis 3.32.0.
- The model is a 3-layer Transformer encoder of width 128. Each input token emits a fixed number of output frames (4 for cells, 5 for rule units, 12 for BPE) trained with CTC against print characters.
- Training and decoding use pieces of 8 words. UEB keeps print word spacing, so braille and print words pair one to one.
- Every run sees 4,000 steps of 48 pieces, about two passes over the data.
- Noise flips one random dot in each non-blank cell with the stated probability and never erases a cell.
- CER is character error rate against the print sentence. Exact match is at the sentence level.

## Results

### Rule-based round trip (liblouis 3.32.0, 500 test sentences)

| Grade | Exact sentence match |
| --- | --- |
| Grade 1 | 99.2% (the four misses are all-capitals passages) |
| Grade 2 | 97.6% |

The 74% and 41% in the first pilot came from the 2017 JavaScript build of liblouis 3.2.0.

### Tokenizer length and coverage (Grade 2, 500 test sentences, 8,872 words)

| Tokenizer | Vocabulary | Tokens per word | Print characters per token |
| --- | --- | --- | --- |
| T1 cell | 66 | 3.27 | 1.66 |
| T3 rule units | 1,546 | 2.98 | 1.82 |
| T2 BPE 1k | 1,000 | 1.97 | 2.75 |
| T2 BPE 4k | 4,000 | 1.48 | 3.65 |
| T2 BPE 8k | 8,000 | 1.33 | 4.08 |
| T2 BPE 16k | 16,000 | 1.21 | 4.47 |

For reference, the GPT-4 tokenizer uses 11,433 tokens on the same sentences in print, and BPE 8k uses 11,784 on the braille.
No tokenizer produces an unknown token on the test set.
T3 segments braille without liblouis. It reproduces the rule segmentation for 99.7% of test words with its lexicon (55,038 words, covering 94.1% of test words) and for 97.0% with longest match alone.

### From-scratch back-translation by tokenizer (clean training)

| Model | Vocabulary | Parameters | CER, clean | Exact match, clean |
| --- | --- | --- | --- | --- |
| liblouis 3.32 | rules | | 0.35% | 97.6% |
| T1 cell, table embedding | 66 | 0.71M | 0.56% | 82.8% |
| T1 cell, dot embedding | 66 | 0.72M | 0.62% | 78.8% |
| T2 BPE 1k | 1,000 | 0.91M | 1.43% | 54.4% |
| T3 rule units | 1,546 | 0.91M | 2.58% | 48.0% |
| T2 BPE 8k | 8,000 | 1.81M | 11.13% | 0.4% |

### Dot noise (CER on the test set)

| Model | Clean | 1% flips | 2% flips | 5% flips | 10% flips |
| --- | --- | --- | --- | --- | --- |
| liblouis 3.32 | 0.35% | 1.58% | 2.71% | 6.10% | 12.05% |
| T1 table, clean training | 0.56% | 1.76% | 2.81% | 6.08% | 11.90% |
| T1 dots, clean training | 0.62% | 1.79% | 2.79% | 5.92% | 11.49% |
| T1 table, trained with 5% flips (2 seeds) | 1.07% | 1.89% | 2.62% | 4.93% | 9.01% |
| T1 dots, trained with 5% flips (2 seeds) | 1.06% | 1.86% | 2.51% | 4.71% | 8.59% |

### What the numbers say

- The rule-based pipeline is strong on clean English and no toy model matches it.
- From scratch at equal training examples, smaller units learn faster. Cells beat BPE 1k and rule units, and BPE 8k is far behind.
- Rule units give almost no length reduction over cells, because most cells in contracted English are still single letters.
- Dot embeddings learn faster early (dev CER 22.8% against 76.2% at step 250) and end level with the table on clean text.
- liblouis and clean-trained models degrade linearly under dot noise, at about 1.2 CER points per point of flip rate. Rule-based decoding does not collapse.
- Training with noise lets the model repair some errors from context. At 10% flips CER drops from 12.05% (liblouis) to 9.01% with the table and 8.59% with dot embeddings, and the cost is about 0.5 points of clean CER. liblouis stays ahead below roughly 2% flips.
- Dot embeddings beat the table under noise in both seeds, by about 5% relative.

### Caveats

- Train and test braille both come from liblouis, so these numbers measure agreement with liblouis. No human-transcribed braille is used yet.
- The text is 19th-century fiction with few numbers and little technical content.
- Models are tiny and trained for about ten minutes each. The BPE 8k and rule-unit results reflect this budget, and frames per token differ across tokenizers.
- Two seeds exist only for the two noise-trained cell models. Everything else is one seed.
- The noise model is substitution only, at one dot per corrupted cell. It is not calibrated to real recognition errors.

## Not done yet

- Print-aligned initialization (E2), the frozen-body transplant, LoRA, the output head, and the layerwise analysis need pretrained weights and a GPU.
- The `expansions` table in `RuleUnitTokenizer` already stores the most common print expansion of each unit, which is the input E2 needs.

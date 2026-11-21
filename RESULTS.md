# Results

Accuracy in %. GSM8K test (1,319) is scored with Math-Verify and CommonsenseQA dev (1,221) with exact match, both with BrailleBench's own prompts and scoring. Decoding is greedy with 1,024 new tokens. Every Phase A number is one greedy run (no seeds to vary). Raw outputs are in `exp/phaseA/<run>/`, and the tables come from `python scripts/phaseA_table.py`.

## Phase A. Stock models and the rule-based pipeline (2026-11-20)

### Check against BrailleBench (Braille ASCII, clean)

| Model | Setting | GSM8K print | G1 | G2 | CSQA print | G1 | G2 |
|---|---|---|---|---|---|---|---|
| Qwen3 1.7B | BrailleBench reported | 69.4 | 2.4 | 1.4 | 37.4 | 9.6 | 1.5 |
| Qwen3 1.7B | Ours, thinking on | 69.5 | 1.4 | 1.8 | 34.5 | 9.2 | 1.4 |
| Qwen3 1.7B | Ours, thinking off | 82.8 | 3.2 | 1.2 | 47.8 | 8.8 | 3.4 |
| Llama 3.1 8B Instruct | BrailleBench reported | 83.9 | 4.6 | 1.4 | 43.4 | 0.2 | 0.5 |
| Llama 3.1 8B Instruct | Ours | 84.5 | 5.2 | 1.5 | 70.5 | 2.4 | 1.5 |

Our harness reproduces BrailleBench within 3 points in every cell but one. BrailleBench ran Qwen3 with thinking on, the chat template's default. Under the 1,024-token limit, that truncates 64% of print GSM8K answers and nearly all braille ones. Thinking off raises Qwen's print accuracy by 13 points (19% relative) on GSM8K. The exception is Llama on CommonsenseQA print, where we get 70.5 against 43.4. Removing the chat template does not explain it (63.7). Our Llama always answers with the choice text, and BrailleBench's exact match scores a bare letter or a letter-prefixed answer as wrong unless the letter is A. A run that answered with letters would land near their number, but their outputs are not released, so the cause is open.

### Clean input: how much of the gap the pipeline closes

| Model | Dataset | Print | Print, preprocessed | ASCII G1 | ASCII G2 | Unicode G1 | Unicode G2 | Pipeline G1 | Pipeline G2 | Gap closed G1 | Gap closed G2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3 1.7B, thinking off | GSM8K | 82.8 | 81.6 | 3.2 | 1.2 | 0.5 | 0.4 | 81.7 | 81.3 | 98.7 | 98.2 |
| Qwen3 1.7B, thinking off | CSQA | 47.8 | 47.4 | 8.8 | 3.4 | 0.0 | 0.0 | 47.4 | 47.3 | 99.2 | 99.1 |
| Qwen3 1.7B, thinking on | GSM8K | 69.5 | 68.4 | 1.4 | 1.8 | 0.4 | 0.6 | 68.2 | 68.4 | 98.1 | 98.3 |
| Qwen3 1.7B, thinking on | CSQA | 34.5 | 34.5 | 9.2 | 1.4 | 0.0 | 0.0 | 34.3 | 33.9 | 99.4 | 98.3 |
| Llama 3.1 8B | GSM8K | 84.5 | 84.5 | 5.2 | 1.5 | 1.0 | 0.8 | 84.3 | 84.0 | 99.7 | 99.4 |
| Llama 3.1 8B | CSQA | 70.5 | 70.5 | 2.4 | 1.5 | 0.0 | 0.0 | 70.5 | 70.4 | 100.0 | 99.9 |

On clean English the pipeline closes 98% to 100% of the gap for both models and both grades. That leaves no room for a learned front end on clean English reasoning, so the outline's milestone 1 rule applies. "Print, preprocessed" is the text BrailleBench rendered braille from, with paired `$` signs stripped. The pipeline matches it to within 0.6 points everywhere, so the remaining gap of up to 1.2 points to print comes from BrailleBench's preprocessing, not from liblouis. Stock models read Braille ASCII at 1% to 9%, and Unicode braille at 0% to 1%.

### Dot noise

GSM8K, Qwen3 1.7B (thinking off):

| Grade | Input | 0% | 2% | 5% | 10% |
|---|---|---|---|---|---|
| G1 | Braille ASCII | 3.2 | 3.3 | 2.6 | 2.3 |
| G1 | Unicode | 0.5 | 0.6 | 0.1 | 0.1 |
| G1 | Pipeline | 81.7 | 61.9 | 41.7 | 21.3 |
| G2 | Braille ASCII | 1.2 | 1.4 | 1.7 | 1.9 |
| G2 | Unicode | 0.4 | 0.6 | 0.4 | 0.5 |
| G2 | Pipeline | 81.3 | 63.5 | 40.6 | 22.7 |

Pipeline under noise, all runs (G1 / G2):

| Model | Dataset | 0% | 2% | 5% | 10% |
|---|---|---|---|---|---|
| Qwen3 1.7B, thinking off | GSM8K | 81.7 / 81.3 | 61.9 / 63.5 | 41.7 / 40.6 | 21.3 / 22.7 |
| Qwen3 1.7B, thinking off | CSQA | 47.4 / 47.3 | 44.1 / 41.6 | 36.2 / 33.1 | 22.1 / 22.9 |
| Qwen3 1.7B, thinking on | GSM8K | 68.2 / 68.4 | 47.3 / 52.8 | 32.1 / 31.2 | 13.4 / 14.1 |
| Qwen3 1.7B, thinking on | CSQA | 34.3 / 33.9 | 36.8 / 34.1 | 32.9 / 30.5 | 19.8 / 21.1 |
| Llama 3.1 8B | GSM8K | 84.3 / 84.0 | 65.1 / 67.4 | 47.0 / 46.9 | 24.0 / 26.2 |
| Llama 3.1 8B | CSQA | 70.5 / 70.4 | 60.4 / 58.6 | 47.0 / 46.0 | 30.6 / 27.6 |

Dot noise is where the pipeline breaks. On GSM8K it loses about a quarter of its accuracy at 2% flips, half at 5%, and three quarters at 10%, for both models (Qwen G1: 81.7, 61.9, 41.7, 21.3). One flipped dot in a digit changes the number, and liblouis recovers the exact question for only 3% to 7% of GSM8K items at 2% flips. CommonsenseQA degrades more slowly: Qwen keeps 93% of its clean pipeline accuracy at 2% flips. Stock braille accuracy stays at its floor under every noise level, so the pipeline's gap closed falls from 99% clean to about 24% at 10% flips (Qwen GSM8K G1). This is the room a learned front end can target. Full noise tables for every run, including ASCII and Unicode, are in `exp/phaseA/` and in the output of `scripts/phaseA_table.py`.

### Pipeline input quality

Share of questions that liblouis 3.32 back-translates exactly to the text BrailleBench rendered:

| Dataset | Grade | 0% | 2% | 5% | 10% |
|---|---|---|---|---|---|
| GSM8K | G1 | 95.9 | 3.1 | 0.1 | 0.1 |
| GSM8K | G2 | 94.9 | 7.4 | 0.5 | 0.0 |
| CSQA | G1 | 100.0 | 34.6 | 7.5 | 1.3 |
| CSQA | G2 | 99.8 | 45.2 | 15.2 | 3.4 |

### Truncation

Unicode braille makes both models loop. Qwen (thinking off) hits the 1,024-token limit on 97% of Unicode GSM8K items, against 3% on Grade 1 ASCII and 0.3% in print. Llama hits it on 65% of Unicode Grade 1 items.

### Not done yet in Phase A

- Step 6, the human-transcribed test set, is not collected yet.

### Hypotheses

- RQ4, pipeline matches or wins on clean English: **supported.** It closes 98% to 100% of the gap on both models.
- RQ4, learned front end wins on noisy input: **open.** The pipeline loses half its GSM8K accuracy at 5% flips, so there is room, but no learned model exists yet.
- RQ1, RQ3, RQ5, RQ6: **open.**
- RQ2, dot structure helps under noise: **supported in the toy setting** (below). Untested on an LLM.

## Toy models (2026-11-20)

### E1 table vs E3 dot embeddings under noise, five seeds

T1 cell tokenizer, trained with 5% flips, 4,000 steps, CER (%) on the 500 test sentences, mean ± standard deviation over seeds 0 to 4:

| Embedding | Clean | 1% | 2% | 5% | 10% |
|---|---|---|---|---|---|
| E1 table | 1.08 ± 0.07 | 1.90 ± 0.08 | 2.61 ± 0.10 | 4.94 ± 0.10 | 9.04 ± 0.09 |
| E3 dots | 1.05 ± 0.07 | 1.86 ± 0.06 | 2.51 ± 0.09 | 4.74 ± 0.09 | 8.61 ± 0.14 |
| liblouis 3.32 | 0.35 | 1.58 | 2.71 | 6.10 | 12.05 |

The two-seed lead holds with five seeds. Dot-composed embeddings lower CER at 10% flips by 4.8% (relative) on average, and they win in all five seeds (gains of 2.2% to 7.6%). On clean text the two are level. Both noise-trained models beat liblouis from 2% flips upward, by 28% (E3) and 25% (E1) relative at 10%. These numbers measure agreement with liblouis on 19th-century fiction. Runs are in `exp/toy/noise_seeds/`.

## Phase A step 6. Human-authored test set (2026-11-20)

The Grade 2 set is the 1,057 plain-text examples from *The Rules of Unified English Braille* (ICEB) that liblouis keeps as tests (`en-ueb.yaml`). The braille was written by the rulebook authors. Items are words and short phrases, 9.7 print characters on average. The Grade 1 harness (Torch Trust) adds 58 items of about 2 characters and is too small to use. Built by `scripts/human_ueb_testset.py`, data in `data/human/ueb_rulebook.jsonl`, scores in `exp/phaseA/human_ueb/result.json`.

| Set | n | liblouis forward matches the human braille | Back-translation exact | Back-translation CER |
|---|---|---|---|---|
| Grade 2 rulebook | 1,057 | 97.9% | 92.0% | 4.3% |

liblouis is developed against these examples, so its score is in-sample. The set measures learned models against human braille, and it covers rule cases that liblouis output never shows a model. It has no running text. Human-transcribed English sentences are still missing, and the BrailleLLM release contains only Chinese data (49,554 training and 1,000 test pairs), which serves Phase E.

## Phase A with longer limits and choice scoring (2026-11-20)

These runs replace the 1,024-token tables above. Thinking off uses 8,192 new tokens and thinking on uses 32,768. CommonsenseQA is scored with `brltok.scoring.choice_correct`, which counts a letter answer as that choice (NOTES.md, 2026-11-20). Raw outputs are in `exp/phaseA/qwen3-1.7b-8k`, `llama3.1-8b-instruct-8k` and `qwen3-1.7b-thinking-32k`.

| Model | Dataset | Print | ASCII G1 | ASCII G2 | Unicode G1 | Unicode G2 | Pipeline G1 | Pipeline G2 |
|---|---|---|---|---|---|---|---|---|
| Qwen3 1.7B, thinking off | GSM8K | 82.8 | 3.4 | 1.4 | 0.5 | 0.5 | 81.8 | 81.4 |
| Qwen3 1.7B, thinking off | CSQA | 68.3 | 61.8 | 21.5 | 0.0 | 0.0 | 68.1 | 68.1 |
| Qwen3 1.7B, thinking on | GSM8K | 90.3 | 4.5 | 1.7 | 0.4 | 0.5 | 90.1 | 89.3 |
| Qwen3 1.7B, thinking on | CSQA | 74.6 | 72.7 | 23.8 | 0.0 | 0.0 | 74.5 | 74.5 |
| Llama 3.1 8B | GSM8K | 84.5 | 4.6 | 1.5 | 1.4 | 1.3 | 84.8 | 84.4 |
| Llama 3.1 8B | CSQA | 70.8 | 63.9 | 21.3 | 0.0 | 0.0 | 70.9 | 70.8 |

- Stock models cannot do GSM8K from braille even with 32k thinking tokens: Qwen scores 90.3 in print and 4.5 in Grade 1.
- Grade 1 Braille ASCII reaches 90% to 97% of print accuracy on CommonsenseQA, so the models read uncontracted words. Grade 2 sits at 21% to 24% (chance is 20%). The Grade 1 GSM8K failure is then more likely about numbers than words.
- The pipeline closes at least 98% of the GSM8K gap and at least 96% of the CommonsenseQA gap everywhere. The 8,192-token limit changes thinking-off results by at most 0.2 points.
- Under noise the pipeline falls as before (Qwen GSM8K G1: 81.8, 61.9, 41.8, 21.3 at 0, 2, 5, 10%). At 10% flips the stock model on noisy Grade 1 ASCII CommonsenseQA is within 6 points of the pipeline (29.9 against 36.0).

## Phase B gate B3: braille through a frozen model before training (2026-11-20)

Qwen3 1.7B, frozen, no training. Mean NLL of the print sentence after "Braille: <input>\nPrint:", 200 dev sentences of `data/phaseB/dev.jsonl`. Output in `exp/phaseB/zero_shot/result.json`.

| Input | Grade 1 | Grade 2 |
|---|---|---|
| Print copy (floor) | 0.27 | 0.27 |
| Stock tokenizer, Braille ASCII | 0.82 | 2.72 |
| Stock tokenizer, Unicode | 3.73 | 3.74 |
| T1 cell, E1 / E2 | 6.88 / 3.12 | 6.67 / 3.59 |
| T2 BPE 8k, E1 / E2 | 5.24 / 0.83 | 5.24 / 0.91 |
| T3 rule units, E1 / E2 | 6.64 / 0.66 | 6.30 / 0.66 |

Print-aligned initialization alone lets the frozen model read braille. Rule units with E2 lower the Grade 2 loss by 76% relative to the stock tokenizer on Braille ASCII and win on all 200 sentences; in Grade 1 they lower it by 20% and win on 142 of 200. E2 beats E1 for every tokenizer, by 46% to 90%. Cells gain least, because under the first-cell rule many cells have no print expansion (gate B2). B3 passes for both grades.

## Phase B stage 1: frozen-body transplant grid (Qwen3 1.7B, thinking off)

Training only the braille input closes 93% of the Grade 1 GSM8K gap and 94% of the Grade 2 gap, with the base model unchanged. Checkpoint 2 (more than half of the Grade 1 GSM8K gap) is met. The pipeline still closes more (98.7% and 98.3%). The base model's parameter hash matched at the start and end of all 26 training runs.

GSM8K accuracy (%) and gap closed (GC) against print 82.8 and stock Braille ASCII 3.4 (G1) and 1.4 (G2). T3 with E2 and E3 are means over seeds 0 to 2 (standard deviation in brackets); all other cells are seed 0.

| Tokenizer | G1 E1 | G1 E2 | G1 E3 | G2 E1 | G2 E2 | G2 E3 |
|---|---|---|---|---|---|---|
| T1 cell | 75.3 (GC 90.5) | 75.9 (91.3) | 76.3 (91.8) | 75.7 (91.2) | 77.6 (93.6) | 76.5 (92.3) |
| T2 BPE 8k | 59.4 (70.6) | 58.6 (69.5) | 64.4 (76.9) | 62.1 (74.6) | 56.0 (67.1) | 64.5 (77.5) |
| T3 rule units | 75.0 (90.2) | 77.2 [0.2] (93.0) | 76.9 [0.6] (92.6) | 75.1 (90.5) | 77.8 [0.4] (93.9) | 78.3 [0.5] (94.5) |

CommonsenseQA accuracy (%, choice scoring); print 68.3, stock Braille ASCII 61.8 (G1) and 21.5 (G2), pipeline 68.1.

| Tokenizer | G1 E1 | G1 E2 | G1 E3 | G2 E1 | G2 E2 | G2 E3 |
|---|---|---|---|---|---|---|
| T1 cell | 50.2 | 52.0 | 54.5 | 53.6 | 54.0 | 52.1 |
| T2 BPE 8k | 50.1 | 60.1 | 50.3 | 49.1 | 55.6 | 51.4 |
| T3 rule units | 54.2 | 54.7 [0.7] | 54.5 [1.0] | 55.5 | 57.1 [0.6] | 56.5 [1.2] |

- **Tokenizer.** It matters more than initialization once training starts. Rule units and cells close 90% to 95% of the GSM8K gap; BPE closes 67% to 78%. Print-aligned (E2) and dot-composed (E3) initialization add 2 to 3 points over mean init (E1) for rule units.
- **CommonsenseQA.** It transfers less. Grade 2 closes up to 76% of its gap (T3 E2). In Grade 1 every trained input scores below stock Braille ASCII, whose gap is only 6.5 points.
- **Diagnostic (T3 E2, seed 0, CommonsenseQA).** It locates the Grade 1 loss in reading braille:

  | Question | Choices | G1 | G2 |
  |---|---|---|---|
  | braille | braille | 54.0 | 56.9 |
  | print | braille | 61.4 | 61.7 |
  | braille | print | 60.6 | 62.8 |
  | print | print | 69.4 | 69.4 |

  The last row uses the braille-input template with everything in print, so the template itself costs nothing (GSM8K 83.0 and 81.8 in that setting).
- **Selection.** T3 E2 and T3 E3 were picked for extra seeds by mean accuracy on the held-out training questions (`eval_qadev`, 250 GSM8K + 250 CommonsenseQA), not by test scores.
- **Cost.** 3,000 steps take 17 to 25 minutes per run on one H200. Stage 1 so far (26 runs, their evaluations, debug runs, QA targets) used 10.5 GPU-hours (sacct, jobs named brl-phaseB).

Runs: `exp/phaseB/train/g{grade}_{tok}_{emb}_s{seed}/` (`train_log.json`, `eval/result.json`, `eval_qadev/`, diagnostics in `eval_print_{question,choices,all}/`). Command: `bash scripts/phaseB_grid.sh <seed> "<grades>" "<toks>" "<embs>"`. Git commit and liblouis 3.32.0 are in each `train_log.json` and `eval/config.json`. Data: `data/phaseB/manifest.json` and `data/phaseB/qa/manifest.json`.

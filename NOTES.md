# Notes

Decisions, assumptions, dead ends, and deviations from the outline. Newest last.

## 2026-11-20. Setup

- **Cluster and storage.** Work runs on Tillicum (Slurm account `argon`). The repository is cloned at `/gpfs/projects/stf/stelli/braille_tokenizer_internal`, since home is capped at 10 GB. Rebuildable artifacts (liblouis build, venv, model weights, generated data) live in `/gpfs/scrubbed/stelli/braille`, which is mode 700. Checkpoints and eval outputs that nothing else holds go under `exp/` in the repository folder, and large files there are gitignored.
- **Compute.** Stella approved GPU inference for Phase A. Everything else runs on CPU until she says otherwise. No total budget was given, so each job stays under 12 GPU-hours and the project stays inside the 100 free GPU-hours unless Stella approves more.
- **Base models.** Qwen3 1.7B is the main model. Llama 3.1 8B is optional and only used if the Hugging Face token has access to the gated weights. Neither is in `/gpfs/models`, so weights are downloaded to scrubbed storage.
- **Environment.** `scripts/env.sh` sets `LD_LIBRARY_PATH`, `LOUIS_TABLEPATH`, and `PYTHONPATH` for liblouis 3.32.0 and activates the venv. Every job sources it. `scripts/setup_env.sbatch` rebuilds liblouis and the venv from scratch on a CPU node (Python 3.12, torch 2.8 cu128).
- **Removed `scripts/queue.sh`**, which held paths from the machine it was written on.
- **The shipped `runs/` folder is kept as the record of the toy results.** The liblouis re-check writes to `exp/setup/liblouis_rerun/` instead of overwriting `runs/liblouis/`.
- **`brltok/noise.py` already has `drop_cells` and `insert_cells`,** although the handoff file lists noise as substitution only. They are untested and unused by the shipped runs.
- **Setup checks pass.** All 16 tests pass, including the liblouis alignment test. The rebuilt data has 36,841 train, 500 dev, and 500 test sentences, and `scripts/eval_liblouis.py` reproduces `runs/liblouis/result.json` exactly (Grade 2 clean: 97.6% exact, 0.35% CER). Output is in `exp/setup/liblouis_rerun/`.
- **NLTK data was fetched with curl.** NLTK 3.10 refuses downloads through the cluster proxy. The `gutenberg` and `punkt_tab` zips came from the official nltk_data repository (sha256 recorded in the setup log) into `$BRL_SCRATCH/nltk_data`. The proxy check was left on.

## 2026-11-20. Phase A preparation

- **BrailleBench release found and used.** Benchmark: github.com/jinghanzhang1998/braillebench at `ca8997a`. Toolkit: github.com/jinghanzhang1998/braille-toolkit at `2231d4e`. Both use MIT for code and are cloned to `$BRL_SCRATCH/ext`. Phase A imports the benchmark's own prompt builder, answer extraction, and scoring, so prompts and extraction are identical across conditions and to the paper.
- **Splits.** GSM8K test (1,319) and CommonsenseQA dev (1,221), as in the release.
- **Data licenses.** GSM8K is MIT. CommonsenseQA's official repository names no license, and BrailleBench redistributes the dev set "for research reproducibility" without relicensing it. Stella said licenses do not matter for this project and waived the stop-and-ask rule on licenses (2026-11-20). Provenance is still recorded here.
- **Our liblouis matches theirs.** Re-rendering all 5,078 GSM8K and CommonsenseQA questions with our liblouis 3.32.0 reproduces the released braille cell for cell in both grades (`exp/phaseA/render_check/`).
- **BrailleBench strips paired `$` before rendering.** Its LaTeX preprocessing treats `$...$` as math delimiters, so 171 of 403 GSM8K questions with dollar signs lose them in the braille versions. liblouis itself round-trips `$` correctly. The print condition uses the original question, so I added a `print_clean` condition on the preprocessed text to separate this effect. An earlier note in conversation that liblouis drops `$` was wrong.
- **Braille ASCII convention.** BrailleBench writes the lowercase BRF forms (`` ` { | } ~ ``) where `brltok.cells.to_ascii` writes `@ [ \ ] ^`. Noisy ASCII inputs keep the released character at unchanged cells and use the lowercase form at flipped cells, and the builder asserts that clean inputs equal the release byte for byte.
- **Noise.** `brltok.noise.flip_dots` on the Unicode cells, one RNG per item seeded from `4242|dataset|grade|rate|index`. ASCII, Unicode, and pipeline conditions see the same corrupted cells. Answer choices in CommonsenseQA are corrupted too, since they are part of the braille content.
- **Pipeline input quality.** Clean liblouis round trips recover 95.9% (G1) and 94.9% (G2) of GSM8K questions exactly and 100% and 99.75% of CommonsenseQA questions. At 2% flips exact recovery falls to 3% to 7% on GSM8K and 35% to 45% on CommonsenseQA.
- **Decoding.** Greedy, 1,024 new tokens, one user turn through the model's chat template, Qwen3 thinking mode off. The paper does not state the thinking setting. Off is the choice that fits a 1,024-token budget, and the print number will show whether it matches.
- **Models.** `Qwen/Qwen3-1.7B` at revision `70d244c` (Apache-2.0). The Hugging Face token has access to `meta-llama/Llama-3.1-8B-Instruct`, so Llama is the second model. BrailleBench does not give exact ids; the instruct model is assumed.
- **Inference env.** A separate venv (`$BRL_SCRATCH/venv-vllm`) holds vLLM and the BrailleBench scoring pins (math-verify 0.9.0), so the training env's torch stays fixed.
- **vLLM needs uv-managed Python.** Triton compiles a C launcher and needs `Python.h`, which the system Python 3.12 lacks. `venv-vllm` is built on uv's managed CPython 3.12. vLLM 0.34.0.
- **CommonsenseQA scoring quirk (BrailleBench's metric, kept as is).** Exact match strips punctuation and articles, so a braille-echoed answer `a"> complete job` scores correct for choice A only, because "a" is dropped as an article. A braille echo of any other choice scores zero. This inflates braille-input CommonsenseQA slightly for choice-A items. Kept for comparability with the paper.
- **Paper outline** is in the Overleaf project `6b00b51dab16703958a99d05` (commit `0cbc9da`), compiled with tectonic on a CPU node. Tables with results not yet run are left blank.

## 2026-11-20. Phase A results

- **Thinking mode.** BrailleBench's Qwen3 numbers are reproduced with thinking on (GSM8K print 69.5 vs 69.4), not off (82.8). Both settings are reported. Proposed for Phase B onward: thinking off, because thinking on truncates 64% of print and nearly all braille GSM8K generations at 1,024 tokens, which mixes reading failures with budget failures. Waiting for Stella.
- **Llama CommonsenseQA print does not match BrailleBench** (70.5 vs 43.4). Checked: prompts and scoring are BrailleBench's code; no chat template gives 63.7, so not that; our Llama never answers with letters. Unexplained, flagged at checkpoint 1 under the five-point rule. GSM8K matches (84.5 vs 83.9).
- **vLLM memory.** vLLM 0.34 defaults to `gpu_memory_utilization=1.0` here, and long thinking generations ran out of memory. Capped at 0.85 from the resumed thinking run on; greedy outputs do not depend on it.
- **Diagnostic runs** `exp/phaseA/diag_qwen_thinking` and `exp/phaseA/diag_llama_nochat` cover print conditions only.
- **Toy five-seed rerun** of E1 vs E3 under noise is done (`exp/toy/noise_seeds/`, summary in `summary.json`).
- **GPU use so far:** 1.41 GPU-hours on full H200s (sacct) for all Phase A runs, smoke tests, and diagnostics.
- **Generation limit raised to 8,192 new tokens** (Stella, 2026-11-20: 1,024 is too short). Phase A is rerun for Qwen thinking off, Qwen thinking on, and Llama into `exp/phaseA/*-8k/` (jobs 504794 to 504796, max model length 10,240). The 1,024-token runs stay as the BrailleBench replication. If thinking-on print still truncates more than a few percent at 8,192, the limit goes up again.
- **Thinking mode gets 32,768 new tokens** (Stella, 2026-11-20: 8,192 is still too short), following the Qwen3 model card. Max model length 34,816 (the model supports 40,960). The thinking-8k job was cancelled. The run is split into four shards under `exp/phaseA/qwen3-1.7b-thinking-32k/` (jobs 504801 to 504804), so that looping braille generations cannot push one job past 12 GPU-hours. `scripts/phaseA_table.py` merges the shards. Thinking-off runs stay at 8,192.

## 2026-11-20. CommonsenseQA scorer

- **BrailleBench exact match undercounts CommonsenseQA badly.** It compares the extracted answer text with the gold choice text, so a correct letter answer ("C", or "c" when reading braille) scores 0, and so does a braille echo of a choice label (`c"> casino`). Only choice A survives, because "a" is dropped as an article. Rescoring our saved outputs: Qwen (thinking off) print goes from 47.8 to 68.3 and Grade 1 Braille ASCII from 8.9 to 61.8. Llama print barely moves (70.5 to 71.2) because Llama writes the choice text.
- **New scorer: `brltok.scoring.choice_correct`** (tests in `tests/test_scoring.py`). It decides which choice the response selects (letter, letter prefix, braille-echoed label, or exact choice text; the text wins when a prefixed letter and its text disagree) and checks it against the gold choice, accepting either letter when a gold text appears twice. It takes the span after the last "The answer is" up to the end of its first sentence. Across all saved outputs, exact match marks right what this scorer marks wrong in only 1 case. Chance is 20%: Grade 2 ASCII scores 12% to 22%, at or below chance.
- **`scripts/rescore_csqa.py`** rescores saved outputs offline and writes `<run>/csqa_choice.json`. Choices come from the clean print records.
- **Plan (Stella, 2026-11-20):** once BrailleBench's number is reproduced and their error is clear, use this scorer for CommonsenseQA and keep released exact match only for the comparison with the paper. Reproduction job: `scripts/bb_print_repro.py`.
- **BrailleBench's Llama CommonsenseQA number is a scoring artifact (reproduced 2026-11-20).** Their print numbers come from the July `src/evaluate.py` (commit 20c449c), run on AWS Bedrock with a hand-built Llama 3 prompt (no system block, one newline after each header). Their experiment log records Llama 3.1 8B CommonsenseQA at 43.3 after a letter-prefix fix. Under that prompt Llama mostly answers with a bare letter ("The answer is: A"), which exact match scores 0. `scripts/bb_print_repro.py` (`exp/phaseA/bb_print_repro/`): their prompt gives 18.1 to 19.1 (greedy and temperature 0.01, three seeds) and 35.5 with a duplicated BOS, against 72.2 with the chat template. With `choice_correct`, every variant scores 71.0 to 72.6. Their 43.3 lies between, depending on how often Bedrock's serving led to letter answers, which their unreleased outputs would show. GSM8K is unaffected (83 to 86 in every variant; paper 83.9).
- **Decision: CommonsenseQA is scored with `choice_correct` from here on** (Stella: use a reliable scorer once their error is clear). `scripts/phaseA_eval.py` records it as `choice` and uses it as the primary metric; `scripts/phaseA_table.py` uses it for older runs through `csqa_choice.json`. Released exact match is kept only to compare with the paper.
- **No GPU-hour cap beyond the per-job limit** (Stella, 2026-11-20: forget about the 100 free hours). GPU-hours are still reported.
- **Chunked saving.** `scripts/phaseA_eval.py` now generates each condition in chunks of 256 items and appends each chunk to the details file, so a stopped job resumes at the last saved item (tested by truncating details files and rerunning). Greedy vLLM decoding is not batch-invariant: regenerating 10 items in a different batch changed 1 answer. Expect differences of a fraction of a point between a resumed and an uninterrupted run.
- **GSM8K braille conditions with thinking at 32k tokens run one job per condition** (16 jobs, `exp/phaseA/qwen3-1.7b-thinking-32k/gsm8k_<condition>/`). The two shards that held them were cancelled after 22 minutes, before any condition finished. Looping braille inputs generate close to 32k tokens per item, so one condition can take hours.

## 2026-11-20. CommonsenseQA thinking run split per condition

Job 504804 (Qwen thinking, 32k tokens, CommonsenseQA braille) started before chunked saving and takes about 2.5 hours per Unicode condition (8,881 s for g1_unicode_p0, 516 of 1,221 truncated at 32k). Its remaining conditions would exceed the 20-hour limit. It finishes g1_unicode_p10, then job 505586 cancels it. The other 10 conditions run as per-condition jobs 505576 to 505585 with chunked saving and --requeue, in `exp/phaseA/qwen3-1.7b-thinking-32k/commonsenseqa_<cond>`. `scripts/phaseA_table.py` merges the shard folders and fails if a condition appears twice.

## 2026-11-20. Gate for Phase B training

Stella approved GPU training for Phase B on the condition that the steps before it support the hypothesis. The gate, fixed before looking at any new result:

Phase A (from the eval harness):
- A1. Stock Qwen3 1.7B on clean GSM8K Grade 1 Braille ASCII is at least 50 points below print. Status: met (82.8 vs 3.4, thinking off, 8k).
- A2. The pipeline closes at least 90% of the clean gap. Status: met (98% to 100%, every run).
- A3. The pipeline loses at least 25% of its clean GSM8K accuracy (relative) at 5% flips, so a learned front end has room. Status: met (about 50%).
- A4. The remaining Phase A runs (thinking 32k, Llama 8k) finish and do not contradict A1 to A3.
- A5. Step 6: a human-transcribed test set is collected, with provenance, and liblouis back-translation is scored on it.

Phase B preparation (CPU):
- B1. Training data contains no item from the GSM8K test or CommonsenseQA dev questions (exact and normalized match).
- B2. Every T1, T2 and T3 token has a print expansion, and at least 95% of training tokens have a non-empty one.
- B3. Zero-training check of the print-aligned initialization: with the body frozen and no training, the loss of the print continuation given a braille prefix embedded with E2 is lower than with E1 and lower than with the stock tokenizer on Braille ASCII, on a 200-sentence dev sample. This is the first direct test that braille can reach print representations through the input layer alone.
- B4. A unit test shows the base parameter hash is unchanged after a training step.

If all pass, training starts with thinking off (my recommendation, which Stella has not overruled), short debug runs first. If any fails, I report and wait.

## 2026-11-20. Gate A5: human-authored test set

BrailleLLM (github.com/Tianyuan-Huang/BrailleLLM at 30d3c41) releases only Chinese data, so it cannot be the English human set. Used the ICEB rulebook examples that liblouis 3.32.0 ships in `tests/braille-specs/en-ueb.yaml` instead (provenance in the file header; 96 entries with emphasis or mode options skipped). The braille display is NABCC, converted with the lowercase BRF mapping; liblouis forward output matches 97.9% of items, which confirms the conversion. A5 is met for word and phrase level only; a running-text English set remains an open item and does not block Phase B.

## 2026-11-20. Gate B3 made precise (before running it)

B3 is judged per grade on the first 200 sentences of `data/phaseB/dev.jsonl`, with one template for every condition: print "Braille: ", then the braille input, then print "\nPrint:", and the loss is the mean negative log-likelihood of " " plus the print sentence. It passes for a grade if (a) E2 has lower loss than E1 for each of T1, T2 (8k) and T3, and (b) the best E2 tokenizer has lower loss than the stock tokenizer reading Braille ASCII. A print-to-print copy with the same template is reported as the floor. Qwen3 1.7B at the Phase A revision, no training, float32 on CPU.

B4 (freeze proof) passes: `tests/test_bridge.py::test_freeze_proof_base_hash_unchanged_after_step` shows the base parameter hash unchanged and no base gradients after an optimizer step on the braille embeddings.

## 2026-11-20. Gates B1 and B2

B1 passes. `data/phaseB` has 284,241 training and 2,000 dev segments (WikiText-103 148k, Dolly 56k, Gutenberg 37k, GSM8K train answers 27k and questions 7k, CommonsenseQA train questions 10k), none of the 2,540 GSM8K test or CommonsenseQA validation questions, exactly or normalized. 1,460 segments were dropped because liblouis wrote untranslatable characters as 8-dot cells. Every segment has equal braille and print word counts in both grades. Manifest in `exp/phaseB/data_manifest.json`.

B2 fails as written, for a reason in the expansion rule rather than the code. Report in `exp/phaseB/tokenizer_report.json`:

| Tokenizer | G1 types with expansion | G1 non-empty share | G2 types with expansion | G2 non-empty share | G2 tokens per word |
|---|---|---|---|---|---|
| T1 cell | 44 of 61 | 95.6% | 63 of 64 | 91.8% | 4.83 |
| T2 BPE 8k | 7,961 of 7,992 | 99.5% | 7,992 of 7,995 | 99.1% | 1.53 |
| T3 rule units | 1,035 of 1,035 | 100% | 3,018 of 3,018 | 100% | 3.49 |

The instructions give a unit that a token cuts through to the token holding its first cell. Under T1 every multi-cell sign is cut, so its later cells get no expansion (in "had", dots 456 get the word and the h cell gets nothing in that position), and cells that only occur after an indicator never get one. They fall back to mean initialization. T3 passes fully and T2 nearly. Reported to Stella; training waits for her call.

## 2026-11-20. Stella's decisions on the gate

1. B2 is accepted as met for T2 and T3. T1 tokens without a print expansion fall back to mean initialization under E2.
2. If B3 Grade 1 had failed against stock Braille ASCII, Grade 2 would decide. It did not come to that: Grade 1 passes (T3 with E2 0.655 against stock ASCII 0.821).

With the gate met, Phase B GPU training is approved, thinking off. Debug runs first, then the T1/T2/T3 by E1/E2/E3 grid on Qwen3 1.7B, budget about 40 to 60 GPU-hours.

## 2026-11-20. Gate B3 passes

Grade 1 and Grade 2 both pass (RESULTS.md). All gates are met except A4, whose remaining runs are Qwen thinking on Unicode braille and cannot change A1 to A3. Phase B training starts.

## 2026-11-21 Phase A complete; Phase B stage 1 design

Phase A is complete. The last runs (Qwen thinking on, Unicode Grade 2) score 0.5 on GSM8K and 0.0 on CommonsenseQA, like every other Unicode condition. Job 504804 hit its 20 h limit after finishing its Grade 1 Unicode conditions, and the per-condition jobs covered the rest. Phase A used 101 GPU-hours in total (sacct, jobs named brl-phaseA).

Phase B stage 1 decisions, fixed before the grid:

- **What trains.** Only the braille embedding module. The base model is frozen, and its parameter hash is checked at the start and end of every run (`base_unchanged` in `train_log.json`).
- **Grades.** One run per grade, each with that grade's tokenizer, embeddings and data. Grade 1 first, since Checkpoint 2 is about Grade 1.
- **Examples.** Half of each batch is transcription (`"Braille:" + braille + "\nPrint:"` followed by the print sentence, from `data/phaseB/train.jsonl`). The other half is QA: BrailleBench's braille-input prompt (G1-EN or G2-EN, with its "Unicode Braille characters" note), in the chat template with thinking off, with question and choices spliced in as braille spans.
- **QA targets.** The QA target is the frozen model's own print answer to the same question (`scripts/phaseB_qa_targets.py`, GSM8K train and CommonsenseQA train). Every answer is kept, right or wrong, so training pushes braille toward print behavior rather than toward gold answers. That keeps the gap-closed metric honest: a braille score above print would mean something leaked. The frozen model gets 86.7% of GSM8K train and 64.7% of CommonsenseQA train right; 7 of 17,214 answers hit the token limit and were dropped. 250 items per dataset are held out as QA-dev.
- **Spacing.** Print text right before a braille span drops its trailing space, because a word-start braille token already stands for a print token with a leading space (E2).
- **Loss.** LM loss on the target, plus λ times distillation. The distillation term is the squared error between the student hidden state at the last token of each braille word and the frozen model's hidden state at the last token of the same print word, divided by the teacher's squared norm. It is averaged over words and over layers 4, 8 and 14 (of 28). λ = 1. Words pair only when a span has the same number of words in braille and print.
- **Optimizer.** AdamW at lr 1e-3, no weight decay, 100 warmup steps, then cosine. Batch 16, 3,000 steps, gradient clip 1, seed 0. These are set before any result. Only QA-dev and transcription-dev losses are used to change them.
- **E3 for every tokenizer.** A cell vector is a linear map of the six dot bits, plus a per-cell residual (zero at start), plus a position vector. A token averages its cells, adds word-start and word-end flags, and a two-layer MLP (512 to 2048 to 2048) maps it to the model dimension (`brltok/composed.py`). It is fitted to the E2 table (3,000 Adam steps, MSE) before training.
- **Evaluation.** `scripts/phaseB_eval.py` feeds vLLM prompt embeddings and uses Phase A's items, extraction and scoring, thinking off, 8,192 new tokens. The path itself was checked by sending the Phase A print prompts through it on 200 items per dataset. CommonsenseQA responses are identical (200/200). GSM8K accuracy is identical (82.0 both), with 167/200 responses identical character for character; the rest differ by bf16 batch effects over long generations.
- **Cost.** A debug run takes 0.45 s per training step on one H200, so a 3,000-step run is about 25 minutes plus evaluation.

## 2026-11-21 Phase B stage 1 grid done; Checkpoint 2 met

Training only the braille input closes 93% of the Grade 1 GSM8K gap (T3 E2, three seeds), well over the half-gap criterion, so Phase B continues (results in RESULTS.md). CommonsenseQA Grade 1 is the exception: trained inputs score below stock Braille ASCII. A diagnostic shows the loss is braille reading, about 7 points in the question and 7 in the choices; the prompt template costs nothing. Seeds were added for T3 E2 and T3 E3, chosen by accuracy on held-out training questions, not test. Next: the noise evaluation (p2, p5, p10) for the selected configurations, then stage 2 or the ablations in the outline.

## 2026-11-21 Noise evaluation of stage 1

Clean-trained learned inputs beat the pipeline on GSM8K from 5% flips upward and tie at 2%; the pipeline stays ahead on CommonsenseQA at every rate (RESULTS.md). E3 does not consistently beat E2 under noise. The next step the outline calls for is training with noisy braille; not started.

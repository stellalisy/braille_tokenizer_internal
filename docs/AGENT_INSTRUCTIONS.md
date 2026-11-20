# Agent instructions for the braille tokenizer project

You are continuing a research side project owned by Stella Li.
The repository is https://github.com/stellalisy/braille_tokenizer_internal and it is internal, so nothing from it gets published or uploaded anywhere else.

This folder holds this file, the research outline, and `brltok-0.1.0.tar.gz`.
Read the outline in full before doing anything.
It defines the research questions (RQ1 to RQ6), the tokenizer variants (T0 to T4), the embedding variants (E1 to E4), the baselines, the evaluation, and the milestones.
This file uses those labels without redefining them.

## Fill in before handing over (Stella)

- Compute budget per job and in total, and the scheduler to use.
- Where checkpoints and datasets should live on the cluster.
- Whether a Hugging Face token with access to gated Llama weights is available.

If any of these is still blank when you start, use Qwen only, keep every job under 12 GPU-hours, and record the assumption in `NOTES.md`.

## Goal

Attach a braille input layer to a frozen pretrained LLM, then measure how much of the gap between print and braille accuracy it closes on reasoning tasks.
Every result is reported next to a rule-based pipeline, which is liblouis back-translation followed by the same unmodified LLM.

## What already exists

The tarball unpacks to a folder `brltok/` with a README that documents everything below in more detail.

| Piece | Where | State |
| --- | --- | --- |
| Cell utilities, Braille ASCII conversion | `brltok/cells.py` | Done, tested |
| liblouis wrapper with position maps | `brltok/translate.py` | Done, needs liblouis 3.32 |
| Rule units and their print expansions | `brltok/align.py` | Done, tested |
| Tokenizers T1, T2, T3 | `brltok/tokenizers.py` | Done, tested |
| Dot noise | `brltok/noise.py` | Done, substitution noise only |
| Embeddings E1 and E3 | `brltok/embed.py` | Done for the toy model |
| Toy CTC back-translation model | `brltok/ctc.py`, `scripts/train_backtranslate.py` | Done |
| Trained tokenizer files | `runs/tokenizers/` | BPE at 1k, 4k, 8k, 16k and the rule-unit tokenizer |
| Toy results and logs | `runs/*/result.json`, `runs/logs/` | 11 runs plus the liblouis baseline |

Missing from the tarball are the generated data (rebuild it with `scripts/build_data.py`) and the toy checkpoints.
`scripts/queue.sh` contains paths from the machine it was written on, so delete it.

Nothing so far touches a pretrained LLM.
T4, E2, E4, the frozen-body training, LoRA, the output head, the reasoning evaluation, the layerwise analysis, and the Chinese track are all yours.

## Findings that constrain your work

- liblouis 3.32 recovers 97.6% of Grade 2 test sentences exactly on a round trip, so the pipeline baseline is strong on clean English. Expect the pipeline to sit close to print accuracy on clean reasoning benchmarks.
- The outline's milestone 1 rule therefore applies. The case for a learned front end on English rests on noisy input and on the representation analysis, and clean English accuracy is a sanity check.
- Old liblouis builds are badly wrong. The 2017 build gave 41% on the same test. Always record the liblouis version, and never mix versions between training and test data.
- Under dot noise liblouis degrades linearly, at about 1.2 CER points per point of flip rate. A toy cell model trained with 5% flips reached 9.01% CER at 10% flips against 12.05% for liblouis.
- Dot-composed embeddings (E3) matched the table on clean text and were about 5% better (relative) under noise in two seeds. Treat this as a lead to confirm with more seeds.
- Rule units (T3) give only 2.98 tokens per word against 3.27 for cells. BPE at 8k gives 1.33 and matches print token counts.
- From scratch, cells learned fastest and BPE 8k was far behind at equal training examples. Whether this reverses with print-aligned initialization on a pretrained model is the open part of RQ1.
- The training text is 19th-century fiction with almost no numbers. GSM8K is all numbers, so the data for the LLM work must include number-rich and modern text.
- All toy numbers measure agreement with liblouis, because train and test braille both come from it.

## Setup

1. Clone the repository and unpack the tarball so that `brltok/`, `scripts/`, `tests/`, `runs/`, `README.md`, `pyproject.toml`, and `test_sentences.json` sit at the repository root.
2. Put the research outline and this file under `docs/`.
3. Add a `.gitignore` for `data/`, checkpoints, caches, and scheduler logs, then make the first commit.
4. Build liblouis 3.32.0 from the release tarball into a local prefix and export `LD_LIBRARY_PATH`, `LOUIS_TABLEPATH`, and `PYTHONPATH` as the README shows. Keep the three exports in a script that every job sources.
5. Install the Python dependencies in a fresh environment and add `transformers`, `peft`, `datasets`, and `accelerate`.
6. Run `python -m pytest tests`. All 16 tests must pass, including the one that needs liblouis.
7. Rebuild the data with `scripts/build_data.py` and rerun `scripts/eval_liblouis.py`. Clean Grade 2 exact match must come out at 97.6% and CER at 0.35%. If it differs, stop and find out why before going further.

Keep the shipped `runs/` folder as the record of the toy results.
Put all new work under `exp/<phase>/<run_name>/` with a `config.json` and a `result.json` per run.

## Work plan

Do the phases in order.
Stop and write a report at the two checkpoints marked below.

### Phase A. Baselines on pretrained models

1. Pick the base models. Qwen3 1.7B is the main one and Llama 3.1 8B is the second if access exists. Confirm the exact Hugging Face ids yourself.
2. Build the reasoning evaluation for GSM8K and CommonsenseQA with the question rendered in print, Grade 1, and Grade 2. Follow the BrailleBench setup (arXiv 2608.27268). The instructions stay in print and only the question content is braille. Use its released toolkit if you can find it, since it handles the linearization of math. Otherwise render with liblouis tables `en-ueb-g1.ctb` and `en-ueb-g2.ctb`.
3. Measure the stock model on print, on Braille ASCII, and on Unicode braille. BrailleBench reports GSM8K accuracy of 69.4, 2.4, and 1.4 for Qwen3 1.7B and 83.9, 4.6, and 1.4 for Llama 3.1 8B in print, Grade 1, and Grade 2 with Braille ASCII. Your harness should land within a few points of these.
4. Measure the pipeline baseline, where liblouis back-translates the braille question and the stock model answers the result.
5. Repeat the braille and pipeline conditions with dot noise at 2%, 5%, and 10% flips using `brltok.noise.flip_dots` with a fixed seed.
6. Collect a human-transcribed test set. Candidates are listed in the outline's Data section, starting with https://github.com/Tianyuan-Huang/BrailleLLM. Record the license and provenance of anything you download.

Done when one table gives accuracy for every model, script, and noise level, and the gap closed by the pipeline is known.

**Checkpoint 1.** Write the Phase A table and a short reading of it into `RESULTS.md`, commit, and tell Stella before starting any training.

### Phase B. Frozen-body transplant

1. Build the training data. Translate a broad print corpus to Grade 1 and Grade 2 with liblouis and keep the position maps. Include modern prose, instruction data, and number-rich text such as math word problems from training splits. Never include test items from the evaluation sets.
2. Compute the print expansion of every braille token. `RuleUnitTokenizer.expansions` already holds this for T3. For T1 and T2, walk each token's cell span over the stored rule units and collect the print text it covers, giving a unit that a token cuts through to the token that holds its first cell. Keep the counts.
3. Implement E1 (mean initialization), E2 (count-weighted mean of the base model's input embeddings for the tokenized print expansions, with a leading space when the braille token starts a word), and E3 (dot-composed, trained to regress onto the E2 targets first).
4. Feed braille through a separate embedding module and pass the result to the base model with `inputs_embeds`, concatenated with the print embeddings of the surrounding prompt. The base model's tokenizer, embeddings, body, and output head stay untouched.
5. Train stage 1 with the base model frozen. Use a language-modeling loss on braille input followed by print continuation, including transcription and question-answer formats, plus the distillation loss from the outline, which pulls the hidden state at the last token of each braille word toward the frozen model's hidden state at the same print word. Skip any sentence whose braille and print word counts differ.
6. Run the grid of T1, T2 at 8k, and T3 against E1, E2, and E3 on Qwen3 1.7B. Start each with a short run to catch bugs before committing compute.
7. Evaluate every run on the Phase A harness, clean and noisy, and on back-translation exact match and CER.

Prove the freeze in every run.
Hash the base parameters before and after training, and check that print accuracy is identical to Phase A.

Done when the grid is filled in with three seeds for the best two configurations.

**Checkpoint 2.** The go or no-go criterion is closing more than half of the Grade 1 gap on GSM8K with the body frozen. Report the result either way and wait for Stella's decision on Phases C to E.

### Phase C. Encoder, noise, LoRA, analysis

1. Add T4, a small cell-level encoder that pools one vector per word, and compare it with the best table from Phase B.
2. Train the best configurations with noisy input and report the reasoning accuracy under noise against the pipeline. This is the comparison the project's practical case depends on.
3. Add stage 2, LoRA on the lowest layers, and report print accuracy before and after alongside the braille gain.
4. Run the plain baseline from the outline, LoRA on the stock tokenizer with the same data.
5. Do the layerwise analysis for RQ6. Compare hidden states of braille sentences and their print twins per layer, and apply logit lens at braille positions.

### Phase D. Output

Add braille tokens to the output head as stage 3 in the outline.
Check generated braille with a liblouis round trip.

### Phase E. Chinese braille

Repeat the best configuration on Chinese braille with dropped tones, where back-translation is ambiguous and the pipeline should be weak.
Data and baselines come from BrailleLLM and Vision-Braille (arXiv 2407.06048).
Check which Chinese tables liblouis ships before choosing the rule-based baseline.

### Low-priority background work

- Rerun the toy noise comparison of E1 and E3 with five seeds each.
- Train the toy BPE 8k and rule-unit models to convergence to see whether they catch up with cells.
- Add cell drops and insertions to the noise tests, and calibrate rates on optical braille recognition output if you can obtain it.

## Experiment rules

- Fix and log every seed. Report the mean and the spread whenever more than one seed exists, and say so when there is only one.
- Tune on dev sets only. The 500 sentences in `test_sentences.json` and the benchmark test sets are for final numbers.
- Use the same answer extraction and the same prompts across print, braille, and pipeline conditions.
- Report negative and null results with the same care as positive ones. The pipeline may win on clean English, and that is a valid finding.
- Keep the headline metric from the outline. It is the gain in braille accuracy over the stock model divided by the stock model's gap between print and braille accuracy.
- Save the exact command, git commit, liblouis version, and data version with each result.
- Write tests for new library code and keep the existing ones passing.
- Commit small and often on a branch per phase. Do not commit data, checkpoints, or credentials.

## Limits

- Do not release weights, data, or results outside the repository.
- Do not claim any benefit for blind readers. That needs braille readers and transcribers involved, which this project does not have yet.
- Before any model with a braille input layer is shared, measure whether refusal behavior on a standard refusal benchmark is the same for braille prompts as for print prompts, and report it.
- Do not edit the research outline. Results and deviations go in `RESULTS.md` and `NOTES.md`.

## Reporting

Keep two files at the repository root.
`RESULTS.md` holds the result tables per phase with one paragraph of reading under each, and each hypothesis from the outline marked as supported, refuted, or open.
`NOTES.md` holds decisions, assumptions, dead ends, and anything that deviates from the outline, each with a date.

Stella prefers short, plain prose in anything she reads.
State the claim first and the evidence after it, give relative effect sizes, and avoid em-dashes.

## When to stop and ask

- A setup check in step 6 or 7 fails and you cannot explain it.
- The Phase A harness is more than five points away from the BrailleBench numbers after you have checked prompts and answer extraction.
- A job would exceed the compute budget.
- A dataset's license is unclear or forbids research use.
- You reach checkpoint 1 or checkpoint 2.

For everything else, choose the reasonable default, record it in `NOTES.md`, and continue.

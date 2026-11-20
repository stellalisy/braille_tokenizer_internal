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

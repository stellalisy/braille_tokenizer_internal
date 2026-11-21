# Braille tokenizer: project overview

Nov 20, 2026. Internal, do not share outside the repository. Details live in `docs/Braille tokenizer research outline.md` (the plan), `RESULTS.md` (numbers) and `NOTES.md` (decisions).

## One paragraph

LLMs that answer a question well in print fail when the same question arrives in braille: Qwen3 1.7B drops from 83% to 3% on GSM8K in Grade 1 Braille ASCII. The model already has the knowledge, so the failure is in how braille gets in. We attach a new braille tokenizer and input layer to a frozen pretrained LLM, train only the new parameters on free synthetic parallel data from liblouis, and measure how much of the print-to-braille gap closes. The first results reshape the project. A rule-based pipeline (liblouis back-translation, then the unchanged model) already closes 98% to 100% of the gap on clean English, but it collapses under dot noise, losing about half its GSM8K accuracy at 5% flipped dots. Noise robustness, languages whose braille drops information, and the analysis of how a frozen model absorbs a new script are where the project now earns its place.

## Intellectual motivation

Braille is a second orthography for languages the model already knows. That makes it an unusually clean test of a general question: how much of a model's ability is tied to the surface form of its input, and how cheaply can a new input vocabulary be connected to the representations it already has?

Three properties make braille a good laboratory for this.

1. **Exact supervision.** liblouis converts any print corpus into braille and returns character-level alignments, so parallel data is unlimited and every braille token has a known print expansion.
2. **Known structure.** Each cell is six dots, and the mapping from cells to print is a published rule system (UEB). Every model error can be traced to a rule, and dot-level structure can be built into embeddings.
3. **Controlled difficulty.** Grade 1 is close to a letter-by-letter transliteration, while Grade 2 adds contractions whose meaning depends on context. The same content can be presented at two levels of difficulty with nothing else changed.

With the model body frozen, print behavior cannot change, so every gain on braille comes from the new input layer. That isolates the input question from the usual confound of fine-tuning the whole model.

## Why it matters

- **A large, unexplained failure.** BrailleBench (Zhang et al., 2026) shows strong models losing most of their reasoning ability on braille input, and names better braille tokenization as future work. Stock tokenizers spend about three byte tokens per Unicode braille cell, and Grade 2 Braille ASCII costs 1.5x the tokens of print despite having 29% fewer cells.
- **A general method question.** Attaching a new script, modality or vocabulary to a frozen model is a recurring problem (new languages, transliteration, encoders for speech). Braille gives it exact ground truth, so initialization and tokenizer choices can be compared cleanly.
- **Cases where rules are not enough.** Rule-based translation passes every wrong dot through, and it cannot resolve Chinese braille where writers omit most tones. These are the settings where a learned reader has something to add.
- **Safety.** Fluent braille input is an encoding channel around safety training, as cipher jailbreaks show. Any model that reads braille well needs its refusal behavior measured before it is shared.

We make no claim that this work benefits blind readers. Braille users today are served by screen readers and displays driven by rule-based translators, and any accessibility claim would need braille readers and transcribers involved in choosing tasks and judging output.

## Novelty and contributions

Prior braille work (BrailleLLM, Vision-Braille, BT5) adds braille tokens and fine-tunes, usually the whole model or with LoRA, and evaluates translation. What this project adds:

1. **A controlled comparison of braille tokenizers** (cells, braille BPE, rule units, learned patches) crossed with embedding schemes (mean init, print-aligned init, dot-composed, hypernetwork), with the base model frozen.
2. **Transfer of reasoning, not only translation.** The headline metric is the share of the print-to-braille gap closed on GSM8K and CommonsenseQA by the same frozen model.
3. **Dot-composed embeddings**, which build a cell vector from its six dot bits. New here, and aimed at robustness to dot errors.
4. **A properly measured rule-based baseline.** liblouis 3.32 round-trips 97.6% of Grade 2 sentences exactly (an old JavaScript build gave 41%), and the full pipeline closes 98% to 100% of the gap on clean English. Prior work either omits this baseline or reports a weak one.
5. **Corrections to the BrailleBench evaluation.** Three artifacts change its numbers: Qwen3 thinking mode truncated at 1,024 tokens, the stripping of `$` signs before rendering, and an exact-match scorer that gives zero to answers written as a choice letter. The last one fully explains its Llama CommonsenseQA print score (43.4 reported, 72.2 with a choice-based scorer, reproduced from its released code).
6. **Representation analysis (planned).** Where braille and print hidden states converge inside a frozen model, by layerwise similarity and logit lens.

## What is done

**Infrastructure.** The `brltok` package (tokenizers T1 to T3, dot noise, choice scorer, with tests), liblouis 3.32 on the cluster, a vLLM evaluation harness that uses BrailleBench's own prompts and extraction, saves every 256 items and resumes, and records the command, commit, model revision and liblouis version with each result. Our braille rendering matches BrailleBench's for all 5,078 questions.

**Toy models (CPU, from scratch).** Small back-translators confirm the tokenizer pipeline. BPE at 8k reaches print sequence length (1.03x GPT-4 tokens), while rule units barely shorten sequences. Trained with noise, a cell model beats liblouis from 2% flips upward (28% lower CER at 10% flips), and dot-composed embeddings beat a plain table by 4.8% relative at 10% flips in all five seeds.

**Phase A: stock models and the pipeline** (Qwen3 1.7B, Llama 3.1 8B Instruct, 1,024-token runs complete).

| Finding | Number |
|---|---|
| Harness reproduces BrailleBench | within 3 points everywhere (Qwen thinking on: 69.5 / 1.4 / 1.8 vs 69.4 / 2.4 / 1.4) |
| Stock models on braille | GSM8K 1% to 5% in Braille ASCII, about 0% to 1% in Unicode |
| Pipeline on clean English | closes 98% to 100% of the gap, both models, both grades |
| Pipeline under dot noise | Qwen GSM8K G1: 81.7, 61.9, 41.7, 21.3 at 0, 2, 5, 10% flips |
| CommonsenseQA with a choice scorer | Grade 1 ASCII is near print (Qwen 61.8 vs 68.3); the real gap is Grade 2, which sits near chance (21.5) |

The pipeline result triggers the outline's milestone 1 rule: on clean English the learned front end cannot beat rules, so the English work narrows to noise robustness and analysis.

## In progress

- 8k-token Phase A reruns for Qwen (thinking off) and Llama, and 32k-token runs for Qwen with thinking on (16 GSM8K conditions as separate jobs). All 20 jobs are running. These remove truncation as a confound.
- Then: rescore CommonsenseQA with the choice scorer, update `RESULTS.md` and the Overleaf tables.
- Phase A step 6: a human-transcribed test set, starting with BrailleLLM's expert-corrected data, so evaluation is not only agreement with liblouis.

## Planned

| Phase | Work | Status |
|---|---|---|
| Checkpoint 1 | Report Phase A; decide thinking mode for training (recommend off) and approve GPU training | waiting on the reruns |
| B. Frozen-body transplant | Train T1, T2 (8k), T3 × E1, E2, E3 on Qwen3 1.7B with the body frozen; prove the freeze by hashing weights and matching print accuracy | needs GPU approval |
| Checkpoint 2 | Go if a frozen-body model closes more than half of the Grade 1 GSM8K gap | |
| C. Encoder, noise, LoRA, analysis | Cell-level encoder (T4); noise-trained models against the pipeline (the main practical comparison); LoRA on low layers; LoRA on the stock tokenizer as a baseline; layerwise analysis | after checkpoint 2 |
| D. Output | Braille output head, checked by liblouis round trip | |
| E. Chinese braille | Best configuration on Chinese braille with dropped tones, where the pipeline should be weak | |
| Paper | Workshop-length paper; outline and tables in Overleaf | outline written, tables partly filled |

## Main risks

- Clean English is already solved by rules, so the practical case rests on noisy input and Chinese. If a frozen-body model cannot beat the pipeline under noise, the contribution becomes the analysis and the evaluation corrections.
- Training and testing on liblouis output measures agreement with liblouis. Human-transcribed test data is the guard.
- The noise model (one flipped dot per corrupted cell) is simulated. Rates should be calibrated on optical braille recognition output.
- BrailleLLM already initializes braille tokens from print, so novelty rests on the controlled comparison, the frozen body, reasoning transfer, dot-composed embeddings and the analysis.

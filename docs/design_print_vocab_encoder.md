# Design note: reading braille into the model's own vocabulary

Status: proposal for review, 2026-11-22. Not started. Internal.

## Claim this design is meant to support

A frozen language model reads braille almost as well as print when a small encoder turns each braille word into a distribution over the model's own print tokens. Clean braille then reaches the model as the subword pieces it already reads, including words never seen in training, and noisy braille arrives as a mix of plausible readings instead of one committed guess.

## Why the current design falls short

Stage 1 (one learned vector per braille unit, trained through the frozen model) closes 93% of the Grade 1 GSM8K gap but loses 15 points on Grade 1 CommonsenseQA with thinking off. The diagnostics in NOTES.md (2026-11-22) locate the loss in misread words:

- Misreads track how rare a word is and how many print tokens it has, not how braille is segmented. Grade 1, rule units, thinking traces: 44% of options with a word unseen in training are misread, against 2 to 4% for words seen over 100 times; 4% for one-print-token words, 45% for four or more. Braille units per word barely matter (3 to 7%).
- Swapping the segmentation changes little. Cells, BPE and rule units misread 6.3%, 3.5% and 5.2% of options (Grade 1); none is near print.
- Without thinking, the answer is read off the braille states directly and the model is unsure (top-option probability 0.65 against 0.98 in print). With thinking, it restates the item in print first and the gap shrinks to 4 points, which suggests the model does best when it gets print tokens.
- Matching hidden states harder does not help: distilling the answer-position states to cosine 0.98 left accuracy unchanged.

The common cause: every braille unit is a new vector the frozen model never saw in pretraining, so it must learn to compose them, and it only learns that for words that are common in training. Print gives rare words as known subword pieces.

## Design

### Interface to the model

For each braille word w (cells between blank cells, punctuation included), the encoder produces K_w vectors, where K_w is the number of print tokens the model's tokenizer gives the print form of w with its leading space. Each vector is

    x_k = sum_v p_k(v) E[v]

where E is the model's frozen input embedding matrix and p_k is a distribution over the print vocabulary. Print segments of the prompt keep their ordinary embeddings, so splicing works as it does now (`brltok/splice.py`).

- Clean input with a confident encoder gives x_k close to a real token embedding, so the model sees its native input.
- Uncertain input gives a convex mix of real embeddings, which stays in the region of embedding space the model was trained on, unlike a free vector.

### Encoder

- Input: the braille span as cells. Each cell is embedded from its six dot bits (linear map) plus a learned cell embedding, as in E3, so a flipped dot moves the input a little instead of to an unrelated id.
- Context: a small transformer over the cells of the whole span (4 layers, width 512, about 15M parameters). Context is needed for Grade 2, where some contractions read differently by position and neighbors.
- Output per word: a small autoregressive decoder (2 layers, width 512) attends to the word's cells, with the rest of the span as context, and emits print tokens until an end-of-word symbol. The output layer is tied to E (scores are h · E[v]), so no new 151k-row matrix is trained, and the prediction is directly in the model's vocabulary.
- Length: teacher-forced to the gold token count in training; at inference the decoder stops at end-of-word (greedy over the end decision, soft over token identity).

### Training

Stage A, supervised reading (no language model in the loop):

- Pairs of braille word in context and its print tokens, from the existing 284k training sentences (`data/phaseB/train.jsonl`, aligned per word) plus a word list for coverage of rare words.
- The word list comes from sources independent of the test sets (training corpus vocabulary, plus a public English word-frequency list). Words that appear only in test items are not added on purpose; unseen-word results are reported against the training vocabulary.
- Loss: cross-entropy per print token, with label smoothing 0.05 so that probabilities stay calibrated.
- Noise augmentation: dot flips at rates drawn from U[0, 10%] using `brltok.noise.flip_dots`, label always the clean print.
- Cheap: a few GPU-hours at most, mostly much less.

Stage B, through the frozen model:

- The current objectives: LM loss on the self-distilled QA answers and transcription targets, plus word-level distillation, with x_k fed as input embeddings. Gradients reach the encoder through p_k.
- Stage A's cross-entropy stays on as a regularizer so that p_k keeps meaning print tokens.
- Whether Stage B helps at all is itself a question for the pilot (ablation: Stage A only).

### What is learned versus rule-based

At inference nothing calls liblouis. Liblouis is used only to make Grade 2 training pairs, as now. This is still a learned back-translator in effect, so the paper's case against the pipeline rests on two things it can measure: graceful behavior under noise (soft mixes instead of hard wrong words), and a single forward pass with no rule tables. It should not claim to beat the pipeline on clean braille.

## Measurements

Before the language model (fast, per checkpoint):

- Token accuracy and word exact match on the CommonsenseQA and GSM8K test words, split by seen and unseen in training and by print token count. Target for Grade 1 clean: at least 99% on seen words and 95% on unseen words.
- The same under dot noise at 2, 5 and 10%, against the liblouis back-translation of the same noisy braille.
- Calibration: probability of the top token when right and when wrong.

Through the model (existing scripts):

- GSM8K and CommonsenseQA, Grades 1 and 2, thinking off and on (`phaseB_eval.py`).
- Misread rate in thinking traces (`scripts/analysis/phaseB_tok_misreads.py`) and option probabilities (`phaseB_choiceprobs.py`).
- Noise grid against the pipeline, as in the stage 1 table.

Baselines: print, stock tokenizer, liblouis pipeline, and stage 1 rule units with print-aligned init (T3 E2), which becomes the ablation for "new vectors per unit".

## Pilot and decision points

1. Grade 1, Stage A only. Check word accuracy on unseen words. If it is under 90% clean, fix the encoder before going further.
2. Grade 1, Stage A only, through the frozen model. If CommonsenseQA with thinking off reaches 65 or more (print 68.3, T3 E2 54.0), the design works without any LLM-in-the-loop training.
3. Grade 1 with Stage B. Keep Stage B only if it adds at least 1 point.
4. Grade 2, the same steps, since contractions are where context and learning matter.
5. Noise grid. The design earns its place against the pipeline if it is ahead at 5% and 10% flips on both datasets.

Ablations once the pilot works: soft versus argmax (hard) tokens; no context (word-only encoder); dot-bit input versus cell ids; rule units as encoder input instead of cells.

Estimated cost of the pilot (steps 1 to 4, one seed): 6 to 10 GPU-hours. Phase B has used 14.7.

## Risks

- Grade 2 ambiguity: some contractions need context beyond the word. The span-level encoder covers this, and errors will be reported by contraction type.
- Length errors: a wrong end-of-word decision shifts tokens. Measured directly as word exact match.
- Soft mixes may still confuse the model when they are spread out. The calibration and noise results will show whether the mix helps or hurts compared with argmax.
- Novelty: the method is close to neural back-translation. The paper should present it as a design question with a measured answer (does reading into the model's vocabulary fix misreads, and does keeping it soft help under noise), not as a new kind of tokenizer.

## Changes elsewhere

- Code: a new `brltok/vocab_encoder.py` (encoder, decoder, soft-embedding interface, with tests), a Stage A training script, and an option in `phaseB_train.py` and `phaseB_eval.py` to embed braille spans with the encoder instead of a table.
- Paper: the method section would center on this encoder, with rule units as the comparison. The research outline is not edited; this note and NOTES.md record the change.

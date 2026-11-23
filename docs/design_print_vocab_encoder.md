# Design note: reading braille into the model's own vocabulary

Status: revision 2, 2026-11-22. Replaces revision 1 (commit c93d60d). Internal.

Revision 1 proposed an encoder that reads each braille word into a distribution over the frozen model's print tokens. Stage A of that design is trained and evaluated (NOTES.md, 2026-11-22). It reads common words well and is far more robust to dot flips than liblouis, but it does not read letters: it recognizes words and lets sentence context override the cells. This revision restates the objective from first principles and changes the encoder, the training data and the noise model to match it.

## What Stage A showed

- Test-set words, clean: 98 to 99.7% exact, within 0.7 points of liblouis. At 10% dot flips: 90 to 93% against 62 to 71% for liblouis.
- Copy probe (clean braille, greedy exact copy, Grade 1): real held-out words 95% alone and 69% inside a sentence; one-letter typos 52% and 27%; random letter strings 26% and 11%. In a sentence, 58% of typos come out as a different known word.
- The "unseen" test words are mostly source misspellings and rare names. The encoder returns the intended word (libary as library) or a different one (laddle as paddle, Locsin as Lopez).
- Real photos read by an optical recognizer (401 images): the encoder beats liblouis in every band of recognizer quality (26.0 vs 21.8 overall), but the recognizer's errors are mostly inserted, deleted and duplicated cells, which training never showed it. Single-dot flips are about 1% of recognizer errors.

Cause: Stage A is solvable by recognizing words. Every target is a real word, and in real sentences the context always agrees with the cells, so the encoder learned a lexical and contextual prior instead of reading.

## Objective, from first principles

1. With a frozen model, print is the ceiling and print embeddings are the way to reach it. Braille encodes print and carries no extra meaning; the model's knowledge is indexed by print token embeddings. An input layer that closes the whole gap is in effect a transcriber into print embeddings.
2. "Same as print" defines the target only for unambiguous braille. For noisy braille there is a distribution over the print text that produced it, and the target is the model behaving as if reading that text, weighted by its probability.
3. That posterior has two parts: how well a candidate print text explains the cells (likelihood: braille rules plus the braille-side error process) and how plausible the candidate is in context (prior). The frozen model is a far better prior than anything we can train. So the reader supplies the likelihood and the model supplies the prior.
4. Consequences for the reader: copy exactly when the cells are clean, including typos and unfamiliar names; spread probability over plausible readings when the cells look corrupted; never choose words from sentence context.

## Two sources of error, only one of which the reader models

- Source-text errors, such as "diffrent" typed by the print author, exist before braille translation. Print shows the same error, and the model learned to cope with such errors from pretraining text, as text models do: text tokenizers never correct typos, they segment faithfully, and robustness comes from the model seeing misspellings in context. The reader's job here is only to copy, so the braille condition gives the model exactly what print gives it.
- Braille-side errors arise after translation, in writing, on the page or in optical recognition. Print never has them and the model never saw them, so the reader must handle them.

The two distributions do not line up. Contractions break the link between print and braille edit distance: in Grade 2, "different" is ⠙⠊⠖⠻⠢⠞ and "diffrent" is ⠙⠊⠖⠗⠢⠞, one cell three dots apart, and "the" is the single cell ⠮ while "teh" is ⠞⠑⠓. Languages with heavier contraction systems separate them further. Braille-side errors follow dot and cell structure instead of keyboard layout.

## What is known about braille-side errors

Evidence is thin and from small studies; rates below are for orientation, not to be copied as parameters.

- Writing on six-key keyboards (Perkins-style braillers, notetakers, touchscreen braille). Dot errors are the largest category of writing errors (46% in a JVIB 2007 case study), mostly one dot or a few dots. Chorded input mostly fails by a finger missing a key (dot omitted), an extra finger (dot added), or a neighbouring dot pressed instead (dot 3 for dot 2). Space errors are a separate category. Omission is less likely in the right column and in short words (PubMed 36508777). Uncorrected error rates of 3.5 to 11% per character are reported for different input methods.
- Mirror and inversion errors (e and i, d and f) are reported for beginning braillists and slate-and-stylus writers, where cells are written reversed, but only qualitatively.
- Optical recognition (measured here on 401 real photos): insertions, often repeated cells, are about a third of cell edits, then unequal-length replacements, deletions, multi-dot substitutions and space errors.
- Physical wear flattens raised dots, which is dot omission.
- Tactile confusion matrices for human readers (d, f, h; u, v, x; m, n, p) describe how people misread braille by touch. They are not our input channel and are not used.

## Design changes

1. Word-local reader. Each word's decoder attends only to that word's cells; the cell encoder's attention is masked to the word. Grade 2 contractions whose reading depends on position are decided within the word, which covers most of the code; the few cases that need neighbouring words are measured, not assumed away.
2. Dot-pattern input, as now: six dot bits through a linear map plus a cell embedding, so related cells (one dot apart, mirror images) start close.
3. Copying is required by the data. Training targets keep the source spelling exactly. The training text keeps its natural typos, and synthetic non-words are added: one-letter edits of real words (substitute, insert, delete, swap) and random letter strings, alone and in sentences, labelled with their own spelling. This is how text models get robust: from data, not from a designed correction step.
4. Output that can spell any string. The decoder keeps the print vocabulary (tied to the frozen embeddings) but must be able to emit any letter sequence; the random-string probe is the check. If token-level decoding cannot reach it, add a character-level path (single-character print tokens are already in the vocabulary) as a fallback inside the decoder, not an external one.
5. A braille-side noise model in place of uniform dot flips. Per cell: dot omission more likely than addition, rates varying by dot position; neighbouring-dot substitution; mirror swaps at a low rate. Per span: cell insertion with a bias to repeating the previous cell, cell deletion, space insertion and deletion. Recognizer rates are estimated from detector output against gold cells on the Angelina dataset (separate from the 401-image test set). Writing rates come from the studies above where they give numbers and are otherwise set conservatively and reported as assumptions.
6. Calibrated output. Label smoothing stays. On clean input the reader should be confident; on corrupted input its top probability should match its accuracy. Measured, not assumed.

Unchanged from revision 1: the interface (soft mixes x_k = sum_v p_k(v) E[v] spliced into the prompt), Stage B through the frozen model as an optional second stage, no liblouis at inference.

## Acceptance tests

Before the model, on held-out items:

- Copy probe, clean, alone and inside sentences: real held-out words, one-letter typos and random strings. Target: within a few points of each other and of 100%, and no gap between alone and in sentence. This is the main test of faithfulness.
- Test-set words under the braille-side noise model and under uniform dot flips, against liblouis. Target: keep most of Stage A's noise advantage; any loss is the price of faithfulness and is reported as such.
- Calibration: expected calibration error of the top token, clean and noisy.
- Real photos: word accuracy by recognizer-quality band, against liblouis and against Stage A.

Through the model:

- Clean Grades 1 and 2, CommonsenseQA and GSM8K, thinking off and on, against print (the ceiling) and the pipeline.
- Noise: does the soft mix let the model resolve ambiguity? Compare soft, hard (argmax) and an n-best rescoring reference in which the model scores the reader's top candidate readings. If soft is no better than hard, the interface, not the reader, is the open problem.

## Plan

1. Implement the word-local mask, the non-word data and the new noise model (with tests). Measure the recognizer channel on Angelina.
2. Grade 1 Stage A, four runs as 5,000-step fine-tunes from the current encoder: control (more steps, no change), word-local only, non-word data only, both. One more full run from scratch for whichever wins, to rule out leftovers from Stage A's prior. Acceptance tests before the model.
3. Grade 2, the winning configuration.
4. Through the model: clean, then noise and real photos, soft against hard against rescoring.

Estimated cost: steps 2 and 3 about 6 to 8 GPU-hours; step 4 about 4 to 6.

## Claim this design is meant to support

On clean braille the reader should match print and the pipeline; it cannot beat liblouis on braille liblouis generated, and the paper does not claim to. The claim is about braille liblouis did not generate: a faithful, calibrated word-local reader, trained on braille-side errors, passes the uncertainty of real or noisy braille to the frozen model, and the model's own context resolves it. Both halves are measurable: faithfulness by the copy probe, and the value of passing uncertainty by soft against hard against rescoring under noise and on real photos.

## Risks

- Faithfulness costs noise robustness. Some of Stage A's advantage under dot flips came from its prior. If the word-local reader loses much of it and the model does not make it up through context, the method's case narrows.
- Soft mixes may not be read as "one of these". The model is nonlinear, so a mix of two embeddings need not behave like a mix of two readings. The rescoring reference shows how much is lost.
- The writing-error evidence is thin. Rates that are not measured are reported as assumptions, and results are shown across a range of rates.
- The real-photo test set is small (1,342 label words) and dominated by recognizer failures. Results are reported by recognizer-quality band.

## Sources on braille-side errors

- A cognitive approach to brailling, JVIB 101(7), 2007: https://files.eric.ed.gov/fulltext/EJ772093.pdf
- Phonological type errors in braille spelling: https://pubmed.ncbi.nlm.nih.gov/36508777/
- An evaluation of BrailleTouch: https://dl.acm.org/doi/pdf/10.1145/2371574.2371623
- Enabling auto-correction on a soft braille keyboard: https://pmc.ncbi.nlm.nih.gov/articles/PMC12723526
- Evaluation of the BrailleEnter keyboard: https://www.researchgate.net/publication/320651818
- Reversal errors in young braille readers (ICEVI 2000): https://icevi-europe.org/cracow2000/proceedings/chapter04/04-11.doc
- Perceptual similarity in braille letter recognition (2025), not used as a channel: https://link.springer.com/article/10.1186/s41235-025-00690-x

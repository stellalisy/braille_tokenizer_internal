# Braille tokenizer: research outline

Nov 20, 2026 · @Stella

## Problem

Braille is a second orthography for languages that pretrained models already know, yet general LLMs lose much of their ability when the same text arrives as braille. On [BrailleBench](https://arxiv.org/pdf/2608.27268), Claude Opus 4.8 scores 97.3 on GSM8K in print, 89.8 when the question is in uncontracted (Grade 1) braille, and 67.9 in contracted (Grade 2) braille. Claude Haiku 4.5 drops from 96.2 to 35.6 and 11.7. The model already has the knowledge, so the loss comes from how braille is read in.

This project builds a braille tokenizer and input layer, then attaches them to a frozen pretrained LLM so that braille reaches the representations the model already uses for print. Parallel data is free because [liblouis](https://github.com/liblouis/liblouis) converts any print corpus into braille. The print-to-braille mapping is known, so every error can be traced to a rule. The new parameters are small enough to train on one GPU, which keeps the work side-project sized.

A rule-based back-translator in front of an unmodified LLM is the baseline to beat for English reading tasks. A learned front end earns its place on 1) languages whose braille drops information, such as Chinese, where writers omit most tone marks and a toneless syllable maps to many characters, 2) noisy input from optical braille recognition, where rules pass every wrong dot through and a learned reader can repair some from context, 3) tasks about the braille itself, such as proofreading a transcription or fitting text to a 40-cell display, and 4) the general question of how to attach a new input vocabulary to a frozen model, for which braille gives exact supervision.

## Background

A 6-dot cell has 63 non-blank patterns, and the meaning of a pattern depends on the braille code and on the neighboring cells. Segmenting text into cells is therefore trivial, and the real tokenization question is which units above the cell to use and how to embed them.

- Grade 1 English spells words letter by letter and marks capitals and numbers with indicator cells, and digits reuse the cells for a to j.
- Grade 2 Unified English Braille (UEB) adds contractions, so one cell can be a letter, a whole word, or part of a word. In the pilot text, ⠅ is "k" inside a word and "knowledge" when it stands alone. UEB also has [75 shortforms](https://iceb.org/RUEB%20Appendix1%20Unicode.pdf) such as "ab" for "about".
- Chinese braille writes sound with up to three cells per syllable (consonant, vowel, tone), and writers drop most tones, so reading it back into characters needs a language model ([Vision-Braille](https://arxiv.org/pdf/2407.06048)).
- Digital braille comes as Unicode (U+2800 block, three UTF-8 bytes per cell), Braille ASCII (one printable character per cell, used in BRF files), or dot-number notation.

### Pilot measurements

A pilot used 500 held-out sentences from four public-domain novels, translated with liblouis UEB tables. The table gives token count relative to the same sentences in print.

| Tokenizer | Unicode G1 | Unicode G2 | ASCII G1 | ASCII G2 |
| --- | --- | --- | --- | --- |
| cl100k (GPT-4) | 10.4x | 7.5x | 1.09x | 1.54x |
| o200k (GPT-4o) | 10.8x | 7.7x | 1.10x | 1.55x |
| Llama 3 | 10.4x | 7.5x | 1.09x | 1.54x |
| Mistral 7B | 10.6x | 7.7x | 1.10x | 1.56x |
| Llama 2 | 10.4x | 7.6x | 1.10x | 1.53x |

- None of the 63 cells is a single token in cl100k, o200k, or Llama 3. Each cell costs about three tokens, one per UTF-8 byte, and its identity sits in the last byte, a token shared with every other multi-byte script.
- Braille ASCII is cheap for Grade 1 but Grade 2 costs 1.5x print even though it has 29% fewer cells, because contracted strings such as "extraord9>y4" split into fragments that carry no meaning.
- A cell-level tokenizer is 2.5x longer than print BPE on Grade 2. A braille BPE trained on 30,000 Grade 2 sentences reaches print length at an 8k vocabulary (0.97x).
- Round-trip back-translation with liblouis 3.32 recovers 99.2% of Grade 1 and 97.6% of Grade 2 sentences exactly. The 2017 JavaScript build of liblouis 3.2.0 used in the first pilot gave 74% and 41%, so old builds badly understate the rule-based baseline.

## Related work

Braille-specific papers have already added braille tokens to LLMs, but none compares tokenizer designs, keeps the base model frozen, or tests whether reasoning ability transfers to braille input.

### Braille NLP

| Work | What it does | What it leaves open |
| --- | --- | --- |
| [BrailleBench](https://arxiv.org/pdf/2608.27268) (Zhang et al., 2026) | Tests six LLMs on math and QA with UEB Grade 1 and 2 input and output. Braille ASCII is the main format, and Opus scores 0 on every tested task in Unicode braille. | Evaluation only. It names better braille tokenization as future work and reports no token counts. |
| [BrailleLLM](https://arxiv.org/html/2510.18288v1) (Huang et al., EMNLP 2025) | Adds braille-fragment tokens to Qwen2.5-7B and initializes each from the pinyin syllable or English word it maps to, then trains LoRA. The gain over plain fine-tuning is 19 BLEU at 3K samples. | The fragment vocabulary comes from a hand-built knowledge base. There is no rule-based baseline and no test of downstream reasoning. |
| [Vision-Braille](https://arxiv.org/pdf/2407.06048) (Wu et al.) | Adds 63 cell tokens to Qwen3-8B and fully fine-tunes with a tone-dropping curriculum, reaching 83.3 BLEU on passages with 10% of tones kept. | Cell tokens start from random embeddings and the whole model is updated. A LoRA variant that keeps C-Eval at 77.8% loses about 5 BLEU. |
| [BT5](https://aclanthology.org/2026.ltedi-1.8/) (Abdullah, LTEDI 2026) | Fine-tunes T5-small on Korean print and braille pairs and beats prompted LLMs. | It attributes LLM failure to missing braille-aware tokenization without testing a tokenizer. |
| [Grade 2 with ByT5](https://prasanthmj.github.io/ai/grade-2-braille-transformer) (blog) | T5's tokenizer maps every braille cell to the unknown token. 64 randomly initialized cell tokens reach 2.9% exact match, and byte-level ByT5 reaches 92.9% on 42 hand-transcribed sentences. | Small scale. Its liblouis baseline of 9.5% exact match is far below what the tool's reputation suggests and needs an independent check. |
| [Yu et al., 2023](https://doi.org/10.1016/j.displa.2023.102506) | Pre-trains a Transformer for Chinese and braille translation on 10k parallel pairs. | Trained from scratch, with no reuse of a general LLM. |
| [Kitsunai et al., 2024](https://www.anlp.jp/proceedings/annual_meeting/2024/pdf_dir/P8-20.pdf) | Prompts LLMs for kanji-to-kana conversion in Japanese braille translation. | Japanese braille is kana only, so reading it back into kanji has the same homophone problem as Chinese. |

### Attaching a new vocabulary to a pretrained model

| Idea | Source | Use here |
| --- | --- | --- |
| Initialize new tokens at the mean of existing embeddings | [Hewitt, 2021](https://www.cs.columbia.edu/~johnhew//vocab-expansion.html) | Default baseline for new cell tokens. |
| Freeze the body and learn only new token embeddings | [Artetxe et al., 2020](https://arxiv.org/abs/1910.11856v2) | Simplest form of the extension. [Later work](https://arxiv.org/pdf/2309.04679) finds it fails with random initialization on multilingual models, so initialization matters. |
| Initialize target tokens from aligned source tokens | [Trans-tokenization](https://arxiv.org/abs/2408.04303), [SambaLingo](https://arxiv.org/pdf/2404.05829) | liblouis gives exact braille-to-print alignments, so the mapping is free. |
| Hypernetwork that predicts embeddings for any tokenizer | [ZeTT](https://arxiv.org/abs/2405.07883v1) (Minixhofer et al., 2024) | Zero-shot baseline that needs no braille training, available for the base models with a released hypernetwork. |
| Byte-level input with a small local encoder that pools into patches | [ByT5](https://arxiv.org/abs/2105.13626), [BLT](https://arxiv.org/abs/2412.09871v1) | Template for a cell-level encoder in front of a frozen body. |
| Trainable projector between a frozen encoder and a frozen LLM | [SLAM-ASR](https://arxiv.org/abs/2402.08846v1) | Treats braille as a modality with a light adapter. |
| Transliterate an unseen script into one the model knows | [RomanSetu](https://arxiv.org/html/2401.14280v1), [Purkayastha et al., 2023](https://arxiv.org/abs/2304.08865v1) | Grade 1 Braille ASCII is close to romanization, which could explain why ASCII beats Unicode. |
| Encode below the symbol, then learn subwords | [SubChar tokenization](https://arxiv.org/abs/2106.00400) (Si et al., 2023) | Precedent for BPE over cells. |
| Subword embeddings encode spelling | [Kaushal and Mahowald, 2022](https://arxiv.org/abs/2206.02608v1) | Supports initializing letter-level braille tokens from print embeddings. |
| Ciphered text and logit lens | [Fang and Marks, 2025](https://arxiv.org/abs/2512.01222v1), [All Code, No Thought](https://arxiv.org/pdf/2510.09714) | Models can translate ciphered text yet reason poorly in less common ciphers, and the ability tracks how often a cipher appears in pretraining data. Logit lens is the analysis tool for checking whether braille maps onto print representations. |

Neuroimaging gives a parallel question. Tactile braille reading in blind adults activates the ventral occipitotemporal region that print reading uses, but [without the same posterior-to-anterior word-form gradient](https://www.biorxiv.org/content/10.1101/2021.08.24.457544v1.full-text), and [visual experience shapes that region's specialization](https://pubmed.ncbi.nlm.nih.gov/29061700/). The model analogue asks at which layer braille and print converge.

## Research questions

1. **Which unit above the cell should a braille tokenizer use?** Candidates are single cells, BPE merges learned on braille, and units derived from the translation rules. We expect BPE or rule units at a vocabulary near 8k to match print sequence length and to beat cells once their embeddings are initialized from print.
2. **Does the dot structure of a cell help?** We expect embeddings composed from the six dot bits to add little on clean text, because 63 cells are easy to learn from unlimited synthetic data, but to improve robustness when dots are flipped or missing.
3. **Can a frozen LLM read braille through new input parameters alone?** We expect print-aligned initialization plus embedding-only training to close most of the Grade 1 gap on reasoning tasks. Grade 2 should need either word-boundary-aware tokens or a small contextual encoder, because the same cell means different things alone and inside a word.
4. **Where does a learned front end beat liblouis back-translation followed by the same LLM?** We expect the pipeline to match or win on clean English, and the learned front end to win on noisy input and on Chinese braille with dropped tones.
5. **Can the model also write valid braille?** BrailleBench finds output easier than input for Grade 2, so this is a secondary goal.
6. **Where do braille and print representations converge inside the model?** With a frozen body, we expect hidden states for a braille sentence and its print twin to align within the first few layers, and logit lens to recover print tokens at braille positions.

## Tokenizer and embedding design

The design separates the unit the tokenizer emits from the way each unit gets its vector, so the two can be ablated independently.

### Units

| Variant | Unit | Vocabulary | Why test it |
| --- | --- | --- | --- |
| T0 Stock tokenizer | The base model's own BPE over Braille ASCII or Unicode | unchanged | Baseline. Grade 2 costs 1.5x print in ASCII and 7.5x in Unicode. |
| T1 Cell | One token per cell | 64, or 256 for 8-dot | Smallest vocabulary, at 2.5x print length. Vision-Braille uses this. |
| T2 Braille BPE | Merges learned on liblouis output, with a word-start marker | 4k to 16k | Reaches print length at 8k in the pilot. The marker gives a stand-alone wordsign and the same cell inside a word different tokens. |
| T3 Rule units | One token per contraction, indicator, or letter, segmented with the input-to-output position maps that liblouis returns | set by the rule table | Every token has a known print expansion, which makes aligned initialization exact. |
| T4 Learned patches | A small encoder reads cells and pools one vector per word | 64 at the input | No fixed vocabulary above the cell, in the style of BLT. |

Capital, numeric, and grade 1 indicators change how the following cells are read. The tokenizer can keep them as separate tokens or fold them into the next unit, and both options are cheap to try under T2 and T3.

### Embeddings

| Variant | How a token gets its vector | Source idea |
| --- | --- | --- |
| E1 Table, mean init | Lookup table initialized at the mean of existing embeddings | Hewitt, 2021 |
| E2 Table, print-aligned init | Each braille token starts at the mean of the base model's embeddings for the print tokens it expands to | BrailleLLM, trans-tokenization, SambaLingo |
| E3 Dot-composed | A cell vector is a linear map of its six dot bits plus a learned residual, and a multi-cell token runs a tiny encoder over its cells | New here |
| E4 Hypernetwork | ZeTT predicts the embedding from each token's print expansion, with no braille training. Released hypernetworks cover Mistral-7B and TinyLlama, so this variant runs on one of those | ZeTT |

E2 and E3 can be combined by training the E3 encoder to regress onto the E2 targets before any task training.

## Attaching to a pretrained model

The base model's body and print embeddings stay frozen in the first stage, so print behavior cannot change and every gain on braille comes from the new input layer.

&#91;embedded content: architecture · two input paths into one frozen model\]

Print text takes the original path, and braille reaches the same frozen body through its own tokenizer and embeddings.

1. Stage 1 trains the input only. The braille embeddings, and the cell encoder under T4 or E3, learn from a language-modeling loss on braille prompts followed by print continuations. A distillation loss pulls the hidden state at each braille word toward the frozen model's hidden state at the same word of the print sentence. UEB follows print word spacing, and all 500 pilot sentences have the same word count in both scripts, so words align one to one.
2. Stage 2 adds LoRA to the lowest layers if stage 1 plateaus, and reports print accuracy before and after.
3. Stage 3 adds braille tokens to the output head, with output vectors tied to the input embeddings through a learned linear map, and trains on print-to-braille and braille-to-braille data. A liblouis round trip checks that generated braille is valid.

Every result is reported next to these baselines.

- liblouis back-translation followed by the unmodified LLM
- The unmodified LLM reading Braille ASCII and Unicode, as in BrailleBench
- LoRA on the stock tokenizer with the same training data
- ByT5-small trained for back-translation only

Qwen3 1.7B and Llama 3.1 8B are convenient base models because BrailleBench already reports their GSM8K accuracy in print, Grade 1, and Grade 2 (69.4, 2.4, 1.4 and 83.9, 4.6, 1.4). Both leave almost the whole gap to close, and both train on one GPU.

## Data

Training data is synthetic and effectively unlimited, and the scarce resource is human-transcribed braille for testing.

- Parallel text comes from running liblouis over any print corpus, including instruction and reasoning datasets, with the UEB tables that BrailleBench uses (en-ueb-g1.ctb and en-ueb-g2.ctb). The position maps from the same call give the cell-to-character alignments that T3 and E2 need.
- A model trained and tested on liblouis output is scored on agreement with liblouis. Measuring real reading ability needs a held-out set of human-transcribed braille. Candidates are the expert-corrected sets released with [BrailleLLM](https://github.com/Tianyuan-Huang/BrailleLLM), the aligned passages from public braille books in Vision-Braille, the NIKL Korean print and braille corpus used by [BT5](https://www.alphaxiv.org/abs/2607.11893), and BANA's [running list of known translator errors](https://brailleauthority.org/sites/default/files/2026-05/Summary%20of%20Current%20Issues%20Spring%202026.pdf) as hard cases.
- Noise for the robustness tests is simulated by flipping, dropping, and inserting dots and cells, at rates measured from optical braille recognition output on the [Angelina](https://github.com/ilyaovodov/angelinadataset) and [DSBI](https://github.com/yeluo1994/DSBI) image sets.
- The toolkit released with BrailleBench builds braille versions of GSM8K, CommonsenseQA, and multi-hop QA for the transfer evaluation.

## Evaluation

The headline number is the share of the print-to-braille gap that a method closes on reasoning tasks, with the same frozen model answering the same questions in both scripts. It is the gain in braille accuracy over the stock model divided by the stock model's gap between print and braille accuracy.

| Task | Metric | Compared with |
| --- | --- | --- |
| Reasoning from braille input on GSM8K and CommonsenseQA, Grade 1 and 2 | accuracy, gap closed | BrailleBench numbers, pipeline baseline |
| Back-translation on synthetic and human-transcribed braille | exact match, character error rate | liblouis 3.32, ByT5 |
| Back-translation under simulated dot noise | character error rate at each noise rate | liblouis 3.32 |
| Locating an injected transcription error, and reflowing text to 40-cell lines | accuracy | pipeline baseline |
| Print retention | print accuracy before and after training | must be identical with a frozen body |
| Sequence length | tokens per print token | T0 to T4 |
| Representation convergence | layerwise similarity of braille and print hidden states, logit-lens recovery of print tokens | print path of the same model |

Ablations cover a grid of T1 to T3 against E1 to E3 on the smaller model, training set sizes of 10k, 100k, and 1M sentences, Grade 1 before Grade 2 against mixed training, separate against folded indicators, and stage 1 with and without the distillation loss.

## Milestones

The plan fits about ten weeks of part-time work, and the first week decides how much of the English track is worth pursuing.

1. In week 1, rerun the pilot on liblouis 3.32, then measure the pipeline baseline and the stock model on BrailleBench subsets and on a human-transcribed set. If the pipeline closes nearly the whole gap on clean English, narrow the English goals to noise robustness and analysis.
2. In weeks 2 and 3, build tokenizers T1 to T3 and report sequence length, vocabulary size, and coverage. Train a small back-translation model from scratch on each as a sanity check.
3. In weeks 4 to 6, train E1 to E3 on Qwen3 1.7B with the body frozen. The go or no-go criterion is closing more than half of the Grade 1 gap on GSM8K.
4. In weeks 7 and 8, add the cell encoder and the distillation loss, run the noise tests and the LoRA comparison, and do the layerwise analysis.
5. In weeks 9 and 10, add the braille output head, then repeat the best configuration on Chinese braille.
6. Finish with a workshop-length paper and a release of the tokenizer and data scripts.

## Results so far

The tokenizers, the data pipeline, and toy back-translation models are implemented in the `brltok` package and run on liblouis 3.32 with 36,841 training and 500 test sentences. Every model is a Transformer under 2M parameters trained from scratch on CPU for 4,000 steps, so these results cover the checks in milestones 1 and 2 and say nothing yet about a frozen pretrained model.

| Back-translator | Vocabulary | Tokens per word | CER, clean | Exact match, clean |
| --- | --- | --- | --- | --- |
| liblouis 3.32 | rules |  | 0.35% | 97.6% |
| T1 cell | 66 | 3.27 | 0.56% | 82.8% |
| T2 BPE 1k | 1,000 | 1.97 | 1.43% | 54.4% |
| T3 rule units | 1,546 | 2.98 | 2.58% | 48.0% |
| T2 BPE 8k | 8,000 | 1.33 | 11.13% | 0.4% |

- The rule-based pipeline is strong on clean English, which triggers the narrowing rule in milestone 1.
- Rule units barely shorten sequences, because most cells in contracted English are still single letters. BPE at 8k matches print length (1.03x the GPT-4 token count), so the length part of the first hypothesis holds for BPE and fails for rule units.
- From scratch at equal training examples, smaller units learn faster. Whether BPE or rule units overtake cells once their embeddings start from print is still untested.
- T3 segments braille without liblouis and reproduces the rule segmentation for 99.7% of test words.

The second table gives CER when one dot is flipped in a share of the test cells.

| Back-translator | Clean | 2% flips | 5% flips | 10% flips |
| --- | --- | --- | --- | --- |
| liblouis 3.32 | 0.35% | 2.71% | 6.10% | 12.05% |
| T1 table, clean training | 0.56% | 2.81% | 6.08% | 11.90% |
| T1 table, trained with 5% flips (2 seeds) | 1.07% | 2.62% | 4.93% | 9.01% |
| T1 dots, trained with 5% flips (2 seeds) | 1.06% | 2.51% | 4.71% | 8.59% |

- liblouis degrades linearly under dot noise, at about 1.2 CER points per point of flip rate, so rule-based decoding does not derail.
- A cell model trained with noise repairs some errors from context. At 10% flips its CER is 25% lower than liblouis with a table and 29% lower with dot-composed embeddings, at a cost of 0.7 points of clean CER. liblouis stays ahead below about 2% flips.
- Dot-composed embeddings behave as the second hypothesis predicted. They end level with the table on clean text and are about 5% better (relative) under noise in both seeds.

These numbers measure agreement with liblouis on 19th-century fiction, and most runs have one seed.

## Risks and open questions

- The pipeline baseline nearly solves clean English reading, with 97.6% exact Grade 2 round trips on liblouis 3.32. Work on clean English therefore has to justify itself through analysis, and the practical case rests on noisy input and on Chinese.
- Synthetic data makes evaluation circular, since a model trained on liblouis output learns liblouis's rules along with its errors. Human-transcribed test sets guard against this.
- BrailleLLM already initializes braille-fragment embeddings from print, so the novelty here lies in the controlled tokenizer comparison, the frozen body, transfer to reasoning tasks, dot-composed embeddings, and the representation analysis.
- Blind readers get braille from screen readers and displays driven by rule-based translators, so this project does not help them by default. Any accessibility claim needs braille readers and transcribers involved in choosing tasks and judging output.
- Fluent braille input opens an encoding channel around safety training, as [cipher-based jailbreaks](https://arxiv.org/html/2402.10601v5) show. Refusal behavior on braille prompts should be checked before any release, and BrailleBench lists safety evaluation as future work.
- It is open whether one tokenizer can serve several braille codes, since the same cell means different things in English, Chinese, and Nemeth math.
- The handling of 8-dot computer braille, and whether the braille output head shares parameters with the print head, are undecided.

## References

- Zhang et al., [BrailleBench: Investigating Multi-Criteria Braille Comprehension in Large Language Models](https://arxiv.org/pdf/2608.27268), 2026
- Huang et al., [BrailleLLM: Braille Instruction Tuning with Large Language Models for Braille Domain Tasks](https://aclanthology.org/2025.emnlp-main.1454), EMNLP 2025
- Wu et al., [Vision-Braille: A Curriculum Learning Toolkit and Braille-Chinese Corpus for Braille Translation](https://arxiv.org/pdf/2407.06048)
- Abdullah, [I'm Sorry, but I Can't Help with Braille](https://aclanthology.org/2026.ltedi-1.8/), LTEDI 2026
- Yu et al., [Pre-training model for low-resource Chinese-Braille translation](https://doi.org/10.1016/j.displa.2023.102506), Displays 2023
- Kitsunai et al., [Exploring the Potential of Prompt-Based Method for Kanji-Kana Conversion in Japanese Braille Translation](https://www.anlp.jp/proceedings/annual_meeting/2024/pdf_dir/P8-20.pdf), 2024
- [Building the First ML Model for Grade 2 Braille](https://prasanthmj.github.io/ai/grade-2-braille-transformer), blog post
- Hewitt, [Initializing New Word Embeddings for Pretrained Language Models](https://www.cs.columbia.edu/~johnhew//vocab-expansion.html), 2021
- Artetxe et al., [On the Cross-lingual Transferability of Monolingual Representations](https://arxiv.org/abs/1910.11856v2), ACL 2020
- [Embedding structure matters: Comparing methods to adapt multilingual vocabularies to new languages](https://arxiv.org/pdf/2309.04679), 2023
- Remy et al., [Trans-Tokenization and Cross-lingual Vocabulary Transfers](https://arxiv.org/abs/2408.04303), 2024
- [SambaLingo: Teaching Large Language Models New Languages](https://arxiv.org/pdf/2404.05829), 2024
- Minixhofer et al., [Zero-Shot Tokenizer Transfer](https://arxiv.org/abs/2405.07883v1), NeurIPS 2024
- Xue et al., [ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models](https://arxiv.org/abs/2105.13626), TACL 2022
- Pagnoni et al., [Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2412.09871v1), 2024
- Ma et al., [An Embarrassingly Simple Approach for LLM with Strong ASR Capacity](https://arxiv.org/abs/2402.08846v1), 2024
- Husain et al., [RomanSetu: Efficiently Unlocking Multilingual Capabilities of LLMs via Romanization](https://arxiv.org/html/2401.14280v1), 2024
- Purkayastha et al., [Romanization-based Large-scale Adaptation of Multilingual Language Models](https://arxiv.org/abs/2304.08865v1), Findings of EMNLP 2023
- Si et al., [Sub-Character Tokenization for Chinese Pretrained Language Models](https://arxiv.org/abs/2106.00400), TACL 2023
- Kaushal and Mahowald, [What do tokens know about their characters and how do they know it?](https://arxiv.org/abs/2206.02608v1), NAACL 2022
- Fang and Marks, [Unsupervised decoding of encoded reasoning using language model interpretability](https://arxiv.org/abs/2512.01222v1), 2025
- [All Code, No Thought: Current Language Models Struggle to Reason in Ciphered Language](https://arxiv.org/pdf/2510.09714), 2025
- [liblouis](https://github.com/liblouis/liblouis) braille translator, and ICEB, [Rules of Unified English Braille](https://iceb.org/Rules%20of%20Unified%20English%20Braille%202024.pdf), 2024

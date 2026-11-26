# Noisy-channel braille reading with a braille prior

Revision 3 of the reader. It reads braille into print strings, not into one model's tokens, so any
LLM can consume the result, and it adds an explicit prior over braille.

## Why

N-best rescoring (NOTES, revision 2) showed that the reader's candidates hold most of the right
answers (93 to 97% of words have the right spelling in the top 8 at 10% noise) and that a language
model can pick them out (word accuracy up 6 to 10 points). But the reader emits Qwen tokens, so it is
tied to one model, and the rescoring used the reader's score as a stand-in for a likelihood although
that score already folds in a prior over print words learned from the training mix.

## Model

Observed braille b, intended braille c, print y.

    print y --(translation)--> intended braille c --(errors)--> observed braille b
    P(y): any LLM         P(c | y): braille rules            P(b | c): braille-side errors
                          P(c): braille prior

Each candidate reading of a word is scored

    score(y) = log P(b | c) + log P(c | y) + beta log P(c) + alpha log P_LLM(y) [+ lam log P_reader(y | b) - gamma log P_words(y)]

with weights tuned on dev. The bracketed terms are the alternative likelihoods compared in E2.

1. **Error channel P(b | c)** (`brltok/channel.py`). A pair hidden Markov model over the cells of one
   word: each intended cell is deleted, emitted as one observed cell drawn from a substitution
   distribution, or emitted followed by an inserted cell (a repeat or a random cell). The probability
   of the observed word sums over all alignments by dynamic programming. The substitution
   distribution comes from the same events as `brltok.noise` (dot omission and addition by position,
   moves to a neighbouring dot, mirror, random cell). Profiles: `flip`, `writing` (assumed rates),
   `ocr` (event mix and per-dot omission and addition counts measured on Angelina,
   `exp/realimg/angelina_channel.json`), and `mix`, their average. The overall error rate is tuned
   on dev. The main results use `mix`, which does not know which profile corrupted the input;
   the profile-matched channel is reported as an upper bound.
2. **Translation P(c | y)** (`brltok/seq2seq.py`, task `translator`). A small character-to-cell
   transformer trained on print and braille word pairs from the training corpus, the word list and
   the non-word list. It returns its top 3 brailings per word, so non-standard brailing stays
   possible. No liblouis at inference.
3. **Braille prior P(c)** (`brltok/cell_lm.py`). A causal transformer language model over cells,
   trained on the liblouis braille of the training sentences. It scores the whole intended braille
   sentence with one word replaced, so it sees context on both sides like the print prior. Trained on
   liblouis output, it learns liblouis's brailing style.
4. **Print prior.** Any causal LM's log probability of the whole print text. Qwen3-1.7B and
   Llama 3.1 8B.
5. **Proposal reader** (`brltok/seq2seq.py`, task `reader`). The word-local reader of revision 2
   with a character decoder in place of the Qwen-token decoder. Because it is word-local, it is
   trained on single words (each sentence contributes all its words, so word frequencies follow the
   corpus), with the same non-word share and noise mix as revision 2. Beam search gives its top 16
   spellings per word. A word-prior model P_words(y) (task `wordprior`, the same decoder with no
   input) estimates the reader's built-in print prior, so it can be divided out (the hybrid
   speech-recognition trick).
6. **Joint decoding** (`brltok/nc_decode.py`). Start from each word's top reading, re-pick every
   uncertain word given the current picks of all other words, and repeat until no pick changes (at
   most 4 rounds). One round is per-word rescoring.

## Experiments, most important first

| # | Question | Measure |
|---|---|---|
| E1 | What does each prior add: none, braille prior, LLM, both? | dev word accuracy at 10% flip, writing, ocr |
| E2 | Which likelihood: reader score, reader score with word prior divided out, explicit chain, chain plus reader? | same |
| E3 | Joint decoding against per-word rescoring | dev word accuracy, then QA through the LLM |
| E4 | LLM-agnostic: same reader with Qwen3-1.7B and Llama 3.1 8B as prior and as QA model | QA accuracy against print and liblouis |
| E5 | Real photos | word accuracy by recognizer-quality band |
| E0 | Go/no-go: character reader against token reader | copy probe, word accuracy under noise, oracle@16 |

Dev protocol: the first 400 dev sentences whose braille and print word counts match. Weights are
tuned on sentences 0 to 199 and reported on 200 to 399. Noise seeds `nc-dev|{profile}|{i}`. Only
words whose top reader probability is below 0.99 are rescored, as in revision 2.

Grade 1 first; Grade 2, E4 and E5 if it holds.

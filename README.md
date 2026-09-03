# llm-scaling-from-scratch

A GPT-style language model written in raw PyTorch, trained at five sizes on arXiv abstracts, used to measure how validation loss falls as parameter count grows and to compare the fitted exponent against Kaplan et al. (2020).

## Overview

Large language models get better as they get bigger, and they do so in a surprisingly regular way. Kaplan et al. (2020) showed that a model's loss follows a power law in its parameter count: each time you multiply the number of parameters by ten, the loss drops by a roughly constant factor. This project rebuilds that experiment at a scale that fits on one GPU in about a day:

1. A decoder-only transformer implemented from scratch (embeddings, positional encoding, multi-head causal self-attention, layer normalisation, feedforward blocks), with no `nn.TransformerEncoder` and no Hugging Face model classes.
2. A byte-level BPE tokeniser trained on a corpus of arXiv abstracts from the cs.AI, cs.LG and cs.CL categories.
3. Five model sizes, roughly log-spaced from about 0.8M to 100M non-embedding parameters, each trained on exactly the same token budget.
4. A power-law fit `loss = a * N^-alpha` to the results, with a regression confidence interval and a bootstrap interval on `alpha`, compared against Kaplan et al.'s reported value of 0.076.
5. One ablation: learned positional embeddings against rotary positional encoding (RoPE) at a single mid-sized model, over three seeds.

### What this does and does not test

This experiment tests **Kaplan-style parameter scaling**: how loss changes with model size when the training data is held fixed. Every model sees the same training tokens once, and only the model size changes.

It does **not** test the **compute-optimal** question posed by Hoffmann et al. (2022), often called the Chinchilla result. That work asks, for a fixed compute budget, how to split it between a bigger model and more data, and finds that data and parameters should grow together at roughly 20 tokens per parameter. Answering it requires jointly varying model size and data volume across a grid of runs at a scale well beyond a single GPU-day. Hoffmann et al. is cited here as context: it is the reason a fixed-token comparison only answers part of the question, and the reason the largest models in this sweep are expected to be data-starved rather than a fair reflection of what they could achieve with more tokens. Nothing in this repository validates or contradicts the 20-tokens-per-parameter figure.

Repository layout:

```
src/scaling_lm/
  config.py        every constant in one place: paths, sizes, budgets, schedule
  model.py         GPT, CausalSelfAttention, LayerNorm, FeedForward, TransformerBlock
  positional.py    learned, sinusoidal and rotary positional encodings
  data.py          arXiv download, category filter, dedup, hash-based split
  tokenizer.py     BPE training and encoding to uint16 memmaps
  dataset.py       fixed-length token windows and deterministic epoch batching
  train.py         training loop, schedule, evaluation, checkpointing
  sweep.py         fixed-token model-size sweep
  ablation.py      learned vs RoPE at the medium size, three seeds
  generate.py      qualitative samples from each trained model
  scaling_fit.py   power-law fit with t-interval and bootstrap
  plots.py         log-log scaling plot, training curves, ablation plot
  report.py        writes results/summary.md and figures from the JSON artefacts
tests/             architecture, positional encoding, data pipeline and fit tests
```

## Method

### What a scaling law is

Suppose you train a series of language models that are identical except for size, and measure how well each one predicts held-out text. The natural measure is the cross-entropy loss: on average, how surprised the model is by the next token, in nats. Lower is better. A model that always assigned probability 1 to the correct next token would score 0; a model guessing uniformly over an 8,192-token vocabulary would score `ln(8192)`, about 9.0.

A scaling law is the observation that, when you plot loss against model size on log-log axes, the points fall close to a straight line. A straight line in log-log space is a power law:

```
L(N) = a * N^(-alpha)
```

where `N` is the number of parameters, `a` is a constant that depends on the data and tokeniser, and `alpha` is the exponent. The exponent is the interesting part because it is roughly independent of the details. It tells you how much you gain per order of magnitude: multiplying `N` by 10 multiplies the loss by `10^(-alpha)`. With Kaplan et al.'s `alpha_N = 0.076`, a tenfold larger model reduces loss by about 16%. Small exponents like this are why frontier models are so large: each halving of loss costs thousands of times more parameters.

Taking logarithms turns the power law into a line:

```
ln L = ln a - alpha * ln N
```

so `alpha` is minus the slope of a straight-line fit through the log-log points. That is how it is estimated here.

### Why bigger models have lower loss

The intuition runs in three steps.

1. **Language has structure at many scales.** Some of it is cheap to learn (common words follow common words). Some of it is expensive (which of several plausible technical terms belongs in this sentence, given the paper's topic three sentences ago). A model with more parameters can store more of these patterns at once.
2. **Loss is an average over many rare cases.** Most of the remaining loss in a trained model comes from the long tail of situations it has not learnt to handle. Each doubling of capacity lets the model pick off the next slice of that tail, but the slices get thinner, so the returns diminish. A power law with a small exponent is exactly the shape you get from steadily diminishing returns.
3. **Data caps the benefit.** A model cannot learn a pattern it never sees. With a fixed corpus, the biggest models run out of useful signal before they run out of capacity, which flattens the curve at the top end. Kaplan et al.'s `0.076` was measured with enough data that this did not bite. This sweep has a much smaller fixed budget, so a shallower slope at the largest sizes would be expected rather than surprising, and it is one of the ways the result is compared to theirs rather than treated as a replication.

### Why non-embedding parameters

Following Kaplan et al. (2020), model size is counted as the parameters in the attention and feedforward layers, excluding token and position embeddings. Embeddings scale with vocabulary size, not with the model's capacity to process context, and Kaplan et al. found that leaving them out gives a cleaner power law. Total counts are reported alongside for completeness. The input embedding and output projection share one weight matrix (Press and Wolf, 2017), so the vocabulary is only paid for once.

### Architecture

A pre-norm decoder-only transformer in the GPT-2 style (Radford et al., 2019; Xiong et al., 2020):

- Token embeddings tied to the output projection.
- Positional information from one of three interchangeable modules: learned position embeddings (the default), fixed sinusoidal encodings (Vaswani et al., 2017), or rotary encoding applied to queries and keys inside attention (Su et al., 2024).
- Multi-head causal self-attention written out explicitly: projection to queries, keys and values, scaled dot products, a lower-triangular boolean mask, softmax, weighted sum of values, output projection. Head dimension is fixed at 64.
- A hand-written `LayerNorm` (Ba, Kiros and Hinton, 2016).
- A feedforward block with a fourfold hidden expansion and GELU activation (Hendrycks and Gimpel, 2016).
- Weights initialised from `N(0, 0.02)`, with the two residual-branch output projections in each block scaled down by `1 / sqrt(2 * n_layer)` so that the residual stream does not grow with depth.

The five sizes keep head dimension and MLP ratio fixed and grow width and depth together, holding the width-to-depth ratio in a narrow band. Non-embedding counts below are the exact values reported by the code; Kaplan et al.'s `12 * n_layer * d_model^2` estimate is within 1% for every size.

| Name | Layers | `d_model` | Heads | Non-embedding params | Total params (8,192 vocab) |
|---|---|---|---|---|---|
| tiny | 4 | 128 | 2 | 0.79M | 1.87M |
| small | 6 | 256 | 4 | 4.74M | 6.90M |
| medium | 7 | 384 | 6 | 12.42M | 15.66M |
| large | 8 | 512 | 8 | 25.22M | 29.54M |
| xlarge | 14 | 768 | 12 | 99.23M | 105.71M |

Context length is 256 tokens throughout. Abstracts are short (a few hundred tokens), so a longer context would mostly attend across document boundaries.

### Data

- **Source.** The `librarian-bots/arxiv-metadata-snapshot` dataset on Hugging Face, pinned to a single revision so that the corpus can be rebuilt byte for byte. It is a Parquet mirror of the arXiv metadata snapshot maintained by Cornell (Clement et al., 2019). Only `id`, `title`, `categories` and `abstract` are read.
- **Filter.** A paper is kept if any of its categories is `cs.AI`, `cs.LG` or `cs.CL`. Each document is `title + newline + abstract`, whitespace-normalised, with abstracts under 200 characters dropped. Exact-duplicate texts are removed (arXiv versions of the same paper share an id, so they never appear twice; the check catches cross-listed re-uploads).
- **Split.** Documents are assigned to train (98%), validation (1%) and test (1%) by hashing the arXiv identifier. Each paper is in exactly one split, so there is no document-level leakage, and the assignment does not depend on the order the data was read. Within a split, documents are shuffled by a second hash so that the token stream is not in chronological order.
- **Tokeniser.** A byte-level BPE tokeniser (Sennrich, Haddow and Birch, 2016) with a vocabulary of 8,192 is trained on the training split only, then applied to all three. Each document ends with an `<|endoftext|>` token. Tokens are stored as flat `uint16` arrays and read through a memory map.

Actual token counts are written to `data/corpus_stats.json` when the corpus is built. Building from the pinned revision gives 457,738 unique documents in the three categories and the following splits:

| Split | Documents | Tokens |
|---|---|---|
| train | 448,573 | 118,703,695 |
| validation | 4,616 | 1,234,923 |
| test | 4,549 | 1,207,519 |

At 256 tokens per window and 128 windows per step, one pass over the training split is 463,686 windows and 3,622 optimiser steps, so every model trains on 118,685,696 tokens (the final 17,999 tokens of the split do not fill a batch and are dropped for every size).

### Training and the fixed token budget

Every model trains for exactly one pass over the training split, in the same deterministic order of 256-token windows, so the token budget is identical across sizes. The final partial batch is dropped so that each model sees exactly the same windows. Validation loss is measured on the whole validation split at the end of training; a cheap 20-batch validation estimate is logged every 200 steps to draw training curves. Test loss is computed once at the end and not used for any decision.

Optimiser settings follow common GPT-2 scale practice: AdamW (Loshchilov and Hutter, 2019) with `beta = (0.9, 0.95)`, weight decay 0.1 on matrices only, gradient clipping at 1.0, 128 sequences of 256 tokens per step (32,768 tokens), 5% linear warmup then cosine decay to 10% of the peak. The peak learning rate is set per size from the rule Kaplan et al. (2020, appendix D.6) fitted to their own sweep, `lr(N) = 0.003239 - 0.0001395 * ln(N)`, so that no per-size tuning is needed and no size is favoured by hand. Mixed precision (`bfloat16` autocast) is used on GPU.

Kaplan et al. trained with about 23 billion tokens and reported `alpha_N` for models trained to convergence. This sweep uses a fixed budget of 119 million tokens, which is all the training text the three arXiv categories provide. That works out at about 150 tokens per parameter for the smallest model and 1.2 tokens per parameter for the largest, so the two ends of the sweep sit on opposite sides of Hoffmann et al.'s (2022) roughly 20-tokens-per-parameter guideline. That is the main reason not to expect the exponents to agree exactly, and it is discussed under Limitations.

Compute is modest: with `6 * N * D` FLOPs per training run the largest model needs about 7.5e16 FLOPs, and the whole five-model sweep plus the ablation runs is under 2e17 FLOPs, comfortably under a day on one A40 even at low utilisation.

### Fitting the power law

`scaling_fit.py` fits a straight line to `ln(loss)` against `ln(non-embedding parameters)` by ordinary least squares and reports:

- `alpha`, minus the slope, with its standard error and a 95% t-interval from the regression;
- a 95% percentile bootstrap interval on `alpha` from 10,000 resamples of the five points, as a check that does not assume Gaussian residuals;
- `R^2` and whether Kaplan et al.'s 0.076 falls inside the regression interval.

With five points and two fitted parameters there are three residual degrees of freedom, so the regression interval is wide by construction. Reporting it is the point: a single exponent from five runs is not a precise number and should not be read as one.

### Ablation

One controlled comparison at the `medium` size (12.4M non-embedding parameters): learned position embeddings against rotary positional encoding, everything else identical, three seeds each. The seed changes both the initialisation and the order in which training windows are visited. Because the two schemes are run with the same three seeds, the design is paired: the learned and RoPE runs with seed 0 saw the training windows in the same order, and likewise for seeds 1 and 2. The report therefore gives per-seed losses, the mean and standard deviation per scheme, and then works on the per-seed differences: their mean, their standard deviation, and a paired t-test (`scipy.stats.ttest_rel`). A paired test is the right one here because it removes the part of the run-to-run variance that the seed explains; treating the six runs as two independent samples would ignore that matching. Three seeds cannot detect a small effect, so the honest reading of the result is "the difference is or is not larger than run-to-run noise", not a precise effect size.

### Reproduction

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && pip install -e .

python -m scaling_lm.data          # download, filter, split (about 3.4 GB of Parquet, 30 min)
python -m scaling_lm.tokenizer     # train BPE, encode splits, write corpus_stats.json (1 min)
python -m scaling_lm.sweep         # five sizes, one epoch each, results/runs/*/result.json
python -m scaling_lm.ablation      # learned vs rope, medium, seeds 0 1 2
python -m scaling_lm.generate      # samples from every sweep checkpoint
python -m scaling_lm.report        # scaling_fit.json, figures, results/summary.md
pytest
```

`train.py` can also run a single configuration (`python -m scaling_lm.train --model-size small --positional rope --seed 1`). Every script skips runs whose `result.json` already exists, so the sweep can be resumed after an interruption. A skipped run is only accepted if it is exactly the run the current code and data describe: each result stores the full run specification (corpus fingerprint, resolved architecture, training schedule and constants) and a SHA-256 hash over the whole of it. `runs.py` is the single place this check lives, and the sweep, single-model, generation and report scripts all load results through it, so any mismatch is an error rather than a silent reuse. `--max-steps` and `--batch-size` exist for smoke tests only and are never used for reported results.

## Results

Pending the GPU sweep. `python -m scaling_lm.report` writes `results/summary.md` and the figures below once `results/scaling_sweep.json` exists; this section will be filled in from that output and will link to the committed artefacts.

- `results/figures/scaling_law.png`: log-log validation loss against non-embedding parameters with the fitted line, its interval, and a reference line at Kaplan et al.'s slope.
- `results/figures/training_curves.png`: validation loss against tokens seen, one curve per size.
- `results/figures/positional_ablation.png`: per-seed final losses for learned and rotary positions.
- `results/generations.json`: three fixed prompts continued by each model. These are qualitative colour only. Models of this size trained on this little text produce fluent-looking but unreliable prose, and nothing about the scaling result rests on them.

## Limitations

- **Fixed data, not compute-optimal.** Every model sees the same tokens. By Hoffmann et al.'s (2022) estimate the largest model here would want on the order of two billion tokens to be trained compute-optimally; it gets a small fraction of that. The fitted exponent describes loss against size *at this data budget*, and should be expected to come out shallower than one measured with abundant data.
- **Five points.** A fit through five sizes spanning two orders of magnitude gives a real but wide confidence interval. Agreement or disagreement with 0.076 is a comparison, not a hypothesis test with much power.
- **Different setup from Kaplan et al.** Different corpus (arXiv abstracts against WebText2), different tokeniser and vocabulary (8,192 against 50,257, which changes the absolute loss), shorter context (256 against 1,024), and a learning-rate rule borrowed from their sweep rather than tuned here. The slope is more comparable across setups than the intercept, but none of these differences is controlled for.
- **One seed for the sweep.** The five sweep runs use seed 0. The ablation gives an estimate of seed-to-seed noise at the medium size, which is used to judge whether the sweep's residuals are plausible, but the sweep itself was not repeated.
- **Small domain.** The cs.AI, cs.LG and cs.CL abstracts are narrow, formulaic text. Loss values and the difficulty of the long tail will not transfer to general web text.
- **Deduplication is exact-match only.** Near-duplicate abstracts (revised versions posted under new identifiers, or heavy self-overlap between papers from one group) are not removed and could leak lightly between splits.

## Future Work

- Extend this to a proper compute-optimal sweep: train each size at several token budgets, fit `L(N, D)` jointly and read off the optimal tokens-per-parameter ratio. That is the experiment Hoffmann et al. (2022) ran and the one this project deliberately scoped out; it needs a larger corpus and a grid of runs rather than a single line of them.
- Repeat the sweep with two more seeds so that the scaling fit has its own error bars per point rather than borrowing them from the ablation.

## References

- Ba, J.L., Kiros, J.R. and Hinton, G.E. (2016) 'Layer normalization', *arXiv preprint* arXiv:1607.06450. doi: 10.48550/arXiv.1607.06450. Note: preprint, not peer reviewed, but the standard reference for the technique.
- Clement, C.B., Bierbaum, M., O'Keeffe, K.P. and Alemi, A.A. (2019) 'On the use of arXiv as a dataset', *arXiv preprint* arXiv:1905.00075. doi: 10.48550/arXiv.1905.00075. Note: preprint describing the Cornell arXiv metadata snapshot that the Hugging Face mirror used here is built from.
- Hendrycks, D. and Gimpel, K. (2016) 'Gaussian error linear units (GELUs)', *arXiv preprint* arXiv:1606.08415. doi: 10.48550/arXiv.1606.08415. Note: preprint, not peer reviewed.
- Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L.A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., Rae, J.W., Vinyals, O. and Sifre, L. (2022) 'Training compute-optimal large language models', *Advances in Neural Information Processing Systems*, 35. Preprint: arXiv:2203.15556, doi: 10.48550/arXiv.2203.15556. Note: NeurIPS proceedings do not assign DOIs; the arXiv DOI is given instead.
- Kaplan, J., McCandlish, S., Henighan, T., Brown, T.B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J. and Amodei, D. (2020) 'Scaling laws for neural language models', *arXiv preprint* arXiv:2001.08361. doi: 10.48550/arXiv.2001.08361. Note: never published in a peer-reviewed venue, but one of the most cited papers in the field and the source of the `alpha_N = 0.076` figure used here.
- Loshchilov, I. and Hutter, F. (2019) 'Decoupled weight decay regularization', *International Conference on Learning Representations*. Available at: https://openreview.net/forum?id=Bkg6RiCqY7. Note: ICLR does not assign DOIs.
- Press, O. and Wolf, L. (2017) 'Using the output embedding to improve language models', *Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics*, Volume 2, pp. 157-163. doi: 10.18653/v1/E17-2025.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D. and Sutskever, I. (2019) 'Language models are unsupervised multitask learners', OpenAI technical report. Available at: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf. Note: not peer reviewed and no DOI; it is the primary description of the GPT-2 architecture followed here.
- Sennrich, R., Haddow, B. and Birch, A. (2016) 'Neural machine translation of rare words with subword units', *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*, Volume 1, pp. 1715-1725. doi: 10.18653/v1/P16-1162.
- Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W. and Liu, Y. (2024) 'RoFormer: Enhanced transformer with rotary position embedding', *Neurocomputing*, 568, 127063. doi: 10.1016/j.neucom.2023.127063.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L. and Polosukhin, I. (2017) 'Attention is all you need', *Advances in Neural Information Processing Systems*, 30. Preprint: arXiv:1706.03762, doi: 10.48550/arXiv.1706.03762. Note: NeurIPS proceedings do not assign DOIs.
- Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L. and Liu, T.-Y. (2020) 'On layer normalization in the transformer architecture', *Proceedings of the 37th International Conference on Machine Learning*, PMLR 119, pp. 10524-10533. Available at: https://proceedings.mlr.press/v119/xiong20b.html. Note: PMLR does not assign DOIs.

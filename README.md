# How to run this project

```bash
conda env create -f ntc_finrep_environment.yml
conda activate ntc-finrep
python main.py
```

Full instructions, including the notebook option and a quick smoke test, are under "How to run it".

---

# FinRep: Can Transaction History Help Predict Credit Risk?

A research prototype asking one question: **do patterns hidden in a person's transaction history tell us
more about their credit risk than simple monthly totals do?**

Everything here runs on **simulated data** — no real customers, no real bank, no real lending
decisions. It is a controlled experiment built to test a method.

---

## Start here if you have no banking or machine-learning background

- **Credit risk and default** — how likely a borrower is to repay. A *default* is failing to repay as
  agreed; the window here is 12 months, so the target is "did this person default within a year?"
- **New-to-Credit (NTC)** — someone with little or no borrowing history, so bureaus hold nothing
  useful and conventional scoring is weak. Banks still want to serve them, hence other signals.
- **The idea under test** — a bank already sees transactions (salary in, rent out, groceries,
  transfers) and, crucially, *when* each happens. That timing may reveal stress before any missed
  repayment. Can a model learn it from the raw sequence?
- **Feature** — a hand-built number, such as average monthly income.
- **Embedding** — a short learned list of numbers (here 32) the network produces to describe a person,
  rather than a summary a human chose.
- **Self-supervised learning** — training on a puzzle the data generates itself: hide some
  transactions and guess them back. No default label is needed.

---

## The experiment in five sentences

1. A simulator creates 48,000 fake applicants and ~4.7 million fake transactions, with a hidden
   "stress" signal that changes *when* transactions happen, not their monthly totals.
2. Inputs are 29 hand-made monthly cash-flow features and 32 learned embeddings from a Transformer
   trained on transaction sequences.
3. Six models are fitted: three on cash-flow only, three on cash-flow plus embeddings.
4. All are scored on the same untouched future cohort, with confidence intervals, five seeds, a
   random-embedding control and a tuned baseline.
5. Nobody can cheat: no model sees the hidden signal, the simulated answer, or post-application data.

## What the experiment found

On the untouched test cohort (9,318 applicants, 11.46% default rate):

| Model | ROC-AUC | PR-AUC | KS |
|---|---|---|---|
| Cash-flow features only | 0.6905 | 0.2143 | 0.2844 |
| Cash-flow + 32 random numbers (control) | 0.6887 | 0.2094 | 0.2798 |
| Cash-flow + learned FinRep embeddings | **0.6975** | **0.2169** | **0.2909** |
| Cash-flow, tuned harder | 0.6948 | **0.2234** | 0.2951 |

The gain is real but narrow: embeddings add **+0.0072** ROC-AUC over cash-flow only, 95% interval
**[0.0009, 0.0134]** excluding zero, and the hybrid leads on all five seeds. Little else survives:
PR-AUC +0.0030 [−0.0083, 0.0156], and the KS and Brier differences include zero. The tuned cash-flow
benchmark matches or beats the hybrid on PR-AUC.

Random numbers did not help — the control scores below cash-flow only on all three metrics, so the
gain comes from learned structure rather than extra inputs, and the hybrid beats the control on
ROC-AUC (+0.0089, [0.0022, 0.0162]). Label efficiency is mixed: the hybrid leads at 10% and 100% of
labels but trails at 5%, 25% and 50%. The simulator's ceiling is 0.7343 ROC-AUC, so part of the signal
is recovered, not all.

### What that means for a lending decision

A ranking metric is not a business result. Fixing approval at 70% and letting each model set its own
cutoff, the hybrid approves a book defaulting at 7.36% against 7.53% — about **17 fewer defaults per
10,000 approved loans**, or roughly $71,000 of avoided loss per 10,000 applicants at $10,000 exposure
and 60% loss given default. But the interval [−10.3, +47.5] includes zero, as it does at every
approval rate from 50% to 90%: the ranking gain is not yet a demonstrated decision gain.

> Results describe a simulation, not real-world credit performance. Scores may move by about ±0.001
> between runs because models are retrained each time.

---

## Repository map

```
.
├── finrep-ntc-credit-risk.ipynb   ← the main document: whole study end to end, with results
├── main.py                        ← command-line entry point
├── config.yaml                    ← all settings: seed, sizes, model hyper-parameters
├── requirements.txt               ← Python packages needed
├── requirements-lock.txt          ← pinned versions for reproducible installs
├── ntc_finrep_environment.yml     ← Conda environment (Python 3.11)
├── README.md                      ← this file
├── src/                           ← all source code (explained below)
├── tests/                         ← automated tests (41 checks)
├── data/generated/                ← created when you run; ignored by Git
└── artifacts/                     ← results written when you run; ignored by Git
```

`data/generated/` (about 120 MB) and `artifacts/` are rebuilt on every run and are listed in
`.gitignore`. This working copy already contains one completed 48,000-applicant run; delete both to
start from scratch.

---

## How to run it

**Option A — the notebook (recommended).** Open `finrep-ntc-credit-risk.ipynb` and run all cells top
to bottom. It is self-contained and takes roughly 15–45 minutes; try the smoke test first to gauge
your machine.

**Option B — the command line.**

```bash
conda env create -f ntc_finrep_environment.yml
conda activate ntc-finrep
python main.py
```

Quick smoke test (small, fast):

```bash
python main.py --n-applicants 300 --epochs 1
```

**Tests:**

```bash
python -m pytest -q
```

**Requirements:** Python 3.11, 16 GB RAM recommended. GPU is optional (CUDA or Apple Silicon);
everything also runs on CPU.

---

## Code guide: what every Python file does

Each section names a file, says what it is for, and lists the functions inside it.

### `main.py` — the front door (11 lines)

Reads three optional flags (`--config`, `--n-applicants`, `--epochs`) and hands them to
`src.experiment.main()`. Nothing else happens here.

---

### `src/experiment.py` — the conductor (203 lines)

Runs the whole study in order; if you read one file, read this one. `main(...)` executes the pipeline:
load settings → build the dataset → pretrain and embed → fit and compare models → tune the benchmark,
calibrate, measure label efficiency and score stability → write the split audit and run manifest.
`_write_json`, `_git_hash` and `_package_versions` record what was run.

---

### `src/data/dataset.py` — the modelling dataset (142 lines)

Turns the simulated book into the four cohorts and writes every input table.

- `temporal_split(...)` — sorts by application date and cuts into train / validation / test, keeping
  whole dates together so no month appears on both sides.
- `temporal_split_with_calibration(...)` — four groups: train, validation (tuning), calibration
  (probability scaling), test.
- `build_dataset(...)` — one-call entry point: generate → build features → keep the simulator's
  probability as the audit ceiling → assemble IDs, date and target → split → write six parquet files.
- `TARGET` — the single definition of `default_12m`, so no file can disagree.

Splitting by date stops the model predicting the past from the future, or being tested on data it was
tuned on.

---

### `src/representation_learning.py` — the self-supervised stage (65 lines)

- `pretrain_encoder(...)` — builds sequences for training-vintage applicants only, trains on the
  masked-transaction puzzle, saves the model and history, and returns an audit record.
- `embed_applicants(...)` — runs the frozen encoder over everyone, giving 32 numbers per person
  (`finrep_000` … `finrep_031`).

The audit record lets the pipeline assert that no validation, calibration or test applicant was seen
during pretraining.

---

### `src/risk_modelling.py` — the supervised stage (147 lines)

- `fit_feature_sets(...)` — fits three model families on every feature set from one shared sample;
  returns a combined metrics table.
- `tune_feature_sets(...)` — a small LightGBM grid selected on validation ROC-AUC only; test is
  scored once, after selection.
- `oracle_ceiling(...)` — scores the simulator's noise-free probability, the ceiling a real model
  could reach.
- `calibrate_hybrid(...)` — Platt scaling fitted on the calibration vintage, reported on test.
- `label_efficiency(...)` — refits both feature sets at 5%, 10%, 25%, 50% and 100% of labels.
- `score_stability(...)` — PSI between validation and test scores as a drift check.

---

### `src/data/generate_data.py` — the simulator (387 lines)

Creates the fake world: applicants, their transactions, and whether they defaulted.

**Fixed lists:** `CATEGORIES` (ten transaction types), `PROFILES` (five archetypes: stable salaried,
volatile salaried, self-employed, high obligation, low liquidity), `ACCOUNT_TYPES`,
`HOUSING_STATUSES`, and `GENERATOR_ONLY_COLUMNS` — the authoritative list of simulation-only columns
that must never reach a model.

**Functions:** `sigmoid(x)` squashes values into 0–1; `generate_data(...)` picks each applicant's
profile, income, obligation level, opening balance, housing, account type and four independent stress
traits, writes 12 months of prior transactions, computes the true default probability and draws the
outcome, returning one table of applicants and one of transactions; `main()` is a standalone CLI
wrapper.

**Two ideas inside `generate_data`:** *cash-flow drivers* (obligation, income volatility, opening
liquidity) appear in monthly totals, so hand-built features can see them. *Sequence-stress drivers*
(payment lateness, income timing, spending tempo) change **when** money moves, not how much —
invisible to monthly totals, visible to a sequence. That is the signal the embeddings should recover.

Two probabilities are stored for auditing only: `p_default_conditional` (the noise-free ceiling) and
`p_default_realized` (used to draw the label). Neither is ever a model input.

---

### `src/data/sequences.py` — turning rows into a sequence (155 lines)

Neural networks need a fixed-shape numeric sequence, not a spreadsheet row.

- `CATEGORIES`, `CAT_TO_ID` — a fixed category-to-number map; `MASK_ID`, `PAD_ID`, `N_CATEGORIES` —
  special tokens; `N_TIME_BUCKETS` — 16 slots per month so *when* is representable.
- `encode_sequences(...)` — drops anything on or after the application date, keeps the most recent
  transactions, and encodes category, signed amount, amount relative to that person's typical
  transaction, time slot, direction and padding flag. Returns per-applicant NumPy arrays, which
  matter at scale: lists would cost gigabytes.
- `to_tensors(...)` — stacks them into six tensors in a fixed applicant order.

---

### `src/features/cashflow_features.py` — the hand-built baseline (282 lines)

The traditional approach: 29 interpretable numbers per person.

- `SPEND_CATEGORIES`, `INCOME_CATEGORIES` — which transaction types feed each spending bucket and
  income stream; transfers are excluded because moving your own money is neither spending nor earning.
- `build_cashflow_features(...)` — the 29 features: average credits and debits, balance levels,
  counts, seven spending shares, income volatility, expense-to-income ratio, transfer ratios,
  liquidity months, balance volatility, income frequency, account age and type shares, housing
  payment size.
- `filter_extreme_applicants(...)` — legacy, no longer used to drop rows.
- `winsorize_extreme_features(...)` — the replacement: caps extreme values using training-only
  bounds, so every applicant stays.

Deleting "weird" applicants using training statistics would quietly change the test population — a
form of cheating. Capping keeps everyone.

---

### `src/models/transaction_encoder.py` — the neural network (77 lines)

- `TransactionEncoder` — a small Transformer (4 attention heads, 2 layers) that reads a transaction
  sequence and emits one 32-number summary per person.
- `TransactionEncoder.forward(...)` — encodes category, amount, timing and direction, lets
  transactions attend to each other, then averages into a single vector; also returns
  per-transaction outputs for training.
- `masked_category_loss(...)` — hides about 15% of categories and scores how well the model guesses
  them back. Guessing well requires understanding typical patterns, so the embedding becomes
  informative without the default label.

---

### `src/models/pretrain.py` — training and using the encoder (137 lines)

- `SequenceDataset` — batches the tensors for PyTorch.
- `select_device()` — picks CUDA, Apple Silicon or CPU.
- `pretrain(...)` — trains on the masked puzzle with a held-out slice, early stopping and a saved
  checkpoint; returns the training history.
- `extract_embeddings(...)` — runs a trained encoder without masking and returns one 32-number vector
  per applicant.

Only training-period applicants are used for pretraining; validation and test are encoded later with
a frozen model.

---

### `src/models/factory.py` — model builder (14 lines)

`make_lgbm(params, seed)` builds a LightGBM classifier inside a pipeline that fills missing values
first.

---

### `src/models/risk_model.py` — the three comparison models (46 lines)

`fit_and_evaluate(...)` trains logistic regression, histogram gradient boosting and LightGBM on the
training cohort, scores validation and test, saves each model and returns a results table. All scaling
and imputation is fitted on training rows only. The same function runs with cash-flow features and
with cash-flow plus embeddings, so only the inputs differ.

---

### `src/evaluation/metrics.py` — how performance is measured (68 lines)

- `ks_statistic(...)` — gap between defaulter and non-defaulter score distributions; bigger is better.
- `expected_calibration_error(...)` — do predicted probabilities mean what they say?
- `evaluate(...)` — bundles ROC-AUC, PR-AUC, KS, Brier score, cohort size and default rate.
- `psi(...)` — Population Stability Index between two score distributions; below about 0.1 means
  consistent behaviour.

ROC-AUC 0.5 is random guessing and 1.0 perfect ordering; PR-AUC depends on how common defaults are; a
lower Brier score is better.

---

### `src/evaluation/embedding_diagnostics.py` — sanity check on the embeddings (66 lines)

`embedding_activity_diagnostics(...)` correlates each of the 32 dimensions with real activity
measures (transaction count, months observed, income, spending ratio) to check the embeddings are
alive and track behaviour rather than noise.

---

### `src/evaluation/approval_policy.py` — from ranking to decisions (228 lines)

ROC-AUC sorts the whole population; a lender only cares near its cutoff. This module turns scores into
approve/reject decisions and money.

- `cutoff_at_approval_rate(...)` — the score quantile approving the lowest-risk share.
- `approval_policy_table(...)` — per model and approval rate: approved count, bad rate, defaults
  captured, expected loss at the stated EAD and LGD.
- `approval_policy_deltas(...)` — challenger minus baseline: bad-rate change, defaults avoided per
  10,000 approved, loss avoided.
- `bootstrap_policy_delta(...)` — a paired bootstrap interval on those figures.
- `swap_analysis(...)` — who is approved by one model but rejected by the other, and how risky each
  group is.
- `fixed_cutoff_comparison(...)` — one shared threshold, so the approval rate floats instead.
- `capture_curve(...)` — cumulative share of defaults captured as more applicants are rejected.

Holding approval fixed keeps loan volume constant, so a difference is a pure mix effect readable as
"defaults avoided"; the fixed-cutoff view is reported alongside because it is a different policy.

---

### `src/decile.py` — risk ranking tables (48 lines)

- `decile_analysis(...)` — sorts by predicted risk into ten groups, reporting observed default rate,
  default count and lift.
- `pentile_analysis(...)` — the same with five groups; used to answer "if we flag the riskiest 20% or
  40%, how many defaults do we catch?"
- `_capture_table(...)` — the shared helper; it ranks before binning so tied scores cannot collapse a
  group.

These tables turn abstract scores into something a credit team can act on.

---

### `src/__init__.py` and the `__init__.py` files in sub-folders

Empty markers declaring `src`, `src.data`, `src.models` and so on as importable packages.

---

### `tests/` — the automated checks (41 tests)

- `test_generation.py` — the simulator behaves as documented: stress traits show only in timing,
  generator-only columns exist, conditional and realised probabilities differ, prevalence intercept
  sets the default rate.
- `test_features.py` — features are leakage-safe, pre-application only, and winsorisation preserves
  every row.
- `test_sequences.py` — window, ordering and padding are correct.
- `test_models.py` — encoder shapes, padding invariance, saved-model reproducibility, seeds.
- `test_metrics.py` — metrics match known cases.
- `test_decile.py` — bucketing, lift and capture add up.
- `test_temporal_split.py` — splits never mix dates across cohorts.
- `test_approval_policy.py` — cutoffs, deltas, bootstrap intervals, swaps, fixed-cutoff comparisons
  and capture curves.
- `test_experiment.py` — a small end-to-end smoke run writes its manifest and audit files.

Run `python -m pytest -q`; they should all pass before you trust any result.

---

## Glossary

| Term | Plain meaning |
|---|---|
| **Default** | Failing to repay on the agreed terms within the observation window. |
| **Feature** | A hand-built number describing a person, e.g. average monthly income. |
| **Embedding** | A short learned list of numbers representing a person, written by the network, not a human. |
| **Transformer** | Architecture that weighs how each item in a sequence relates to the others; the basis of modern language models. |
| **Self-supervised** | Training on a puzzle generated from the data itself, without needing the label. |
| **Latent** | A hidden simulator variable; never a model input. |
| **Leakage** | Letting a model see information it would not have at decision time; flatters results. |
| **Chronological split** | Split by date: trained on the past, tested on the future. |
| **Calibration** | Adjusting probabilities so "30% risk" really means about 30 in 100. |
| **ROC-AUC** | Chance a random defaulter ranks riskier than a random non-defaulter. 0.5 = coin flip. |
| **PR-AUC** | Ranking quality focused on defaulters; useful when defaults are rare. |
| **KS** | Largest gap between defaulter and non-defaulter score distributions. |
| **Brier score** | Average squared error of predicted probabilities; lower is better. |
| **PSI** | Score-drift measure; near zero means two cohorts score similarly. |
| **Bootstrap interval** | A range from repeatedly resampling the test cohort, showing how uncertain a difference is. |
| **Oracle** | The simulator's true probability, known only to the experimenter; the ceiling. |

---

## What is deliberately *not* done

- No post-application data is generated or used.
- Generator-only columns are stripped and asserted absent by tests.
- The encoder is pretrained on training-period applicants only.
- Tuning uses validation; calibration its own cohort; the test cohort is touched once.
- Extreme values are capped, not deleted, so the test cohort is never filtered with training
  statistics.
- The random-embedding control has the same number of inputs as the real model, so a gain cannot be
  blamed on extra dimensions.

## Honest limitations

- The simulator is a model of reality, not reality; its "stress" signal was designed to be learnable.
- Results depend on one synthetic dataset, one encoder and one tuned benchmark.
- Confidence intervals capture sampling noise on the test cohort, not uncertainty about the simulator
  itself.
- A small gain says the method *can* recover timing information when it exists — not that real
  transaction data contains it, or that a bank should lend on it.

## Licence

See `LICENSE`.

---

## References and Acknowledgements

### Research and technical references

1. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS. https://arxiv.org/abs/1706.03762
2. Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
   https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
3. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
   https://doi.org/10.1145/2939672.2939785
4. Saito, T., & Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC
   Plot When Evaluating Binary Classifiers on Imbalanced Datasets*. PLOS ONE, 10(3).
   https://doi.org/10.1371/journal.pone.0118432
5. Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*.
   NeurIPS. https://arxiv.org/abs/1705.07874

### AI assistance and acknowledgements

Two AI assistants were used at different stages. In both cases the output was reviewed, adapted and
validated by the author, and the synthetic data generation, implementation, execution and
interpretation of results remain the author's responsibility.

#### Claude (Anthropic), free version

Used as an AI-assisted development and research support tool throughout: project architecture and
modelling methodology, Python implementation and debugging, code review, exploratory data analysis and
visualisation, experimental design and interpretation, and technical writing.

#### DeepSeek V4.1 Flash (DeepSeek), via the OpenCode coding agent

A verification-focused review pass over the notebook and its dependencies: auditing the notebook and
`src/` for target leakage, generator-only inputs and stale feature names; confirming reported metrics
match the regenerated `artifacts/`; reviewing the generator and cash-flow features for consistency
with the documented methodology; checking imports, config and cross-references; running the test
suite; and cleaning up the reproducibility log.

#### AI tool references

- Anthropic. (2026). *Claude* [Large language model], free version. https://claude.ai/
- DeepSeek. (2026). *DeepSeek V4.1 Flash* [Large language model], accessed via OpenCode.
  https://www.deepseek.com/

### Data disclaimer

All applicant and transaction records are synthetically generated for research and demonstration; they
do not represent actual customers, institutions or credit outcomes. Performance and conclusions are
conditional on the simulator's assumptions and should not be generalised without independent
validation.

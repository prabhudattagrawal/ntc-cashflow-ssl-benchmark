# How to run this project

```bash
conda env create -f ntc_finrep_environment.yml
conda activate ntc-finrep
python main.py
```

Full instructions, including the notebook option and a quick smoke test, are in
"How to run it" below.

---

# FinRep: Can Transaction History Help Predict Credit Risk?

A research prototype that asks one question: **do patterns hidden in a person's transaction history
tell us more about their credit risk than simple monthly totals do?**

Everything in this repository runs on **simulated data**. No real customers, no real bank, no real
lending decisions. It is a controlled experiment, built to test a method.

---

## Start here if you have no banking or machine-learning background

**What is "credit risk"?** When a bank lends money, it wants to know how likely a person is to pay
it back. "Credit risk" is simply that likelihood.

**What is a "default"?** A default is when a borrower fails to repay as agreed within a set period.
Here the period is 12 months, so the outcome we predict is "did this person default within a year?"

**What is "New-to-Credit" (NTC)?** Someone with little or no borrowing history. Credit reference
agencies ("bureaus") have nothing useful on them, so the usual scoring methods are weak. Banks still
want to serve these people, so they look for other signals.

**What is the idea being tested?** A bank already sees a customer's transactions: salary in, rent
out, groceries, transfers, and *when* each one happens. Some of that information may reveal financial
stress before any missed repayment shows up. This project tests whether a computer can learn those
patterns automatically from the raw sequence of transactions.

**What is a "feature"?** A number a model can use, such as "average monthly income" or "how many
transactions per month". Features are hand-built summaries.

**What is an "embedding"?** A short list of numbers (here 32 of them) that a neural network learns to
represent a person. Instead of writing "income volatility" by hand, the network produces its own
compact description of that person's behaviour. Think of it as a machine-written summary that no
human had to define.

**What is "self-supervised" learning?** Training a model on a puzzle it can generate itself, without
needing the answer (default) at that stage. Here the puzzle is: hide some transactions and ask the
model to guess what they were. To do that well, it must learn how people's transactions fit together.

---

## The experiment in five sentences

1. A simulator creates 48,000 fake applicants and about 4.7 million fake transactions, with a known
   hidden "stress" signal that affects *when* transactions happen but not their monthly totals.
2. Two kinds of inputs are built: 29 hand-made monthly cash-flow features, and 32 learned embeddings
   produced by a Transformer neural network trained on the transaction sequences.
3. Six models are fitted: three on cash-flow features only, three on cash-flow plus embeddings.
4. All models are scored on the same untouched future cohort, and the difference between them is
   measured with confidence intervals, five random seeds, a random-embedding control and a tuned
   baseline.
5. Everything is designed so that nobody can cheat: the model never sees the hidden signal, the
   simulated answer, or any data from after the application date.

## What the experiment found

On the untouched test cohort (9,318 applicants, 11.46% default rate):

| Model | ROC-AUC | PR-AUC | KS |
|---|---|---|---|
| Cash-flow features only | 0.6905 | 0.2143 | 0.2844 |
| Cash-flow + 32 random numbers (control) | 0.6887 | 0.2094 | 0.2798 |
| Cash-flow + learned FinRep embeddings | **0.6975** | **0.2169** | **0.2909** |
| Cash-flow, tuned harder | 0.6948 | **0.2234** | 0.2951 |

The gain is real but narrow. Learned FinRep embeddings raise ROC-AUC by **+0.0072** over
cash-flow-only, with a 95% interval of **[0.0009, 0.0134]** that excludes zero, and the hybrid leads
in all five random seeds. Almost nothing else survives the confidence intervals: the PR-AUC
difference is +0.0030 with an interval of [−0.0083, 0.0156], and the KS and Brier differences also
include zero. The tuned cash-flow benchmark matches or beats the hybrid on PR-AUC.

Adding *random* numbers instead of learned embeddings did not help. The control scores below
cash-flow-only on all three metrics, which is the point: the gain comes from learned structure, not
from simply giving the model more inputs. The hybrid also beats that control on ROC-AUC with an
interval that excludes zero (+0.0089, [0.0022, 0.0162]).

Label efficiency is mixed rather than uniform: the hybrid leads at 10% and 100% of the training
labels, but trails at 5%, 25% and 50%.

The simulator's own ceiling (the true probability it used to generate defaults) is 0.7343 ROC-AUC,
so the models recover part of the signal but not all of it.

### What that means for a lending decision

A ranking metric is not a business result. Holding the approval rate fixed at 70% and letting each
model set its own cutoff, the hybrid approves a book with a 7.36% default rate against 7.53% for
cash-flow-only — about **17 fewer defaults per 10,000 approved loans**, or roughly $71,000 of avoided
loss per 10,000 applicants at $10,000 exposure and 60% loss given default. The 95% interval is
[−10.3, +47.5] defaults, so it includes zero. At every operating point from 50% to 90% approval the
interval includes zero: the ranking gain is not yet a demonstrated decision gain.

> These results describe a simulation. They are not evidence of real-world credit performance. Scores
> may move by roughly ±0.001 between runs because the models are retrained each time.

---

## Repository map

```
.
├── finrep-ntc-credit-risk.ipynb   ← the main document: whole study, end to end, with results
├── main.py                        ← command-line entry point
├── config.yaml                    ← all settings: seed, sizes, model hyper-parameters
├── requirements.txt               ← Python packages needed
├── requirements-lock.txt          ← pinned versions for reproducible installs
├── ntc_finrep_environment.yml     ← Conda environment (Python 3.11)
├── README.md                      ← this file
├── CONSISTENCY_LOG.md             ← record of methodology decisions and reruns
├── src/                           ← all the source code (explained below)
├── tests/                         ← automated tests (41 checks)
├── data/generated/                ← created when you run; ignored by Git
└── artifacts/                     ← results written when you run; ignored by Git
```

`data/generated/` (about 120 MB) and `artifacts/` are rebuilt by the code on every run, and both are
listed in `.gitignore` so they stay out of version control. This particular copy of the folder ships
with one completed 48,000-applicant run already in place; delete both to start from scratch.

---

## How to run it

**Option A — the notebook (recommended).** Open `finrep-ntc-credit-risk.ipynb` and run all cells
top to bottom. It is self-contained and typically takes 15–45 minutes on a modern laptop, depending
on hardware; run the smoke test below first to gauge timing on your machine.

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

Each section below names a file, says what it is for in one line, then explains every function inside
it. "Returns" describes what you get back.

### `main.py` — the front door (11 lines)

A tiny launcher. It reads three optional command-line flags (`--config`, `--n-applicants`, `--epochs`)
and hands them to `src.experiment.main()`. Nothing else happens here, so you can read it in ten
seconds.

---

### `src/experiment.py` — the conductor (203 lines)

This file runs the whole study in the right order. If you only read one file, read this one. The heavy
lifting lives in three helper modules described next; this file is the running order, plus the audit
trail written to `artifacts/`.

| Function | What it does |
|---|---|
| `main(config_path, n_override, epochs_override, model_seed_override)` | The full pipeline, in order: load settings → build the dataset → pretrain the encoder and embed applicants → fit and compare the models → tune the benchmark, calibrate, measure label efficiency and score stability → write the split audit and run manifest. |
| `_write_json`, `_git_hash`, `_package_versions` | Small helpers that record what was run: JSON output, the commit hash when the folder is a Git repository, and the installed package versions. |

---

### `src/data/dataset.py` — the modelling dataset (142 lines)

Turns the simulated book into the four cohorts the study compares, and writes every input table to
`data/generated/`.

| Function | What it does |
|---|---|
| `temporal_split(applicants, valid_fraction, test_fraction)` | Sorts applicants by application date and cuts them into past / middle / future groups (train, validation, test). Whole dates stay together, so no applicant from the same month appears on both sides. Returns three tables. |
| `temporal_split_with_calibration(applicants, ...)` | The same idea with four groups: train, validation (for tuning), calibration (for probability scaling), and test. Returns four tables. |
| `build_dataset(cfg, seed, n_override)` | The one-call entry point: generate the book → build cash-flow features → keep the simulator's own probability as the audit ceiling → assemble the modelling table from IDs, application date and target only → split the vintages → write six parquet files. Returns a dictionary of frames, cohorts and oracle probabilities. |
| `TARGET` | The single definition of the outcome column, `default_12m`, so no other file can disagree about it. |

*Why the splitting matters:* using future data to predict the past, or testing on data the model was
tuned on, makes results look better than they are. Splitting by date prevents both.

---

### `src/representation_learning.py` — the self-supervised stage (65 lines)

| Function | What it does |
|---|---|
| `pretrain_encoder(applicants, transactions, train_ids, cfg, seed, out_dir, epochs)` | Builds sequences for training-vintage applicants only, splits them into a fit slice and a held-out slice, trains the encoder on the masked-transaction puzzle, saves the model and training history, and returns an audit record of exactly which applicants were used. |
| `embed_applicants(model, applicants, transactions, applicant_ids, max_length)` | Runs the frozen encoder over every applicant and returns one row of 32 numbers per person, plus the column names (`finrep_000` … `finrep_031`). |

The audit record exists so the pipeline can assert that no validation, calibration or test applicant
was ever seen during pretraining.

---

### `src/risk_modelling.py` — the supervised stage (147 lines)

| Function | What it does |
|---|---|
| `fit_feature_sets(...)` | Fits all three model families on every feature set from one shared labelled sample, saves the pipelines, and returns a combined metrics table. |
| `tune_feature_sets(...)` | A small grid over two LightGBM settings, selected on validation ROC-AUC only. The test cohort is scored once, after selection. |
| `oracle_ceiling(test, oracle_by_id, test_ids)` | Scores the simulator's own noise-free probability, which sets the ceiling a real model could reach. |
| `calibrate_hybrid(...)` | Platt scaling fitted on the calibration vintage and reported on test. |
| `label_efficiency(...)` | Refits both feature sets at 5%, 10%, 25%, 50% and 100% of the training labels. |
| `score_stability(model, valid, test, cols)` | Population Stability Index between validation and test scores, as a drift check. |

---

### `src/data/generate_data.py` — the simulator (387 lines)

Creates the fake world: applicants, their transactions, and whether they defaulted.

**Fixed lists at the top**

| Name | Meaning |
|---|---|
| `CATEGORIES` | The ten transaction types: salary, business income, rent, groceries, utilities, travel, online shopping, medical, transfer, other. |
| `PROFILES` | Five applicant archetypes (stable salaried, volatile salaried, self-employed, high obligation, low liquidity). |
| `ACCOUNT_TYPES`, `HOUSING_STATUSES` | Account products and housing situations. |
| `GENERATOR_ONLY_COLUMNS` | The one authoritative list of columns that exist only to drive the simulation. These must never enter a model; the pipeline strips them and tests assert they are gone. |

**Functions**

| Function | What it does |
|---|---|
| `sigmoid(x)` | Squashes any number into the range 0–1, which is how probabilities are produced. |
| `generate_data(...)` | Builds the whole population. For each applicant it picks a profile, income, obligation level (share of income committed to fixed costs), opening balance, housing situation, account type, and four independent "stress" traits. It then writes 12 months of transactions before the application date, computes the true default probability, draws the default outcome, and returns two tables: one row per applicant, one row per transaction. |
| `main()` | A small command-line wrapper so you can generate data standalone, e.g. `python -m src.data.generate_data --n-applicants 4000`. |

**Two ideas worth understanding inside `generate_data`:**

*Cash-flow drivers* — obligation, income volatility, opening liquidity. These show up in monthly
totals, so hand-built features can see them.

*Sequence-stress drivers* — payment lateness, income timing, and spending tempo. These change **when**
money moves, not how much. A monthly total cannot see them; a transaction sequence can. That is the
signal the embeddings are supposed to recover.

The generator also stores two probabilities used only for auditing: `p_default_conditional` (the noise-free
ceiling) and `p_default_realized` (the value actually used to draw the label). Neither is ever a model
input.

---

### `src/data/sequences.py` — turning rows into a sequence (155 lines)

Neural networks cannot read a spreadsheet row; they need a fixed-shape numeric sequence. This file
does that conversion.

| Name / function | What it does |
|---|---|
| `CATEGORIES`, `CAT_TO_ID` | Fixed dictionary mapping each transaction type to a number. It is fixed on purpose, so the same category always has the same number. |
| `MASK_ID`, `PAD_ID`, `N_CATEGORIES` | Special numbers for "hidden during training" and "empty space at the end". |
| `N_TIME_BUCKETS` | Splits each month into 16 time slots, so *when* in the month a transaction happened is representable. |
| `encode_sequences(applicants, transactions, max_length)` | For every applicant: drop everything on or after the application date, keep the most recent transactions (up to `max_length`), and turn each into category, signed amount, amount relative to that person's typical transaction, time slot, direction (money in/out), and a padding flag. Returns one dictionary of fixed-width NumPy arrays per applicant — arrays rather than Python lists, which matters at 250k applicants where lists would cost several gigabytes. |
| `to_tensors(sequence_dict, applicant_ids, max_length)` | Stacks those per-applicant arrays into six big tensors (the array format PyTorch expects), in a fixed applicant order. |

---

### `src/features/cashflow_features.py` — the hand-built baseline (282 lines)

This is the "traditional" approach: summarise each person's transactions into 29 interpretable
numbers, then let a standard model use them.

| Name / function | What it does |
|---|---|
| `SPEND_CATEGORIES`, `INCOME_CATEGORIES` | `SPEND_CATEGORIES` maps each spending bucket (housing, groceries, utilities, transport, shopping, medical, other) to the transaction category that feeds it; `INCOME_CATEGORIES` lists the salary and business-income categories. Transfers are excluded from both because moving money between your own accounts is not spending or earning. |
| `build_cashflow_features(applicants, transactions)` | Produces the 29 features per applicant: average monthly credits and debits, balance levels, transaction counts, seven spending shares, income volatility, expense-to-income ratio, transfer in/out ratios, liquidity months, balance volatility, income frequency, account age and type shares, and housing payment size. Returns one row per applicant. |
| `feature_columns(features)` | Returns the list of columns to feed a model — everything except the ID. |
| `filter_extreme_applicants(...)` | **Legacy, not used to drop rows.** Computes IQR outlier bounds from the training cohort and would remove flagged applicants. The project no longer deletes applicants; the function is kept for reference only. |
| `winsorize_extreme_features(...)` | The replacement. Instead of deleting outliers, it caps extreme values at bounds learned from the training cohort only. Rows are preserved, so the test cohort is never shrunk or filtered using training statistics. |

*Why this matters:* deleting "weird" applicants using statistics from the training set would quietly
change the test population, which is a form of cheating. Capping values keeps every applicant.

---

### `src/models/transaction_encoder.py` — the neural network (77 lines)

| Name | What it does |
|---|---|
| `TransactionEncoder` | A small Transformer — the same family of architecture used in modern language models. It reads a sequence of transactions and produces one 32-number summary (the embedding) for that person. Defaults: 4 attention heads, 2 layers. |
| `TransactionEncoder.forward(...)` | The actual calculation: encode each transaction's category, amount, timing and direction, let the transactions "look at" each other, then average into a single vector. Also returns per-transaction outputs used during training. |
| `masked_category_loss(...)` | The training puzzle. It hides about 15% of transaction categories and scores how well the model guesses them back. Guessing well requires understanding typical transaction patterns, so the embedding becomes informative without ever using the default label. |

---

### `src/models/pretrain.py` — training and using the encoder (137 lines)

| Name / function | What it does |
|---|---|
| `SequenceDataset` | A small wrapper that lets PyTorch serve the tensors in batches. |
| `select_device()` | Picks the best available hardware: CUDA GPU, Apple Silicon GPU, or CPU. |
| `pretrain(tensors, output_path, ...)` | Trains the encoder on the masked-transaction puzzle, checks progress on a held-out slice of the training data, stops early if it stops improving, saves the trained model, and returns the training history. |
| `extract_embeddings(model, tensors, batch_size)` | Runs a trained encoder over applicants *without* hiding anything and returns one 32-number vector per applicant. |

Only training-period applicants are used for pretraining; validation and test applicants are encoded
later with a frozen model. That keeps the comparison honest.

---

### `src/models/factory.py` — model builder (14 lines)

| Function | What it does |
|---|---|
| `make_lgbm(params, seed)` | Builds a LightGBM classifier wrapped in a small pipeline that fills missing values first. LightGBM is a tree-based model: many simple decision rules combined into a strong one. |

---

### `src/models/risk_model.py` — the three comparison models (46 lines)

| Function | What it does |
|---|---|
| `fit_and_evaluate(...)` | Trains three model families — logistic regression (simple and transparent), histogram gradient boosting, and LightGBM — on the training cohort, scores them on validation and test, saves each fitted model, and returns a results table. Any scaling or missing-value filling is fitted on training rows only. |

The same function is used with cash-flow features and again with cash-flow plus embeddings, which is
what makes the comparison fair: identical models, identical settings, only the inputs differ.

---

### `src/evaluation/metrics.py` — how performance is measured (68 lines)

| Function | What it does |
|---|---|
| `ks_statistic(y_true, probability)` | Measures how far apart the score distributions of defaulters and non-defaulters are. Bigger is better. |
| `expected_calibration_error(y_true, probability, n_bins)` | Checks whether predicted probabilities mean what they say. If the model says "20% risk" for a group, about 20% of that group should default. Lower is better. |
| `evaluate(y_true, probability)` | Bundles the standard numbers: ROC-AUC (ranking quality), PR-AUC (ranking quality when defaults are rare), KS, Brier score (probability accuracy), plus the cohort size and default rate. |
| `psi(expected, actual, buckets)` | Population Stability Index: compares two score distributions. Below about 0.1 means the model behaves consistently between cohorts. |

*Plain-language metric guide:* ROC-AUC 0.5 = random guessing, 1.0 = perfect ordering. PR-AUC depends
on how common defaults are. Brier score is an average squared error on probabilities, so lower is
better.

---

### `src/evaluation/embedding_diagnostics.py` — sanity check on the embeddings (66 lines)

| Function | What it does |
|---|---|
| `embedding_activity_diagnostics(embeddings, features, applicant_ids)` | Correlates each of the 32 learned dimensions with real activity measures (transaction count, months observed, income, spending ratio). It answers two questions: are the embeddings alive (not all identical or zero), and do they relate to actual behaviour rather than noise? Returns a JSON-ready summary. |

---

### `src/evaluation/approval_policy.py` — from ranking to decisions (228 lines)

ROC-AUC describes how well a model sorts the whole population. A lender only cares about the
applicants near its cutoff. This module turns a score into an approve/reject decision and a money
figure, which is the question a ranking metric cannot answer on its own.

| Function | What it does |
|---|---|
| `cutoff_at_approval_rate(scores, approval_rate)` | The score quantile that approves the lowest-risk `approval_rate` share of applicants. |
| `approval_policy_table(...)` | One row per model and approval rate: approved count, approved default rate, defaults captured, and expected loss at the stated exposure (EAD) and loss given default (LGD). |
| `approval_policy_deltas(...)` | Challenger minus baseline at every rate: change in bad rate, defaults avoided per 10,000 approved, and loss avoided. |
| `bootstrap_policy_delta(...)` | A paired bootstrap interval on those avoided defaults and dollars, so the result comes with uncertainty. |
| `swap_analysis(...)` | Who is approved by one model but rejected by the other, and how risky each swapped group actually is. |
| `fixed_cutoff_comparison(...)` | The reverse experiment: one shared threshold, so the approval rate floats instead. |
| `capture_curve(...)` | Cumulative share of defaults captured as more applicants are rejected, for plotting. |

Holding the approval rate fixed keeps the number of loans constant, so any difference is a pure mix
effect rather than a change in volume. That is what lets the result be read as "defaults avoided".
The fixed-cutoff view is reported alongside it because the two are different policies, and a model
that looks good under one is not automatically good under the other.

---

### `src/decile.py` — risk ranking tables (48 lines)

| Function | What it does |
|---|---|
| `decile_analysis(y_true, y_prob, n_bins=10)` | Sorts applicants by predicted risk and splits them into ten equal groups. For each group it reports the observed default rate, how many defaults it contains, and "lift" (how much riskier than average this group is). |
| `pentile_analysis(y_true, y_prob, n_bins=5)` | The same thing with five groups. Used in the notebook to answer the practical question: if we flag the riskiest 20% or 40% of applicants, how many defaults do we catch? |
| `_capture_table(...)` | The private helper both functions share. It ranks the scores before binning, so applicants with identical scores cannot collapse into a single group, then summarises each bin. |

These tables translate abstract scores into something a credit team can act on.

---

### `src/__init__.py` and the `__init__.py` files in sub-folders

Empty marker files. They tell Python that `src`, `src.data`, `src.models` and so on are importable
packages. No logic lives in them.

---

### `tests/` — the automated checks (41 tests)

| File | What it guards |
|---|---|
| `test_generation.py` | The simulator behaves as documented: sequence-stress traits are visible only in timing, generator-only columns exist, conditional and realised probabilities differ correctly, and the prevalence intercept sets the default rate. |
| `test_features.py` | Features are leakage-safe, computed only from pre-application transactions, and winsorisation preserves every row. |
| `test_sequences.py` | Sequence encoding keeps the right window, respects ordering, and pads correctly. |
| `test_models.py` | The encoder's output shapes are right, padding does not change results, saved models reproduce their predictions, and seeds are honoured. |
| `test_metrics.py` | Metric implementations give the expected values on known cases. |
| `test_decile.py` | Bucketing, lift and capture add up correctly. |
| `test_temporal_split.py` | Splits never mix dates across cohorts. |
| `test_approval_policy.py` | Approval cutoffs, policy deltas, bootstrap intervals, swap counts, fixed-cutoff comparisons and capture curves give the expected answers on known cases. |
| `test_experiment.py` | A small end-to-end smoke run writes its manifest and audit files. |

Run them with `python -m pytest -q`. They should all pass before you trust any result.

---

## Glossary

| Term | Plain meaning |
|---|---|
| **Default** | Failing to repay a loan on the agreed terms within the observation window. |
| **Feature** | A hand-built number describing a person, e.g. average monthly income. |
| **Embedding** | A short learned list of numbers that represents a person, written by the neural network rather than by a human. |
| **Transformer** | A neural network architecture that weighs how each item in a sequence relates to the others; the basis of modern language models. |
| **Self-supervised** | Training on a puzzle generated from the data itself, without needing the answer label. |
| **Latent** | A hidden variable the simulator uses internally. Never a model input. |
| **Leakage** | Accidentally letting a model see information it would not have at decision time. Makes results look better than reality. |
| **Chronological split** | Splitting data by date so the model is trained on the past and tested on the future. |
| **Calibration** | Adjusting predicted probabilities so that "30% risk" really means about 30 in 100. |
| **ROC-AUC** | Chance that a random defaulter is ranked riskier than a random non-defaulter. 0.5 = coin flip. |
| **PR-AUC** | Ranking quality focused on the defaulters, useful when defaults are rare. |
| **KS** | Largest gap between the score distributions of defaulters and non-defaulters. |
| **Brier score** | Average squared error of predicted probabilities; lower is better. |
| **PSI** | Score-drift measure; near zero means the model scores two cohorts similarly. |
| **Bootstrap interval** | A range built by repeatedly resampling the test cohort, used to show how uncertain a difference is. |
| **Oracle** | The simulator's true probability, known only to the experimenter. It sets the ceiling a model could reach. |

---

## What is deliberately *not* done

- No post-application data is generated or used.
- Generator-only columns are stripped before modelling and asserted absent by tests.
- The encoder is pretrained on training-period applicants only.
- Tuning uses the validation cohort; calibration uses its own cohort; the test cohort is touched once.
- Extreme values are capped, not deleted, so the test cohort is never filtered with training statistics.
- The random-embedding control has exactly the same number of inputs as the real embedding model, so
  a gain cannot be explained by extra dimensions alone.

## Honest limitations

- The simulator is a model of reality, not reality. Its "stress" signal was designed to be learnable.
- Results depend on one synthetic dataset, one pretrained encoder and one tuned benchmark.
- Confidence intervals capture sampling noise on the test cohort, not uncertainty about the simulator
  itself.
- A small improvement on this benchmark says the method *can* recover timing information when it
  exists. It does not say real transaction data contains an equivalent signal, or that a bank should
  use this to make lending decisions.

## Licence

See `LICENSE`.

# Notebook consistency work log

**Notebook:** `finrep-ntc-credit-risk.ipynb`
**Last updated:** 2026-11-11
**Environment:** `~/miniconda3/envs/ntc-finrep/bin/python`

This log is written so another agent can pick up the work if this session ends.
Each task is independent; mark it DONE only after verifying the notebook source,
outputs, and any referenced artifact agree.

---

## Ground truth for the executed run

- Data-generation / model seed: **42** (cell 6 pins this explicitly)
- Applicants: **48,000**
- History: **12 months**
- Pretraining epochs: **16**
- Transactions: **4,777,993**
- After IQR filtering: **45,889** applicants (2,111 removed)
- Splits after filtering: **train 27,951 / validation 6,435 / calibration 2,578 / test 8,925**
- Test default rate: **21.86%**
- Current run artifacts are dated **2026-11-09**; stale smoke-run artifacts are dated
  **2026-11-04 / 2026-11-05**.

Verified: `generate_data(48000, 12, seed=42)` with the optimised generator
reproduces `data/generated/applicants.parquet` and `transactions.parquet`
exactly, including dtypes.

### Current key results

| Comparison (LightGBM) | ROC-AUC |
|---|---:|
| Cash-flow | 0.677984 |
| Hybrid (cash-flow + FinRep) | 0.679808 |
| Random-embedding control | 0.679630 |
| Tuned cash-flow | 0.680505 |

Paired bootstrap (hybrid − cash-flow): **+0.00179**, 95% CI
**[−0.00170, +0.00553]** → not statistically distinguishable.
Hybrid − tuned: **−0.00069** [−0.00392, +0.00256].
Hybrid − random: **+0.00009** [−0.00456, +0.00458].

---

## Task list

| # | Task | Status |
|---|------|--------|
| 1 | Seed mismatch: cell 6 says 2023; run is 42 | DONE |
| 2 | Cell 38 stale default-rate text (22.07/23.06/22.17) | DONE |
| 3 | Cell 40 split-label bug: uses `"valid"` instead of `"validation"`; omits calibration | DONE |
| 4 | Cell 100 stale calibration artifact (27 cal applicants, n=44) | DONE |
| 5 | Cell 109 reads stale `05_decile_analysis.csv` instead of current pentile artifact | DONE |
| 6 | Decile-vs-pentile narrative (cells 105, 108, 110) | DONE |
| 7 | Cell 154 stale conclusion checklist (6,903 / −0.00195 etc.) | DONE |
| 8 | Archive stale 2026-11-04/05 artifacts out of `artifacts/` | DONE |

---

## Detailed tasks and decisions

### Task 1 — Seed mismatch
- Cell 6 output says `Selected seed: 2023`; `artifacts/generation_seed.json` says 2023.
- Cell 8 output, pretraining audit, all models/embeddings use seed **42**.
- Execution counts: cell 8 `exec=4`, cell 6 `exec=70` → cell 6 was re-run late.
- Decision: pin the final notebook to seed **42** (one of the configured seeds).
  Change cell 6 to `SEED = 42` with an explanatory comment, update its stdout,
  and rewrite `generation_seed.json` to seed 42.

### Task 2 — Cell 38 default rates
- Stale: 22.07% train / 23.06% validation / 22.17% test.
- Current post-filter: **22.17% / 22.33% / 21.86%**.

### Task 3 — Cell 40 split labels
- Current source uses `split == "valid"`; split labels are `"validation"`.
  It also omits `calibration`.
- Fix to select `"train"`, `"validation"`, `"calibration"`, `"test"`.
  This is latent in the executed run because cells 46/86 reload splits from disk.

### Task 4 — Cell 100 calibration
- Current cell reads `artifacts/calibration_metrics.json`, which is a 300-applicant
  smoke-run artifact: 27 calibration applicants, test n=44, Brier 0.1047,
  ECE 0.1212.
- Current calibration cohort has **2,578** applicants and test has **8,925**.
- Decision: make cell 100 compute Platt scaling inline (or regenerate the
  artifact from the current saved models) using the hybrid LightGBM model,
  calibration cohort, and test cohort, matching `src/experiment.py` lines 290–317.
- Save the new `calibration_metrics.json` and
  `artifacts/hybrid_models/platt_calibrator.joblib`.

### Task 5 — Cell 109 stale decile file
- Cell 109 reads `artifacts/05_decile_analysis.csv` (2026-11-05, 4,660 test rows).
- Current analysis is `artifacts/05_pentile_analysis.csv` (2026-11-09, 8,925 rows).
- Decision: rewrite cell 109 to read the pentile artifact and report pentile 5
  lift + top-20% / top-40% capture, or remove the stale decile narrative.

### Task 6 — Decile vs pentile narrative
- Cell 105 is a full “Decile Analysis” section; actual analysis is pentile.
- Cell 108 says “decile lift … from 05_decile_analysis.csv”.
- Cell 110 refers to “displayed decile-wise lift chart”.
- Decision: align wording to pentile (5 groups), keep the qualitative conclusion.

### Task 7 — Cell 154 conclusion checklist
- Stale: 6,903 applicants, 22.38% default; hybrid 0.6816 vs cash-flow 0.6835;
  Δ −0.00195; tuned 0.6872; Δ −0.00557; hybrid−random 0.00044;
  PR-AUC Δ −0.00469.
- Current: 8,925 applicants, 21.86%; hybrid 0.679808 vs cash-flow 0.677984;
  Δ +0.00179 [−0.00170, +0.00553]; tuned 0.680505; Δ −0.00069
  [−0.00392, +0.00256]; hybrid−random +0.00009 [−0.00456, +0.00458];
  PR-AUC Δ +0.00211 [−0.00521, +0.00950].
- Decision: rewrite section 7 with current values. Keep the qualitative
  conclusion: no established incremental value; tuned cash-flow is at least
  as good; random control is indistinguishable.

### Task 8 — Stale artifacts
Stale files (2026-11-04/05), to be moved to `artifacts/_stale_2026-11-04/`
(not deleted, so the audit trail is retained):
- `calibration_metrics.json` (will be regenerated by Task 4)
- `oracle_ceiling_metrics.json`
- `strong_baseline_metrics.json`
- `embedding_diagnostics.json`
- `split_summary.json`
- `run_manifest.json`
- `05_decile_analysis.csv`
- `hybrid_models/platt_calibrator.joblib` (will be regenerated by Task 4)

Current (2026-11-09) artifacts to keep:
- `model_metrics.csv`, `05_pentile_analysis.csv`, `label_efficiency.csv`,
  `score_psi.json`, `experiment_summary.json`, `pretraining_*`,
  `baseline_models/*`, `hybrid_models/logistic.joblib`,
  `hybrid_models/hist_gradient_boosting.joblib`, `hybrid_models/lightgbm.joblib`,
  `07_*`, `08_*`.

---

## Progress notes

- 2026-11-10: Created log. No notebook edits made yet.
- 2026-11-10: **Task 1 DONE.** Cell 6 now pins `SEED = 42` with an explanatory
  comment; its stdout reads `Selected seed: 42`; `artifacts/generation_seed.json`
  rewritten to seed 42. Pre-edit notebook backup at
  `/private/var/folders/t1/1c9kbdc16ks7tv2vbflw0ckc0000gn/T/opencode/finrep-ntc-credit-risk.pre-consistency.ipynb`.
- 2026-11-10: **Task 2 DONE.** Cell 38 now reads: post-IQR default rates
  22.17% train / 22.33% validation / 22.42% calibration / 21.86% test
  ("four cohorts"). Verified against `split_assignments.parquet` +
  `applicant_modeling.parquet`.
- 2026-11-10: **Task 3 DONE.** Cell 40 now recreates `train`, `valid`
  (`split == "validation"`), `calibration` and `test` from `split_rows`,
  so the label bug and the missing calibration cohort are fixed.
- 2026-11-10: **Task 4 DONE.** Regenerated calibration from the saved
  `artifacts/hybrid_models/lightgbm.joblib` on the current cohorts, exactly
  following `src/experiment.py`:
  - calibration n = **2,578**, test n = **8,925**
  - raw model reproduces `model_metrics.csv` (ROC-AUC 0.679808, Brier 0.157977)
  - calibrated test Brier **0.157959**, ECE **0.008500** (ROC-AUC unchanged)
  - `artifacts/calibration_metrics.json` and
    `artifacts/hybrid_models/platt_calibrator.joblib` rewritten
  - old smoke-run versions archived to
    `artifacts/_stale_2026-11-04/{calibration_metrics.json,platt_calibrator.joblib}`
  - cell 100 output now: "Platt scaling used 2,578 ... Brier 0.1580 ... ECE 0.0085."
- 2026-11-10: **Task 5 DONE.** Cell 109 now reads
  `artifacts/05_pentile_analysis.csv`, computes pentile-5 lift and top-20% /
  top-40% capture from `default_capture_pct`, and all six output strings were
  replaced with current values (lift 1.69–1.87; top-20% 33.7–37.3%;
  top-40% 58.1–60.2%).
- 2026-11-10: **Task 6 DONE.** Decile→pentile wording aligned in cells 105,
  106 (plot title), 107, 108 and 110; cell 107 heading demoted from "## 4." to
  "### 6.1" to fix the 6→4 numbering jump; cell 104 renamed
  `decile_summary`→`pentile_summary` and `D1/D5`→`P1/P5`, with its displayed
  output updated to match. Only remaining "decile" token is the
  `from src.decile import pentile_analysis` module name.
- 2026-11-10: **Task 7 DONE.** Cell 154 rewritten with current numbers:
  cohort 8,925 / 21.86%; hybrid 0.6798 vs cash-flow 0.6780 (+0.00179,
  CI [−0.00170, +0.00553]); tuned 0.6805 (−0.00069, CI [−0.00392, +0.00256]);
  random control (+0.00009, CI [−0.00456, +0.00458]); PR-AUC +0.00211
  [−0.00521, +0.00950]; five-seed means 0.67847 / 0.67911 / 0.67786;
  Platt Brier 0.1580 / ECE 0.0085 vs raw 0.1580 / 0.0079; PSI 0.0012.
  The qualitative conclusion is unchanged (no established incremental value).
- 2026-11-10: **Task 8 DONE.** Moved the six remaining stale artifacts into
  `artifacts/_stale_2026-11-04/`: `oracle_ceiling_metrics.json`,
  `strong_baseline_metrics.json`, `embedding_diagnostics.json`,
  `split_summary.json`, `run_manifest.json`, `05_decile_analysis.csv`
  (old `calibration_metrics.json` and `platt_calibrator.joblib` were already
  archived there before regeneration in Task 4). Nothing was deleted.
  Verified no notebook cell reads any of the moved files.

### Final verification (2026-11-10)

- Notebook JSON loads; 156 cells, 68 with outputs; all `cell_type` values valid.
- Stale-token scan (old counts/metrics, `05_decile_analysis.csv`,
  `Selected seed: 2023`) returns no hits in sources or outputs.
- Only remaining "decile" token is `from src.decile import pentile_analysis`.
- Cell 100 output matches `artifacts/calibration_metrics.json`
  (2,578; Brier 0.1580; ECE 0.0085).
- Cell 109 outputs exactly match a fresh recomputation from
  `artifacts/05_pentile_analysis.csv` (all six rows).
- `pytest tests -q` → **25 passed**.
- `config.yaml` seed 42 / 48,000 / 12 months / 16 epochs and
  `artifacts/generation_seed.json` seed 42 agree.

### Seed alignment fixes (2026-11-11)

- Primary notebook cell 6 now derives `SEED_LIST` from
  `config.yaml: seed_choices` instead of a duplicated hard-coded list, and
  raises if the pinned `SEED = 42` is missing from that list.
- Primary notebook cell 74 now passes `random_state=SEED` to PCA instead of a
  literal `42`.
- `01_data_generation.ipynb` now pins `SEED = 42` instead of sampling with
  `secrets.choice`, matching the CLI and final report; its stored seed stdout
  was updated accordingly.
- `07_robustness_experiments.ipynb` and `08_final_model_comparison.ipynb` now
  derive `SEED_LIST` from `config.yaml`, replacing the stale
  `[42, 123, 2024, 2025, 2026]` list.
- `src/models/risk_model.py` now passes the supplied `seed` to logistic
  regression and histogram gradient boosting instead of hard-coding `42`.
- Added `test_risk_models_use_supplied_seed`; `pytest tests -q` → **26 passed**.

### 11 Nov rerun and output clearing (2026-11-11)

- `data/generated/` and the model/evaluation artifacts were regenerated at
  00:17–00:32 on 11 Nov. The current hybrid LightGBM test ROC-AUC is
  **0.678448** (cash-flow **0.677984**).
- The primary notebook was saved with all outputs cleared at 08:51 on 11 Nov:
  81 code cells, 0 outputs, all execution counts null. The earlier
  `Final verification (2026-11-10)` statement of 68 outputs is historical only.

### Calibration and conclusion sync (2026-11-11)

- Archived the stale calibration files to `artifacts/_stale_2026-11-10/`.
- Re-fitted Platt scaling from the current
  `artifacts/hybrid_models/lightgbm.joblib` on the 2,578 calibration
  applicants and rewrote `calibration_metrics.json` and
  `hybrid_models/platt_calibrator.joblib`.
- Calibrated test metrics: n = **8,925**, ROC-AUC **0.678448** (unchanged),
  Brier **0.158095**, ECE **0.010544**. Calibration did not improve
  probability accuracy; ECE was higher than the raw hybrid ECE of 0.008061.
- Rewrote cell 154 from the current `08_paired_difference_intervals.csv`,
  `07_five_seed_summary.csv`, `score_psi.json` and the regenerated calibration
  report. Key values: hybrid 0.6784 vs cash-flow 0.6780; paired difference
  +0.00043 [−0.00312, +0.00417]; tuned 0.6805, difference −0.00205
  [−0.00543, +0.00108]; random control 0.6796, hybrid−random −0.00127
  [−0.00595, +0.00335]; five-seed hybrid mean 0.67885; PSI 0.0022.
- Verified no old cell-154 tokens remain.

Remaining known drift: stage notebooks 02/05/06/07/08 still carry outputs from
the older 6,903-test cohort; re-executing the primary notebook is optional;
`deepseek_suggestions.md` still cites the old +0.00179 comparison.


### Generator and feature redesign (2026-11-11)

This is a deliberate, pre-specified change to the simulation and baseline
feature set. It invalidates the 11 Nov data and model artifacts until the
pipeline is rerun end to end.

**Generator (`src/data/generate_data.py`)**

- Added sequence-only stress drivers that shape *when* transactions happen but
  not their monthly totals: payment lateness (housing/utilities), income-timing
  irregularity (salary/business income day), and within-month spending tempo.
  The stress index also becomes slightly more visible in the final quarter.
- Added applicant-specific Dirichlet category weights instead of one fixed
  category split for everyone.
- Aligned initial liquidity with profile (`LOW_LIQUIDITY` gets a smaller
  opening buffer; `HIGH_OBLIGATION` gets an intermediate buffer).
- Let housing status change total fixed costs, not only the housing label.
- Added audit-only latent columns: `obligation_latent`, `income_cv_latent`,
  `initial_balance_ratio_latent`, `sequence_stress_latent`,
  `payment_risk_latent`, `income_timing_risk_latent`,
  `spending_tempo_risk_latent`, `transfer_reliance_risk_latent`.
- Split the default probability into `p_default_conditional` (drivers only,
  used for the oracle) and `p_default_realized` (includes the noise draw).
  `default_probability` remains as a backwards-compatible alias for the
  realised probability.
- Added the shared `GENERATOR_ONLY_COLUMNS` allow-list so notebooks and the CLI
  cannot silently disagree about leakage exclusions.
- The default-rate intercept was retuned from `-2.35` to `-2.47` to keep the
  simulated default rate near its previous level after adding sequence stress.

**Features (`src/features/cashflow_features.py`)**

- `income_cv` is now computed from monthly `SALARY`/`BUSINESS_INCOME` only.
  Transfers no longer enter the income volatility measure.
- Added `expense_to_income_ratio` (non-transfer debits over income) and
  separate `transfer_in_ratio` / `transfer_out_ratio`.
- Replaced `income_gap_days` with the single frequency measure
  `income_event_frequency`; removed `income_events`, `income_total`,
  `median_income_gap_days`, and `net_monthly_cashflow` from the model feature
  list to avoid duplicate level and count measures.
- Removed constant/redundant features: `account_type_count`,
  `account_history_coverage`, `balance_volatility_ratio`,
  `housing_payment_events`, `housing_payment_months`, `housing_payment_total`,
  `housing_payment_std`, `housing_payment_cv`, and
  `housing_payment_regularity`.
- Housing features are now derived from the `RENT` transaction category rather
  than generator metadata. `has_housing_payment` replaces
  `housing_payment_regularity`; `has_renting_payment` /
  `has_mortgage_payment` were removed as redundant and metadata-dependent.
- Removed the unused duplicate `src/features/application_features.py`.
- Added `winsorize_extreme_features` for train-fitted capping without dropping
  applicants.

**Pipeline and notebooks**

- `src/experiment.py` now uses the shared `GENERATOR_ONLY_COLUMNS` list and
  computes the oracle from `p_default_conditional` rather than the
  noise-inclusive realised probability.
- The primary notebook no longer drops applicants with train-fitted IQR
  bounds. The IQR review is diagnostic only; all cohorts keep their original
  temporal assignment. This aligns the notebook with the CLI, which never
  applied the row filter.
- Notebook 02 now builds the modelling table from identifiers, timing and
  target only, matching `src/experiment.py`, instead of merging all generator
  columns and dropping a stale three-name list.
- Notebook 02 outputs were cleared because they referenced the old feature
  schema and old cohort size.
- Added generation and feature tests for sequence-timing signal, audit-only
  latents, conditional-versus-realised probabilities, income-only CV,
  non-transfer expense ratios, and row-preserving winsorization.
- `pytest tests -q` → **31 passed**.

**Artifact status after rerun:** the full 48,000-applicant / 16-epoch pipeline
was rerun on 11 November 2026, followed by `finrep-ntc-credit-risk.ipynb`.
Files under `data/generated/` and `artifacts/` now describe the redesigned
generator and the metrics quoted in the notebook. See the rerun section below.

### Post-redesign rerun (2026-11-11)

The pipeline and the combined notebook were executed end to end after the
generator and feature redesign. `finrep-ntc-credit-risk.ipynb` is the
authoritative document; the numbered stage notebooks are superseded.

**Cohort:** 48,000 applicants; 29,269 train / 6,737 validation / 2,676
calibration / 9,318 test; simulated test default rate 23.35%.

**Test discrimination (LightGBM, notebook artefacts):**

| Model | ROC-AUC | PR-AUC | KS | Brier |
|---|---|---|---|---|
| Cash-flow | 0.7001 | 0.3850 | 0.2985 | 0.1636 |
| Cash-flow + random dimensions | 0.6980 | 0.3825 | 0.2960 | 0.1638 |
| Cash-flow + FinRep | 0.7081 | 0.4040 | 0.3183 | 0.1619 |
| Tuned cash-flow | 0.7025 | 0.3911 | 0.2997 | 0.1629 |

**Paired bootstrap (1,000 resamples), hybrid minus comparator:**

| Comparison | ROC-AUC | PR-AUC | KS | Brier |
|---|---|---|---|---|
| vs cash-flow | +0.0081 [0.0036, 0.0127] | +0.0190 [0.0099, 0.0284] | +0.0177 [0.0027, 0.0320] | −0.0017 [−0.0025, −0.0008] |
| vs tuned cash-flow | +0.0056 [0.0016, 0.0101] | +0.0130 [0.0042, 0.0223] | +0.0159 [0.0009, 0.0295] | −0.0010 [−0.0018, −0.0003] |
| vs random control | +0.0101 [0.0053, 0.0153] | +0.0216 | +0.0218 | −0.0019 |

All primary intervals exclude zero. Across five supervised seeds the hybrid
mean ROC-AUC was 0.70827 (SD 0.00042), cash-flow 0.69990 (0.00038) and the
random control 0.69925 (0.00160); the hybrid leads in all five seeds. The
hybrid also leads at every label fraction from 5% to 100%. The simulator-only
conditional-probability oracle reaches 0.7441 ROC-AUC, so headroom remains.
Platt calibration (2,676 applicants) gives Brier 0.1620 and ECE 0.0135 versus
0.1619 and 0.0111 for raw hybrid scores; validation-to-test score PSI is
0.0059.

**Conclusion change:** the earlier "no distinguishable difference" finding no
longer holds. On the redesigned simulator, FinRep embeddings add a small but
statistically distinguishable improvement over both the cash-flow-only and
tuned benchmarks, and the random-embedding control shows the gain is not
explained by extra dimensions. This remains a statement about the simulation,
not about real-world credit performance.

**Single-notebook consolidation:** all narrative cells (101, 110, 154) and
metric cells were refreshed from the rerun. Stale cross-references to the
numbered notebooks were removed. `pytest tests -q` → **31 passed** after the
rerun.

### Section numbering standardised (2026-11-13)

The notebook previously mixed two schemes: Part headings (`Part A`, `Part A.1`,
`A.1.2`) for the early sections and residual stage-notebook numbering for the
later ones (`# 07 — Robustness ...`, `# 08 — Final ...`, plus bare `## 4.`,
`## 5.`, `## 6.` steps inside Part A and Parts B–C). The document now uses one
hierarchical system throughout:

- `# Part A` … `# Part D` for the four research parts.
- `## Part X.Y` for the subsections of each part.
- `### X.Y.Z` for numbered steps, and `#### X.Y.Z.W` for nested detail.
- Unnumbered descriptive headings (`Objective`, `Interpretation`,
  `Key Observations`, `Completion criteria`, …) sit one level below their
  numbered parent.

Resulting outline: A.1.1–A.1.5, A.2.1–A.2.6, B.1.1–B.1.5, B.1.3.1–B.1.3.7,
B.2.1–B.2.5, C.1.1–C.1.6, C.1.5.1–C.1.5.6, C.1.6.1, C.2.1–C.2.4,
D.1.1–D.1.7, D.2.1–D.2.7, D.2.7.1–D.2.7.3, D.3.1–D.3.4.

Two further consistency fixes in the overview cell: the "eight sequential
notebooks" wording and the `01`–`08` subsection labels in the Research
Framework now map to the Part names, and the stale sentence stating that an
absence of improvement is expected was rewritten so it no longer contradicts
the Section D.2.7 conclusion.

Only markdown cells changed; all 81 code cells and their stored outputs are
unchanged, so no re-execution was required. Artefact filenames such as
`08_test_point_metrics.csv` retain their stage prefixes because they are file
names written by the pipeline, not document sections.

### Duplicate cells removed and notebook re-executed (2026-11-14)

Five back-to-back repetitions were removed, with user approval:

- `A.2.5`/`A.2.6`: the build cell displayed `applicant_modeling.head()`
  and the reload cell displayed the identical frame again. The reload now
  prints the shape instead.
- `B.2.5`: PCA was fitted twice on the same embedding matrix, once for the
  scatter plot and once for the quartile table and heatmap, and the explained
  variance was printed twice. There is now a single fit whose components
  feed all three views.
- `C.1.6`: the raw 30-row pentile table was displayed under the label
  "Pentile analysis summary" immediately before the real 6-row summary. The
  raw table still goes to `05_pentile_analysis.csv`; only the summary is
  displayed.
- `A.2.1`: repeated `head()` displays of `applicants` and `transactions`
  were removed; shapes and column counts are kept.
- `D.2.1`: a `print()` block repeating the cohort sizes, feature counts and
  default prevalence from `D.1.2` was removed.

Verified after re-execution: no two cells emit byte-identical output blocks,
no cell raises an error, and all 81 code cells carry an execution count.

**Metric drift from re-execution.** The notebook retrains its own models, so a
fresh run reproduces generator and oracle artefacts exactly but moves the
supervised metrics in the fourth decimal:

| Quantity | Before | After |
|---|---|---|
| Hybrid ROC-AUC | 0.70813 | 0.70819 |
| Hybrid PR-AUC | 0.40399 | 0.40177 |
| Hybrid KS | 0.31827 | 0.31938 |
| Hybrid Brier | 0.16193 | 0.16202 |
| Hybrid − cash-flow PR-AUC | +0.0190 [0.0099, 0.0284] | +0.0167 [0.0075, 0.0261] |
| Hybrid − tuned ROC-AUC | +0.0056 [0.0016, 0.0101] | +0.0057 [0.0016, 0.0102] |
| Five-seed hybrid mean | 0.70827 | 0.70851 |
| Validation-vs-test PSI | 0.0059 | 0.0041 |

The qualitative conclusion is unchanged: every primary interval still
excludes zero, the hybrid still leads all five seeds and every label
fraction. Narrative cells 104, 113 and 157 were refreshed to the values from
this run, and the numbers in the 2026-11-11 table above describe that earlier
execution rather than the current file. `pytest tests -q` → **31 passed**.

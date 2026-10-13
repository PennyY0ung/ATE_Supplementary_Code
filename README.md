# Outdoor Thermal Satisfaction — Modeling Code

This repository contains the original 12A → 12B → 13-0 modeling workflow with **English file, worksheet, output-folder, and result-worksheet names**. The training methodology was **not revised** as part of this naming update.

## Input dataset

Place `Analytical_Dataset_1001.xlsx` beside the three `.py` scripts before running them. The required Excel workbook has two sheets:

| Worksheet | Samples | Measurement groups | Survey days |
|---|---:|---:|---:|
| `summer` | 477 | 190 | 4 |
| `winter` | 524 | 296 | 4 |

**Important:** The input uses **all 1,001 observations** from the original August 13 cleaned dataset; it does **not** train on the 200-observation analytical subset. Its 32 columns have the same names and order as `Analytical_Subset_200.xlsx`. Four unused columns from the original workbook were omitted, without changing any retained cell value. `Thermal_satisfaction_3` remains the three-category outcome coded 1/2/3. The data's `Season` column still uses `Summer`/`Winter`; only the Excel **worksheet names** are lowercase `summer`/`winter`.

The complete 1,001-record dataset is **not included in this code-only repository archive**. Check data-sharing consent and ethics requirements before publishing respondent-level information.

## Files

- `12A_BroadBayesian_NestedLODO_SevenModels.py` — broad Bayesian screening of seven model families under nested leave-one-day-out validation.
- `12B_TargetedBayesian_Stability_NestedLODO_FiveModels.py` — targeted Bayesian optimization and inner-only stability-based selection of five candidate families; reads 12A's completed SQLite study history.
- `13-0_XGB_CrossFittedLODO_Evaluation_V2.py` — final XGB cross-fitted leave-one-day-out evaluation; reads the completed 12B Excel workbook.

## Install

Use a Python environment with the dependencies in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

## Run in order

```bash
python 12A_BroadBayesian_NestedLODO_SevenModels.py
python 12B_TargetedBayesian_Stability_NestedLODO_FiveModels.py
python 13-0_XGB_CrossFittedLODO_Evaluation_V2.py
```

All three scripts resolve default input and intermediate results relative to their own directory, without a hard-coded author's Windows user path.

Optionally, `12A` can be smoke-tested first:

```bash
python 12A_BroadBayesian_NestedLODO_SevenModels.py --smoke-test
```

Do not use smoke-test results for publication or as the completed 12A optimization history required by 12B.

## Output directory linkage

- 12A creates `12A_BroadBayesian_NestedLODO_SevenModels/` with `optuna_studies.db`.
- 12B reads that SQLite database and creates `12B_TargetedBayesian_Stability_NestedLODO_FiveModels/` with the matching `.xlsx` workbook.
- 13-0 reads the 12B workbook and creates `13-0_XGB_CrossFittedLODO_Evaluation_V2/`.

The worksheet names produced by 12B were renamed in concert with the worksheet references used by 13-0. The names of internal Optuna studies, validation procedures, candidate model families, hyperparameter settings, primary/secondary metrics, and random seeds remain those of the supplied original scripts.

Existing results using **old Chinese output directory / Excel worksheet names** are not automatically found by the new default paths. The complete workflow above is intended for a new run; do not mix old and new output artifacts by accident.

## Verification performed for this name-only release

- All 1,001 retained data records and all 32 output columns were compared cell-by-cell against the respective columns of the original full dataset (0 differences).
- All 200 example-subset records were found unchanged in the corresponding full source dataset.
- All three revised scripts successfully loaded the full English-named workbook and passed the original built-in row, date, group, and target-class checks.
- All three revised scripts passed syntax/CLI checks and an AST structural equivalence check ignoring only translated string literals and the default base-directory path.
- A `12A --smoke-test` end-to-end run completed. The full 12A/12B/13-0 optimization/evaluation sequence was **not** executed as part of this file-naming update.

Source comments, explanatory messages, and the scientific workflow remain as supplied; this is not a translation or redesign of the modeling code.

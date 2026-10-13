# Supplementary Analysis Code — Outdoor Thermal Satisfaction

This package provides three **separate, ordinary Python source files** corresponding to the three analysis stages reported in the manuscript. The files have been standardized for English-language review. Mathematical procedures, model families, feature definitions, data splits, hyperparameter search logic, random seeds, and evaluation metrics were not intentionally altered.

## Files and stage order

| File | Purpose | Principal output |
|---|---|---|
| `01_Model_Family_Screening.py` | Seven-family broad Bayesian model screening across the three predictor sets under nested leave-one-day-out validation | `Model_Family_Screening_Results/Model_Family_Screening_Results.xlsx`, plus `optuna_studies.db` |
| `02_Hyperparameter_Optimization_and_Model_Selection.py` | Five-family targeted search, stability assessment, and selection from **inner-training evidence** | `Hyperparameter_Optimization_Results/Hyperparameter_Optimization_Results.xlsx` |
| `03_Final_XGB_LODO_Evaluation.py` | Final XGB evaluation using the selected fold-specific configurations and held-out-day predictions, without retuning | `Final_XGB_LODO_Results/Final_XGB_LODO_Results.xlsx` and fold-specific predictions/models |

The stage numbers are for **navigation only**. The original source's Optuna study IDs and database table keys are deliberately retained to preserve provenance and resume compatibility.

## Required input dataset

The **complete cleaned analytical dataset (1,001 observations)** is necessary for formal replication and is **not included in this code archive or deposited publicly**. Requests for data access may be directed to the corresponding author and will be considered subject to applicable ethical and privacy restrictions, as specified in the manuscript's Data Availability Statement.

Place the complete workbook beside the scripts and name it `Full_Analytical_Dataset.xlsx`, **or specify its existing filename with `--input`**. For compatibility with the original source data, the workbook must still contain sheets named `夏季` (Summer, 477 records) and `冬季` (Winter, 524 records). Those two Chinese strings in `SHEET_MAP` are intentional input-schema identifiers, **not untranslated output labels**. The scripts also require the original column names, including `Date`, `measurement_group`, `Thermal_satisfaction_3`, the five measured microclimatic predictors, `Location`, `Scene`, and the twelve perception predictors.

The scripts verify four distinct survey days per season and the original frozen grouping structure. They operate on already-cleaned input data; they do **not** recreate the original raw-questionnaire cleaning stage.

## Dependencies

Install the libraries listed in `requirements.txt` in an isolated Python environment. These dependencies are **not an archival record of the software versions used for the original published runs**. Reproducing exact historical numeric outputs may require the original environment and full Optuna search histories.

```bash
python -m pip install -r requirements.txt
```

## Run the three stages separately

Run these commands from the directory containing the scripts. Replace the input path if you retain the historical Chinese filename.

```bash
python 01_Model_Family_Screening.py --input Full_Analytical_Dataset.xlsx --output-dir Model_Family_Screening_Results

python 02_Hyperparameter_Optimization_and_Model_Selection.py --input Full_Analytical_Dataset.xlsx --12a-db Model_Family_Screening_Results/optuna_studies.db --output-dir Hyperparameter_Optimization_Results

python 03_Final_XGB_LODO_Evaluation.py --input Full_Analytical_Dataset.xlsx --result12b Hyperparameter_Optimization_Results/Hyperparameter_Optimization_Results.xlsx
```


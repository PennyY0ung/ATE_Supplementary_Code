from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sqlite3
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

try:
    import optuna
    from optuna.pruners import NopPruner
    from optuna.samplers import TPESampler
except ImportError as exc:
    raise ImportError("Optuna is required. Install with: pip install optuna") from exc

try:
    from xgboost import XGBClassifier
except Exception as exc:
    raise ImportError("XGBoost is required for Stage 12B. Install xgboost.") from exc

try:
    from lightgbm import LGBMClassifier
except Exception as exc:
    raise ImportError("LightGBM is required for Stage 12B. Install lightgbm.") from exc

try:
    from catboost import CatBoostClassifier
except Exception as exc:
    raise ImportError("CatBoost is required for Stage 12B. Install catboost.") from exc


# =============================================================================
# Fixed input paths, data structure, and trial budgets
# =============================================================================

DEFAULT_BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = DEFAULT_BASE_DIR / "Full_Analytical_Dataset.xlsx"
DEFAULT_12A_DIR = DEFAULT_BASE_DIR / "Model_Family_Screening_Results"
DEFAULT_12A_DB = DEFAULT_12A_DIR / "optuna_studies.db"

RUN_DIR_NAME = "Hyperparameter_Optimization_Results"
OUTPUT_XLSX_NAME = "Hyperparameter_Optimization_Results.xlsx"
LOG_NAME = "Hyperparameter_Optimization.log"
STAGEA_DB_NAME = "12B_stageA_optuna.db"
CHECKPOINT_DB_NAME = "12B_stageB_outer_checkpoint.db"
SPACE_JSON_NAME = "12B_targeted_spaces.json"
CONFIG_JSON_NAME = "Hyperparameter_Optimization_Config.json"

SHEET_MAP = {"Summer": "夏季", "Winter": "冬季"}
EXPECTED_N = {"Summer": 477, "Winter": 524}
EXPECTED_GROUPS = {"Summer": 190, "Winter": 296}

TARGET = "Thermal_satisfaction_3"
RAW_TO_MODEL = {1: 0, 2: 1, 3: 2}
MODEL_TO_RAW = {0: 1, 1: 2, 2: 3}
CLASS_ORDER = [0, 1, 2]

OBJECTIVE = [
    "Temperature",
    "Wind_speed",
    "Humidity",
    "Globe_temperature",
    "Illumination",
]
SPATIAL = ["Location", "Scene"]
SUBJECTIVE = [
    "TSV", "TCV", "TPV", "TAV",
    "SSV", "SCV", "SPV", "SAV",
    "HSV", "WSV", "HPV", "WPV",
]
EP_NUMERIC = OBJECTIVE + SUBJECTIVE
EP_CATEGORICAL = SPATIAL
EP_FEATURES = EP_NUMERIC + EP_CATEGORICAL

CANDIDATE_MODELS = ["GB", "CAT", "ET", "LGBM", "XGB"]

RANDOM_STATE = 20260816
INNER_SPLITS = 3

# Targeted spaces use the frozen broad-search rules, drawing only on the current 12A history.
HISTORY_COMPLETE_TRIALS = 80
ELITE_N = 16
ELITE_Q_LOW = 0.10
ELITE_Q_HIGH = 0.90
TARGET_PADDING_FRAC = 0.10
TARGET_MIN_WIDTH_FRAC = 0.30
MIN_ACTIVE_ELITE = 5

# Stage A
STAGEA_TRIALS = 200
STAGEA_STARTUP_TRIALS = 20
STAGEA_PROGRESS_EVERY = 5

# Stage B
TOP_K_STAGEB = 15
FRESH_SGKF_REPEATS = 3
FRESH_SEED_OFFSETS = [101, 1009, 2027]

MODEL_N_JOBS = -1


# =============================================================================
# Broad search-space metadata retained from the frozen five-family procedure
# =============================================================================

@dataclass(frozen=True)
class NumSpec:
    low: float
    high: float
    kind: str = "float"
    log: bool = False
    step: float | None = None
    condition_param: str | None = None
    condition_value: Any | None = None


BROAD_NUM_SPECS: dict[str, dict[str, NumSpec]] = {
    "ET": {
        "n_estimators": NumSpec(300, 1400, "int", step=50),
        "max_depth": NumSpec(3, 30, "int", step=1,
                             condition_param="max_depth_mode", condition_value="finite"),
        "min_samples_split": NumSpec(2, 20, "int", step=1),
        "min_samples_leaf": NumSpec(1, 12, "int", step=1),
    },
    "GB": {
        "n_estimators": NumSpec(100, 800, "int", step=25),
        "learning_rate": NumSpec(0.01, 0.25, "float", log=True),
        "max_depth": NumSpec(1, 5, "int", step=1),
        "min_samples_split": NumSpec(2, 20, "int", step=1),
        "min_samples_leaf": NumSpec(1, 12, "int", step=1),
        "subsample": NumSpec(0.60, 1.00, "float"),
    },
    "XGB": {
        "n_estimators": NumSpec(200, 1200, "int", step=50),
        "learning_rate": NumSpec(0.01, 0.25, "float", log=True),
        "max_depth": NumSpec(2, 9, "int", step=1),
        "min_child_weight": NumSpec(0.5, 15.0, "float", log=True),
        "subsample": NumSpec(0.60, 1.00, "float"),
        "colsample_bytree": NumSpec(0.50, 1.00, "float"),
        "gamma": NumSpec(0.0, 5.0, "float"),
        "reg_alpha": NumSpec(1e-7, 10.0, "float", log=True),
        "reg_lambda": NumSpec(1e-4, 50.0, "float", log=True),
    },
    "LGBM": {
        "n_estimators": NumSpec(200, 1200, "int", step=50),
        "learning_rate": NumSpec(0.01, 0.25, "float", log=True),
        "num_leaves": NumSpec(7, 63, "int", step=1),
        "max_depth": NumSpec(3, 12, "int", step=1,
                             condition_param="max_depth_mode", condition_value="finite"),
        "min_child_samples": NumSpec(5, 50, "int", step=1),
        "colsample_bytree": NumSpec(0.50, 1.00, "float"),
        "reg_alpha": NumSpec(1e-7, 10.0, "float", log=True),
        "reg_lambda": NumSpec(1e-4, 50.0, "float", log=True),
        "subsample": NumSpec(0.60, 0.95, "float",
                             condition_param="bagging_mode", condition_value="on"),
    },
    "CAT": {
        "iterations": NumSpec(200, 1000, "int", step=50),
        "depth": NumSpec(3, 8, "int", step=1),
        "learning_rate": NumSpec(0.01, 0.25, "float", log=True),
        "l2_leaf_reg": NumSpec(1e-3, 30.0, "float", log=True),
        "random_strength": NumSpec(1e-4, 10.0, "float", log=True),
        "bagging_temperature": NumSpec(0.0, 5.0, "float"),
    },
}

BROAD_CATS: dict[str, dict[str, list[Any]]] = {
    "ET": {
        "max_depth_mode": ["none", "finite"],
        "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0],
        "criterion": ["gini", "entropy", "log_loss"],
        "bootstrap": [True, False],
        "max_samples": [None, 0.65, 0.80, 0.95],
    },
    "GB": {
        "max_features": [None, "sqrt", "log2", 0.5, 0.7, 1.0],
    },
    "XGB": {},
    "LGBM": {
        "max_depth_mode": ["none", "finite"],
        "bagging_mode": ["off", "on"],
    },
    "CAT": {
        "border_count": [32, 64, 128],
    },
}

SELECTOR_CHOICES = [50, 65, 80, 100]
WEIGHT_CHOICES = ["none", "balanced"]


# =============================================================================
# Command-line arguments, logging, and ETA
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Five-family targeted Bayesian optimization and stability selection / nested LODO"
    )
    p.add_argument("--input", default=None, help="Path to the original cleaned input workbook")
    p.add_argument("--12a-db", dest="db12a", default=None, help="12A optuna_studies.db")
    p.add_argument("--output-dir", default=None, help="Directory for Stage 12B outputs")
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Smoke test: summer fold 1 and GB, two Stage-A trials, Top-2 candidates; "
            "one fresh grouped-CV repeat. Results are not for publication."
        ),
    )
    return p.parse_args()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def fmt_seconds(seconds):
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "--"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def ascii_bar(done, total, width=30):
    if total <= 0:
        return "[" + "." * width + "]"
    frac = min(max(done / total, 0), 1)
    n = int(frac * width)
    if done >= total:
        return "[" + "=" * width + "]"
    return "[" + "=" * n + ">" + "." * max(width - n - 1, 0) + "]"


class ProgressTracker:
    def __init__(self, total_units: int, initial_done: int = 0, unit_name: str = "units"):
        self.total = int(total_units)
        self.done = int(initial_done)
        self.unit_name = unit_name
        self.start = time.time()
        self.new_durations: list[float] = []

    def add(self, duration: float | None = None):
        self.add_units(1, duration)

    def add_units(self, n: int, duration: float | None = None):
        n = int(n)
        if n <= 0:
            return
        self.done += n
        if duration is not None and np.isfinite(duration) and duration >= 0:
            per_unit = float(duration) / n
            self.new_durations.extend([per_unit] * min(n, 50))

    def eta(self):
        remain = max(self.total - self.done, 0)
        if remain == 0:
            return 0.0
        if not self.new_durations:
            return None
        recent = np.mean(self.new_durations[-20:])
        return float(recent * remain)

    def print(self, prefix="[Overall]"):
        elapsed = time.time() - self.start
        eta = self.eta()
        finish = (
            datetime.now() + timedelta(seconds=float(eta))
            if eta is not None and np.isfinite(eta)
            else None
        )
        pct = 100.0 * self.done / self.total if self.total else 0.0
        print(
            f"{prefix} {ascii_bar(self.done, self.total)} "
            f"{self.done}/{self.total} {self.unit_name} ({pct:.1f}%) "
            f"| Run elapsed {fmt_seconds(elapsed)} "
            f"| ETA {fmt_seconds(eta)} "
            f"| Estimated completion {finish.strftime('%Y-%m-%d %H:%M:%S') if finish else '--'}"
        )


class StageATrialCallback:
    def __init__(self, tracker, local_total, label, print_every, grand_tracker=None):
        self.tracker = tracker
        self.grand_tracker = grand_tracker
        self.local_total = int(local_total)
        self.label = label
        self.print_every = max(int(print_every), 1)
        self.local_done = 0
        self.start = time.time()
        self.last = time.time()

    def __call__(self, study, trial):
        now = time.time()
        duration = now - self.last
        self.last = now
        self.local_done += 1
        self.tracker.add(duration)
        if self.grand_tracker is not None:
            self.grand_tracker.add_units(INNER_SPLITS, duration)

        if self.local_done % self.print_every == 0 or self.local_done == self.local_total:
            complete = [
                t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
                and t.value is not None
                and np.isfinite(float(t.value))
            ]
            best = max([float(t.value) for t in complete], default=np.nan)
            elapsed = now - self.start
            local_eta = (elapsed / max(self.local_done, 1)) * max(self.local_total - self.local_done, 0)
            if np.isfinite(best):
                print(
                    f"\n  [Stage A Study] {self.label}\n"
                    f"  {ascii_bar(self.local_done, self.local_total)} "
                    f"{self.local_done}/{self.local_total} new trials "
                    f"| best inner Macro-F1={best:.4f}"
                )
            else:
                print(
                    f"\n  [Stage A Study] {self.label}\n"
                    f"  {ascii_bar(self.local_done, self.local_total)} "
                    f"{self.local_done}/{self.local_total} new trials | best inner Macro-F1=--"
                )
            print(f"  Elapsed in study {fmt_seconds(elapsed)} | Study ETA {fmt_seconds(local_eta)}")
            self.tracker.print(prefix="  [Stage A Overall]")
            if self.grand_tracker is not None:
                self.grand_tracker.print(prefix="  [Full Workflow - Fit Equivalent]")


# =============================================================================
# Data loading and strict consistency checks
# =============================================================================

def resolve_paths(args):
    input_file = Path(args.input) if args.input else DEFAULT_INPUT
    db12a = Path(args.db12a) if args.db12a else DEFAULT_12A_DB

    if not input_file.exists():
        raise FileNotFoundError(
            f"Stage 12B input workbook not found:\n{input_file}\n"
            "Default input is the original cleaned analysis workbook."
        )
    if not db12a.exists():
        raise FileNotFoundError(
            f"Stage 12A Optuna database not found:\n{db12a}\n"
            "Run Stage 12A first, or pass its Optuna database using --12a-db."
        )

    if args.output_dir:
        requested = Path(args.output_dir)
        run_dir = (
            requested.parent / f"{requested.name}_SMOKE_TEST"
            if args.smoke_test else requested
        )
    else:
        suffix = "_SMOKE_TEST" if args.smoke_test else ""
        run_dir = input_file.parent / f"{RUN_DIR_NAME}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return input_file, db12a, run_dir


def load_data(input_file: Path):
    raw = pd.read_excel(
        input_file,
        sheet_name=list(SHEET_MAP.values()),
        engine="openpyxl",
    )
    required = ["Date", "measurement_group", TARGET] + EP_FEATURES
    out = {}

    for season, sheet in SHEET_MAP.items():
        df = raw[sheet].copy()
        df.columns = [str(c).strip() for c in df.columns]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"{season} is missing columns: {missing}")
        if len(df) != EXPECTED_N[season]:
            raise ValueError(f"{season}: N={len(df)}, expected={EXPECTED_N[season]}")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        for c in EP_NUMERIC:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in EP_CATEGORICAL:
            df[c] = df[c].astype(str).str.strip()
        df["measurement_group"] = df["measurement_group"].astype(str).str.strip()

        if df[required].isna().any().any():
            bad = df[required].isna().sum()
            raise ValueError(f"{season} contains missing or invalid values: {bad[bad > 0].to_dict()}")

        df[TARGET] = df[TARGET].astype(int)
        if set(df[TARGET].unique()) != {1, 2, 3}:
            raise ValueError(f"{season}: target must include all three classes (1/2/3)")
        if df["Date"].nunique() != 4:
            raise ValueError(f"{season}: expected exactly four survey days")
        if df["measurement_group"].nunique() != EXPECTED_GROUPS[season]:
            raise ValueError(
                f"{season}: measurement_group={df['measurement_group'].nunique()},"
                f"expected={EXPECTED_GROUPS[season]}"
            )
        cross = df.groupby("measurement_group")["Date"].nunique().gt(1).sum()
        if cross:
            raise ValueError(f"{season}: {cross} measurement groups span multiple survey dates")

        df["y_model"] = df[TARGET].map(RAW_TO_MODEL).astype(int)
        for d, g in df.groupby("Date", sort=True):
            if set(g["y_model"].unique()) != {0, 1, 2}:
                raise ValueError(f"{season} {d.date()}: day lacks one or more outcome classes")

        df = df.reset_index(drop=True)
        df["row_id"] = np.arange(len(df))
        out[season] = df
    return out


# =============================================================================
# Preprocessing, estimators, and metrics (unchanged frozen workflow)
# =============================================================================

def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float64)


def make_preprocessor():
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_onehot()),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, EP_NUMERIC),
            ("cat", cat_pipe, EP_CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


class CatBoostSklearnAdapter(ClassifierMixin, BaseEstimator):
    def __init__(self, params=None):
        self.params = {} if params is None else params

    def fit(self, X, y, sample_weight=None):
        self.model_ = CatBoostClassifier(**dict(self.params))
        self.model_.fit(X, y, sample_weight=sample_weight)
        self.classes_ = np.asarray(getattr(self.model_, "classes_", np.unique(y)))
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X)).reshape(-1).astype(int)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def decode_config(raw_params: dict, model_name: str):
    raw = dict(raw_params)
    selector = int(raw.pop("selector_percentile"))
    weight_mode = str(raw.pop("weight_mode"))

    if model_name == "ET":
        mode = raw.pop("max_depth_mode", "finite")
        if mode == "none":
            raw["max_depth"] = None
        if not bool(raw.get("bootstrap", False)):
            raw["max_samples"] = None

    if model_name == "LGBM":
        mode = raw.pop("max_depth_mode", "finite")
        if mode == "none":
            raw["max_depth"] = -1
        bag = raw.pop("bagging_mode", "off")
        if bag == "off":
            raw["subsample"] = 1.0
            raw["subsample_freq"] = 0
        else:
            raw["subsample_freq"] = 1

    return selector, weight_mode, raw


def make_estimator(model_name: str, params: dict, seed: int):
    p = dict(params)

    if model_name == "ET":
        return ExtraTreesClassifier(
            n_estimators=int(p["n_estimators"]),
            max_depth=p.get("max_depth"),
            min_samples_split=int(p["min_samples_split"]),
            min_samples_leaf=int(p["min_samples_leaf"]),
            max_features=p["max_features"],
            criterion=p["criterion"],
            bootstrap=bool(p["bootstrap"]),
            max_samples=p.get("max_samples"),
            random_state=seed,
            n_jobs=MODEL_N_JOBS,
        )

    if model_name == "GB":
        return GradientBoostingClassifier(
            n_estimators=int(p["n_estimators"]),
            learning_rate=float(p["learning_rate"]),
            max_depth=int(p["max_depth"]),
            min_samples_split=int(p["min_samples_split"]),
            min_samples_leaf=int(p["min_samples_leaf"]),
            subsample=float(p["subsample"]),
            max_features=p["max_features"],
            random_state=seed,
        )

    if model_name == "XGB":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_estimators=int(p["n_estimators"]),
            learning_rate=float(p["learning_rate"]),
            max_depth=int(p["max_depth"]),
            min_child_weight=float(p["min_child_weight"]),
            subsample=float(p["subsample"]),
            colsample_bytree=float(p["colsample_bytree"]),
            gamma=float(p["gamma"]),
            reg_alpha=float(p["reg_alpha"]),
            reg_lambda=float(p["reg_lambda"]),
            random_state=seed,
            n_jobs=MODEL_N_JOBS,
            tree_method="hist",
        )

    if model_name == "LGBM":
        return LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=int(p["n_estimators"]),
            learning_rate=float(p["learning_rate"]),
            num_leaves=int(p["num_leaves"]),
            max_depth=int(p["max_depth"]),
            min_child_samples=int(p["min_child_samples"]),
            subsample=float(p["subsample"]),
            subsample_freq=int(p["subsample_freq"]),
            colsample_bytree=float(p["colsample_bytree"]),
            reg_alpha=float(p["reg_alpha"]),
            reg_lambda=float(p["reg_lambda"]),
            random_state=seed,
            n_jobs=MODEL_N_JOBS,
            verbosity=-1,
        )

    if model_name == "CAT":
        cat_params = dict(
            loss_function="MultiClass",
            iterations=int(p["iterations"]),
            learning_rate=float(p["learning_rate"]),
            depth=int(p["depth"]),
            l2_leaf_reg=float(p["l2_leaf_reg"]),
            random_strength=float(p["random_strength"]),
            bagging_temperature=float(p["bagging_temperature"]),
            border_count=int(p["border_count"]),
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=MODEL_N_JOBS,
        )
        return CatBoostSklearnAdapter(params=cat_params)

    raise KeyError(model_name)


def make_pipeline(model_name: str, raw_config: dict, seed: int):
    selector, _, model_params = decode_config(raw_config, model_name)
    return Pipeline([
        ("preprocess", make_preprocessor()),
        ("selector", SelectPercentile(score_func=f_classif, percentile=selector)),
        ("model", make_estimator(model_name, model_params, seed)),
    ])


def fit_pipeline(pipe, X, y, weight_mode: str):
    kwargs = {}
    if weight_mode == "balanced":
        kwargs["model__sample_weight"] = compute_sample_weight("balanced", y=y)
    pipe.fit(X, y, **kwargs)
    return pipe


def metrics_bundle(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_ORDER, zero_division=0
    )
    out = {
        "Macro_F1": float(f1_score(y_true, y_pred, labels=CLASS_ORDER,
                                   average="macro", zero_division=0)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Kappa": float(cohen_kappa_score(y_true, y_pred, labels=CLASS_ORDER)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
    }
    for i, cls in enumerate(CLASS_ORDER):
        raw_cls = MODEL_TO_RAW[cls]
        out[f"Class{raw_cls}_Precision"] = float(precision[i])
        out[f"Class{raw_cls}_Recall"] = float(recall[i])
        out[f"Class{raw_cls}_F1"] = float(f1[i])
        out[f"Class{raw_cls}_Support"] = int(support[i])
    out["Worst_Class_F1"] = float(np.min(f1))
    return out


def build_sgkf_splits(train_df: pd.DataFrame, seed: int):
    cv = StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    y = train_df["y_model"].to_numpy()
    groups = train_df["measurement_group"].astype(str).to_numpy()
    splits = list(cv.split(np.zeros((len(train_df), 1)), y, groups))
    for k, (tr, va) in enumerate(splits, start=1):
        if set(y[tr]) != {0, 1, 2} or set(y[va]) != {0, 1, 2}:
            raise RuntimeError(f"SGKF seed={seed} fold{k}: training or validation set lacks one or more classes")
        if set(groups[tr]) & set(groups[va]):
            raise RuntimeError(f"SGKF seed={seed} fold{k}: group overlap")
    return splits


# =============================================================================
# Read 12A history and construct fold-local targeted search spaces
# =============================================================================

def sqlite_url(path: Path):
    return f"sqlite:///{path.as_posix()}"


def study_name_12a(season, outer_fold, held_date, model_name):
    tag = pd.Timestamp(held_date).strftime("%Y%m%d")
    return f"12A__{season}__fold{outer_fold}__{tag}__EP__{model_name}"


def load_12a_history(db12a: Path, season, outer_fold, held_date, model_name):
    name = study_name_12a(season, outer_fold, held_date, model_name)
    try:
        study = optuna.load_study(study_name=name, storage=sqlite_url(db12a))
    except Exception as exc:
        raise RuntimeError(f"Cannot load Stage 12A study: {name}\n{exc}") from exc

    complete = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and np.isfinite(float(t.value))
    ]
    complete = sorted(complete, key=lambda t: t.number)
    if len(complete) < HISTORY_COMPLETE_TRIALS:
        raise RuntimeError(
            f"{name}: COMPLETE trials={len(complete)}, fewer than the required {HISTORY_COMPLETE_TRIALS}."
            "Verify that the complete Stage 12A seven-family history is available."
        )

    history80 = complete[:HISTORY_COMPLETE_TRIALS]
    elite = sorted(history80, key=lambda t: float(t.value), reverse=True)[:ELITE_N]
    return name, history80, elite


def _transform_value(v: float, log: bool):
    return math.log(float(v)) if log else float(v)


def _inverse_value(v: float, log: bool):
    return math.exp(float(v)) if log else float(v)


def _snap_int(value: float, spec: NumSpec, direction: str):
    step = int(spec.step or 1)
    low = int(round(spec.low))
    if direction == "down":
        k = math.floor((value - low) / step)
    else:
        k = math.ceil((value - low) / step)
    ans = low + k * step
    return int(min(max(ans, int(spec.low)), int(spec.high)))


def targeted_numeric_bounds(values: list[float], spec: NumSpec):
    if len(values) < MIN_ACTIVE_ELITE:
        return spec.low, spec.high, "broad_fallback_insufficient_active_elite"

    transformed = np.array([_transform_value(v, spec.log) for v in values], dtype=float)
    broad_lo = _transform_value(spec.low, spec.log)
    broad_hi = _transform_value(spec.high, spec.log)
    broad_span = broad_hi - broad_lo

    qlo = float(np.quantile(transformed, ELITE_Q_LOW))
    qhi = float(np.quantile(transformed, ELITE_Q_HIGH))
    med = float(np.median(transformed))

    lo = qlo - TARGET_PADDING_FRAC * broad_span
    hi = qhi + TARGET_PADDING_FRAC * broad_span

    min_width = TARGET_MIN_WIDTH_FRAC * broad_span
    if hi - lo < min_width:
        lo = med - min_width / 2
        hi = med + min_width / 2

    lo = max(lo, broad_lo)
    hi = min(hi, broad_hi)
    if hi - lo < min_width:
        if abs(lo - broad_lo) < 1e-12:
            hi = min(broad_hi, lo + min_width)
        elif abs(hi - broad_hi) < 1e-12:
            lo = max(broad_lo, hi - min_width)

    lo_raw = _inverse_value(lo, spec.log)
    hi_raw = _inverse_value(hi, spec.log)

    if spec.kind == "int":
        lo_raw = _snap_int(lo_raw, spec, "down")
        hi_raw = _snap_int(hi_raw, spec, "up")
        if lo_raw > hi_raw:
            lo_raw, hi_raw = int(spec.low), int(spec.high)
    else:
        lo_raw = max(float(spec.low), float(lo_raw))
        hi_raw = min(float(spec.high), float(hi_raw))

    return lo_raw, hi_raw, "elite_q10_q90+10pct_broad_padding+30pct_minwidth"


def derive_targeted_space(model_name: str, elite_trials):
    space = {
        "selector_percentile": {"type": "categorical", "choices": SELECTOR_CHOICES},
        "weight_mode": {"type": "categorical", "choices": WEIGHT_CHOICES},
        "model": model_name,
        "numeric": {},
        "categorical": BROAD_CATS[model_name],
    }

    for param, spec in BROAD_NUM_SPECS[model_name].items():
        values = []
        for t in elite_trials:
            if spec.condition_param is not None:
                if t.params.get(spec.condition_param) != spec.condition_value:
                    continue
            if param in t.params:
                try:
                    values.append(float(t.params[param]))
                except Exception:
                    pass

        lo, hi, rule = targeted_numeric_bounds(values, spec)
        space["numeric"][param] = {
            "low": lo,
            "high": hi,
            "kind": spec.kind,
            "log": spec.log,
            "step": spec.step,
            "condition_param": spec.condition_param,
            "condition_value": spec.condition_value,
            "elite_active_n": len(values),
            "broad_low": spec.low,
            "broad_high": spec.high,
            "rule": rule,
        }
    return space


def derive_all_targeted_spaces(data, db12a: Path, models, seasons, folds):
    spaces = {}
    audit_rows = []

    for season in seasons:
        dates = sorted(pd.unique(data[season]["Date"]))
        for outer_fold in folds:
            held_date = dates[outer_fold - 1]
            for model_name in models:
                src_name, _, elite = load_12a_history(
                    db12a, season, outer_fold, held_date, model_name
                )
                space = derive_targeted_space(model_name, elite)
                key = f"{season}__F{outer_fold}__{model_name}"
                spaces[key] = {
                    "season": season,
                    "outer_fold": outer_fold,
                    "held_out_date": pd.Timestamp(held_date).strftime("%Y-%m-%d"),
                    "model": model_name,
                    "source_12a_study": src_name,
                    "space": space,
                }

                for p, info in space["numeric"].items():
                    audit_rows.append({
                        "Season": season,
                        "Outer_fold": outer_fold,
                        "Held_out_date": pd.Timestamp(held_date),
                        "Model": model_name,
                        "Parameter": p,
                        "Broad_low": info["broad_low"],
                        "Broad_high": info["broad_high"],
                        "Target_low": info["low"],
                        "Target_high": info["high"],
                        "Log_scale": info["log"],
                        "Step": info["step"],
                        "Elite_active_n": info["elite_active_n"],
                        "Rule": info["rule"],
                        "Source_12A_study": src_name,
                    })

                for p, choices in space["categorical"].items():
                    audit_rows.append({
                        "Season": season,
                        "Outer_fold": outer_fold,
                        "Held_out_date": pd.Timestamp(held_date),
                        "Model": model_name,
                        "Parameter": p,
                        "Broad_low": np.nan,
                        "Broad_high": np.nan,
                        "Target_low": np.nan,
                        "Target_high": np.nan,
                        "Log_scale": False,
                        "Step": np.nan,
                        "Elite_active_n": np.nan,
                        "Rule": "categorical_keep_all_broad_choices=" + json.dumps(choices, ensure_ascii=False),
                        "Source_12A_study": src_name,
                    })
    return spaces, pd.DataFrame(audit_rows)


# =============================================================================
# 7. Targeted Optuna suggest
# =============================================================================

def _suggest_num(trial, name: str, info: dict):
    if info["kind"] == "int":
        return trial.suggest_int(
            name,
            int(info["low"]),
            int(info["high"]),
            step=int(info["step"] or 1),
            log=False,
        )
    return trial.suggest_float(
        name,
        float(info["low"]),
        float(info["high"]),
        log=bool(info["log"]),
    )


def suggest_targeted_config(trial, model_name: str, space: dict):
    trial.suggest_categorical("selector_percentile", SELECTOR_CHOICES)
    trial.suggest_categorical("weight_mode", WEIGHT_CHOICES)
    num = space["numeric"]

    if model_name == "ET":
        _suggest_num(trial, "n_estimators", num["n_estimators"])
        md_mode = trial.suggest_categorical("max_depth_mode", BROAD_CATS["ET"]["max_depth_mode"])
        if md_mode == "finite":
            _suggest_num(trial, "max_depth", num["max_depth"])
        _suggest_num(trial, "min_samples_split", num["min_samples_split"])
        _suggest_num(trial, "min_samples_leaf", num["min_samples_leaf"])
        trial.suggest_categorical("max_features", BROAD_CATS["ET"]["max_features"])
        trial.suggest_categorical("criterion", BROAD_CATS["ET"]["criterion"])
        bootstrap = trial.suggest_categorical("bootstrap", BROAD_CATS["ET"]["bootstrap"])
        if bootstrap:
            trial.suggest_categorical("max_samples", BROAD_CATS["ET"]["max_samples"])

    elif model_name == "GB":
        _suggest_num(trial, "n_estimators", num["n_estimators"])
        _suggest_num(trial, "learning_rate", num["learning_rate"])
        _suggest_num(trial, "max_depth", num["max_depth"])
        _suggest_num(trial, "min_samples_split", num["min_samples_split"])
        _suggest_num(trial, "min_samples_leaf", num["min_samples_leaf"])
        _suggest_num(trial, "subsample", num["subsample"])
        trial.suggest_categorical("max_features", BROAD_CATS["GB"]["max_features"])

    elif model_name == "XGB":
        for p in [
            "n_estimators", "learning_rate", "max_depth", "min_child_weight",
            "subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda"
        ]:
            _suggest_num(trial, p, num[p])

    elif model_name == "LGBM":
        _suggest_num(trial, "n_estimators", num["n_estimators"])
        _suggest_num(trial, "learning_rate", num["learning_rate"])
        _suggest_num(trial, "num_leaves", num["num_leaves"])
        md_mode = trial.suggest_categorical("max_depth_mode", BROAD_CATS["LGBM"]["max_depth_mode"])
        if md_mode == "finite":
            _suggest_num(trial, "max_depth", num["max_depth"])
        _suggest_num(trial, "min_child_samples", num["min_child_samples"])
        _suggest_num(trial, "colsample_bytree", num["colsample_bytree"])
        _suggest_num(trial, "reg_alpha", num["reg_alpha"])
        _suggest_num(trial, "reg_lambda", num["reg_lambda"])
        bag = trial.suggest_categorical("bagging_mode", BROAD_CATS["LGBM"]["bagging_mode"])
        if bag == "on":
            _suggest_num(trial, "subsample", num["subsample"])

    elif model_name == "CAT":
        for p in [
            "iterations", "depth", "learning_rate", "l2_leaf_reg",
            "random_strength", "bagging_temperature"
        ]:
            _suggest_num(trial, p, num[p])
        trial.suggest_categorical("border_count", BROAD_CATS["CAT"]["border_count"])

    else:
        raise KeyError(model_name)

    return dict(trial.params)


# =============================================================================
# 8. Stage A:Targeted Bayesian
# =============================================================================

def stagea_split_seed(season, outer_fold):
    return RANDOM_STATE + (0 if season == "Summer" else 100000) + outer_fold * 1000


def stagea_study_seed(season, outer_fold, model_name):
    return (
        RANDOM_STATE
        + 10_000_000
        + (0 if season == "Summer" else 200000)
        + outer_fold * 1000
        + CANDIDATE_MODELS.index(model_name) * 17
    )


def model_fit_seed(season, outer_fold, model_name, phase_offset=0):
    return (
        RANDOM_STATE
        + 20_000_000
        + phase_offset
        + (0 if season == "Summer" else 300000)
        + outer_fold * 1000
        + CANDIDATE_MODELS.index(model_name) * 31
    )


def stagea_study_name(season, outer_fold, held_date, model_name):
    tag = pd.Timestamp(held_date).strftime("%Y%m%d")
    return f"12B_A__{season}__fold{outer_fold}__{tag}__EP__{model_name}"


def finished_trial_count(study):
    # For resuming, COMPLETE/PRUNED/FAIL states all count against the trial budget.
    return sum(
        t.state in {
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
        }
        for t in study.trials
    )


def make_stagea_objective(outer_train, model_name, space, inner_splits, fit_seed):
    X = outer_train[EP_FEATURES].reset_index(drop=True)
    y = outer_train["y_model"].to_numpy()

    def objective(trial):
        raw = suggest_targeted_config(trial, model_name, space)
        _, weight_mode, _ = decode_config(raw, model_name)

        fold_macro, fold_bal, fold_kap, fold_worst = [], [], [], []
        for tr, va in inner_splits:
            pipe = make_pipeline(model_name, raw, fit_seed)
            fit_pipeline(pipe, X.iloc[tr], y[tr], weight_mode)
            pred = pipe.predict(X.iloc[va]).astype(int)
            m = metrics_bundle(y[va], pred)
            fold_macro.append(m["Macro_F1"])
            fold_bal.append(m["Balanced_Accuracy"])
            fold_kap.append(m["Kappa"])
            fold_worst.append(m["Worst_Class_F1"])
            del pipe
            gc.collect()

        trial.set_user_attr("inner_macro_f1_sd", float(np.std(fold_macro, ddof=1)))
        trial.set_user_attr("inner_balacc_mean", float(np.mean(fold_bal)))
        trial.set_user_attr("inner_kappa_mean", float(np.mean(fold_kap)))
        trial.set_user_attr("inner_worst_class_f1_mean", float(np.mean(fold_worst)))
        return float(np.mean(fold_macro))

    return objective


def stagea_existing_finished(stagea_db, plans, trials_per_study):
    total = 0
    storage = sqlite_url(stagea_db)
    for p in plans:
        name = stagea_study_name(p["season"], p["outer_fold"], p["held_date"], p["model"])
        try:
            study = optuna.load_study(study_name=name, storage=storage)
            total += min(finished_trial_count(study), trials_per_study)
        except Exception:
            pass
    return total


def run_stagea(data, spaces, run_dir, stagea_db, models, seasons, folds,
               trials_per_study, smoke_test=False, grand_tracker=None):
    plans = []
    for season in seasons:
        dates = sorted(pd.unique(data[season]["Date"]))
        for outer_fold in folds:
            held_date = dates[outer_fold - 1]
            for model in models:
                plans.append({
                    "season": season,
                    "outer_fold": outer_fold,
                    "held_date": held_date,
                    "model": model,
                })

    total_trials = len(plans) * trials_per_study
    existing = stagea_existing_finished(stagea_db, plans, trials_per_study)
    tracker = ProgressTracker(total_trials, existing, "trials")
    storage = sqlite_url(stagea_db)
    trial_dir = run_dir / "stageA_trials"
    trial_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 100)
    print("12B Stage A:Targeted Bayesian Refinement")
    print("=" * 100)
    print(f"Studies={len(plans)} | trials/study={trials_per_study} | total trials={total_trials} | resume={existing}")
    if not smoke_test:
        print("Formal budget: 40 studies x 200 trials, each with three inner folds.")
    print("=" * 100)

    for idx, p in enumerate(plans, start=1):
        season, outer_fold, held_date, model = p["season"], p["outer_fold"], p["held_date"], p["model"]
        outer_train = data[season].loc[data[season]["Date"] != held_date].copy().reset_index(drop=True)
        split_seed = stagea_split_seed(season, outer_fold)
        splits = build_sgkf_splits(outer_train, split_seed)
        key = f"{season}__F{outer_fold}__{model}"
        space = spaces[key]["space"]

        study_name = stagea_study_name(season, outer_fold, held_date, model)
        sampler = TPESampler(
            seed=stagea_study_seed(season, outer_fold, model),
            n_startup_trials=min(STAGEA_STARTUP_TRIALS, max(trials_per_study // 2, 1)),
            multivariate=True,
        )
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            sampler=sampler,
            pruner=NopPruner(),
            load_if_exists=True,
        )
        finished = finished_trial_count(study)
        remaining = max(trials_per_study - finished, 0)

        print("\n" + "-" * 100)
        print(
            f"[Stage A {idx}/{len(plans)}] {season} | F{outer_fold} | held={pd.Timestamp(held_date).date()} "
            f"| {model} | target={trials_per_study} | Existing trials={finished} | Remaining={remaining}"
        )

        if remaining > 0:
            objective = make_stagea_objective(
                outer_train=outer_train,
                model_name=model,
                space=space,
                inner_splits=splits,
                fit_seed=model_fit_seed(season, outer_fold, model, phase_offset=100000),
            )
            cb = StageATrialCallback(
                tracker=tracker,
                local_total=remaining,
                label=f"{season}/F{outer_fold}/{model}",
                print_every=1 if smoke_test else STAGEA_PROGRESS_EVERY,
                grand_tracker=grand_tracker,
            )
            study.optimize(
                objective,
                n_trials=remaining,
                callbacks=[cb],
                gc_after_trial=True,
                show_progress_bar=False,
                catch=(ValueError, FloatingPointError),
            )

        study.trials_dataframe(
            attrs=("number", "value", "datetime_start", "datetime_complete", "duration", "params", "user_attrs", "state")
        ).to_csv(trial_dir / f"{study_name}.csv", index=False, encoding="utf-8-sig")

    tracker.print(prefix="[Stage A Complete]")
    return plans


# =============================================================================
# 9. Stage A Top-K candidate
# =============================================================================

def canonical_params(params: dict):
    return json.dumps(params, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def candidate_key(study_name: str, trial_number: int, params: dict):
    raw = f"{study_name}|{trial_number}|{canonical_params(params)}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def extract_stageb_candidates(stagea_db: Path, plans, top_k: int, trials_per_study: int):
    storage = sqlite_url(stagea_db)
    rows = []
    candidates = []

    for p in plans:
        name = stagea_study_name(p["season"], p["outer_fold"], p["held_date"], p["model"])
        study = optuna.load_study(study_name=name, storage=storage)
        complete = [
            t for t in study.trials
            if int(t.number) < int(trials_per_study)
            and t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None
            and np.isfinite(float(t.value))
        ]
        complete = sorted(complete, key=lambda t: t.number)
        if len(complete) < top_k:
            raise RuntimeError(f"{name}: COMPLETE trials={len(complete)} < TopK={top_k}")

        ranked = sorted(complete, key=lambda t: (-float(t.value), int(t.number)))
        seen = set()
        selected = []
        for t in ranked:
            sig = canonical_params(dict(t.params))
            if sig in seen:
                continue
            seen.add(sig)
            selected.append(t)
            if len(selected) >= top_k:
                break
        if len(selected) < top_k:
            raise RuntimeError(f"{name}: fewer than {top_k} unique candidates after removing exact duplicates")

        for rank, t in enumerate(selected, start=1):
            params = dict(t.params)
            ck = candidate_key(name, int(t.number), params)
            rec = {
                "candidate_key": ck,
                "study_name": name,
                "Season": p["season"],
                "Outer_fold": p["outer_fold"],
                "Held_out_date": pd.Timestamp(p["held_date"]),
                "Model": p["model"],
                "StageA_rank": rank,
                "StageA_trial_number": int(t.number),
                "StageA_inner_Macro_F1": float(t.value),
                "Params": params,
            }
            candidates.append(rec)
            rows.append({**{k: v for k, v in rec.items() if k != "Params"}, "Params_json": canonical_params(params)})
    return candidates, pd.DataFrame(rows)


# =============================================================================
# 10. Stage B SQLite checkpoint
# =============================================================================

def init_checkpoint_db(path: Path):
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS stageb_results (
            candidate_key TEXT PRIMARY KEY,
            season TEXT NOT NULL,
            outer_fold INTEGER NOT NULL,
            held_out_date TEXT NOT NULL,
            model TEXT NOT NULL,
            stagea_rank INTEGER NOT NULL,
            stagea_trial_number INTEGER NOT NULL,
            stagea_inner_macro_f1 REAL NOT NULL,
            params_json TEXT NOT NULL,
            result_json TEXT NOT NULL,
            completed_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS outer_fold_results (
            fold_key TEXT PRIMARY KEY,
            season TEXT NOT NULL,
            outer_fold INTEGER NOT NULL,
            held_out_date TEXT NOT NULL,
            selected_model TEXT NOT NULL,
            candidate_key TEXT NOT NULL,
            params_json TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            completed_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS outer_predictions (
            fold_key TEXT NOT NULL,
            row_id INTEGER NOT NULL,
            y_true INTEGER NOT NULL,
            y_pred INTEGER NOT NULL,
            selected_model TEXT NOT NULL,
            PRIMARY KEY (fold_key, row_id)
        )
    """)
    con.commit()
    con.close()


def load_stageb_checkpoint(path: Path):
    con = sqlite3.connect(path)
    try:
        df = pd.read_sql_query("SELECT * FROM stageb_results", con)
    finally:
        con.close()
    if df.empty:
        return {}
    out = {}
    for _, r in df.iterrows():
        d = json.loads(r["result_json"])
        d.update({
            "candidate_key": r["candidate_key"],
            "Season": r["season"],
            "Outer_fold": int(r["outer_fold"]),
            "Held_out_date": pd.Timestamp(r["held_out_date"]),
            "Model": r["model"],
            "StageA_rank": int(r["stagea_rank"]),
            "StageA_trial_number": int(r["stagea_trial_number"]),
            "StageA_inner_Macro_F1": float(r["stagea_inner_macro_f1"]),
            "Params": json.loads(r["params_json"]),
        })
        out[r["candidate_key"]] = d
    return out


def save_stageb_checkpoint(path: Path, candidate: dict, result: dict):
    con = sqlite3.connect(path)
    try:
        con.execute(
            """
            INSERT OR REPLACE INTO stageb_results
            (candidate_key, season, outer_fold, held_out_date, model, stagea_rank,
             stagea_trial_number, stagea_inner_macro_f1, params_json, result_json, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate["candidate_key"], candidate["Season"], int(candidate["Outer_fold"]),
                pd.Timestamp(candidate["Held_out_date"]).strftime("%Y-%m-%d"), candidate["Model"],
                int(candidate["StageA_rank"]), int(candidate["StageA_trial_number"]),
                float(candidate["StageA_inner_Macro_F1"]), canonical_params(candidate["Params"]),
                json.dumps(result, ensure_ascii=False, sort_keys=True),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        con.commit()
    finally:
        con.close()


# =============================================================================
# 11. Stage B:fresh SGKF + inner-day LODO
# =============================================================================

def fresh_split_seeds(season, outer_fold, repeats):
    base = (
        RANDOM_STATE
        + 30_000_000
        + (0 if season == "Summer" else 400000)
        + outer_fold * 1000
    )
    offsets = FRESH_SEED_OFFSETS[:repeats]
    return [base + x for x in offsets]


def evaluate_candidate_stageb(outer_train: pd.DataFrame, candidate: dict, fresh_repeats: int):
    season = candidate["Season"]
    outer_fold = int(candidate["Outer_fold"])
    model = candidate["Model"]
    raw = candidate["Params"]
    _, weight_mode, _ = decode_config(raw, model)
    X = outer_train[EP_FEATURES].reset_index(drop=True)
    y = outer_train["y_model"].to_numpy()

    fit_seed = model_fit_seed(season, outer_fold, model, phase_offset=500000)

    # Fresh grouped CV: assemble OOF predictions for each seed before evaluating metrics.
    sgkf_metrics = []
    for seed in fresh_split_seeds(season, outer_fold, fresh_repeats):
        splits = build_sgkf_splits(outer_train, seed)
        oof = np.full(len(outer_train), -1, dtype=int)
        for tr, va in splits:
            pipe = make_pipeline(model, raw, fit_seed)
            fit_pipeline(pipe, X.iloc[tr], y[tr], weight_mode)
            oof[va] = pipe.predict(X.iloc[va]).astype(int)
            del pipe
            gc.collect()
        if np.any(oof < 0):
            raise RuntimeError("Fresh grouped-CV OOF predictions did not cover every training observation.")
        sgkf_metrics.append(metrics_bundle(y, oof))

    sg_macro = np.array([m["Macro_F1"] for m in sgkf_metrics], dtype=float)
    sg_bal = np.array([m["Balanced_Accuracy"] for m in sgkf_metrics], dtype=float)
    sg_kap = np.array([m["Kappa"] for m in sgkf_metrics], dtype=float)
    sg_worst = np.array([m["Worst_Class_F1"] for m in sgkf_metrics], dtype=float)

    # Inner-day LODO: combine predictions from each of the three outer-training days.
    dates = sorted(pd.unique(outer_train["Date"]))
    if len(dates) != 3:
        raise RuntimeError(f"Stage B inner-day LODO requires exactly three outer-training days; found={len(dates)}")

    day_oof = np.full(len(outer_train), -1, dtype=int)
    day_fold_metrics = []
    day_fold_rows = []
    for inner_day_fold, d in enumerate(dates, start=1):
        tr_mask = outer_train["Date"] != d
        va_mask = outer_train["Date"] == d
        tr_idx = np.where(tr_mask.to_numpy())[0]
        va_idx = np.where(va_mask.to_numpy())[0]
        if set(y[tr_idx]) != {0, 1, 2} or set(y[va_idx]) != {0, 1, 2}:
            raise RuntimeError(f"inner-day LODO {d}: training or validation set lacks one or more classes")
        pipe = make_pipeline(model, raw, fit_seed)
        fit_pipeline(pipe, X.iloc[tr_idx], y[tr_idx], weight_mode)
        pred = pipe.predict(X.iloc[va_idx]).astype(int)
        day_oof[va_idx] = pred
        mm = metrics_bundle(y[va_idx], pred)
        day_fold_metrics.append(mm)
        day_fold_rows.append({
            "Inner_day_fold": inner_day_fold,
            "Held_out_inner_date": pd.Timestamp(d).strftime("%Y-%m-%d"),
            "Macro_F1": mm["Macro_F1"],
            "Worst_Class_F1": mm["Worst_Class_F1"],
        })
        del pipe
        gc.collect()

    if np.any(day_oof < 0):
        raise RuntimeError("Inner-day LODO OOF predictions did not cover every outer-training observation.")
    day_oof_m = metrics_bundle(y, day_oof)
    day_macro = np.array([m["Macro_F1"] for m in day_fold_metrics], dtype=float)

    fresh_macro_mean = float(np.mean(sg_macro))
    day_macro_oof = float(day_oof_m["Macro_F1"])

    robust_min = min(fresh_macro_mean, day_macro_oof)
    robust_mean = (fresh_macro_mean + day_macro_oof) / 2.0
    robust_worst_class = min(float(np.mean(sg_worst)), float(day_oof_m["Worst_Class_F1"]))
    worst_day = float(np.min(day_macro))
    sg_sd = float(np.std(sg_macro, ddof=1)) if len(sg_macro) > 1 else 0.0
    day_sd = float(np.std(day_macro, ddof=1)) if len(day_macro) > 1 else 0.0
    stability_max_sd = max(sg_sd, day_sd)
    robust_bal = (float(np.mean(sg_bal)) + float(day_oof_m["Balanced_Accuracy"])) / 2.0
    robust_kap = (float(np.mean(sg_kap)) + float(day_oof_m["Kappa"])) / 2.0

    result = {
        "FreshSGKF_MacroF1_mean": fresh_macro_mean,
        "FreshSGKF_MacroF1_SD": sg_sd,
        "FreshSGKF_BalAcc_mean": float(np.mean(sg_bal)),
        "FreshSGKF_Kappa_mean": float(np.mean(sg_kap)),
        "FreshSGKF_WorstClassF1_mean": float(np.mean(sg_worst)),
        "FreshSGKF_seed_MacroF1": [float(x) for x in sg_macro],
        "InnerDayLODO_OOF_MacroF1": day_macro_oof,
        "InnerDayLODO_OOF_BalAcc": float(day_oof_m["Balanced_Accuracy"]),
        "InnerDayLODO_OOF_Kappa": float(day_oof_m["Kappa"]),
        "InnerDayLODO_OOF_WorstClassF1": float(day_oof_m["Worst_Class_F1"]),
        "InnerDayLODO_day_MacroF1": [float(x) for x in day_macro],
        "InnerDayLODO_day_rows": day_fold_rows,
        "WorstDay_MacroF1": worst_day,
        "Day_MacroF1_SD": day_sd,
        "Robust_Min": float(robust_min),
        "Robust_Mean": float(robust_mean),
        "Robust_WorstClassF1": float(robust_worst_class),
        "Stability_MaxSD": float(stability_max_sd),
        "Robust_BalancedAccuracy": float(robust_bal),
        "Robust_Kappa": float(robust_kap),
    }
    return result


def run_stageb(data, candidates, checkpoint_db: Path, fresh_repeats: int, grand_tracker=None):
    cached = load_stageb_checkpoint(checkpoint_db)
    total = len(candidates)
    tracker = ProgressTracker(total, len(cached), "candidates")

    # The checkpoint omits study names; reconstruct them from candidate metadata for reporting.
    candidate_map = {c["candidate_key"]: c for c in candidates}
    results = {}
    for ck, r in cached.items():
        merged = dict(candidate_map.get(ck, {}))
        merged.update(r)
        results[ck] = merged

    print("\n" + "=" * 100)
    print("12B Stage B:Fresh SGKF + Inner-day LODO Stability Selection")
    print("=" * 100)
    print(f"Total candidates={total} | Completed checkpoints={len(cached)}")
    print(
        f"Per candidate: {fresh_repeats} fresh seeds × 3 folds + 3 inner-day LODO "
        f"= {fresh_repeats*3+3} fits"
    )
    print("Formal Stage B: 600 candidates x 12 fits = 7,200 fits")
    print("=" * 100)

    for i, cand in enumerate(candidates, start=1):
        ck = cand["candidate_key"]
        if ck in results:
            continue

        season = cand["Season"]
        outer_fold = int(cand["Outer_fold"])
        held = pd.Timestamp(cand["Held_out_date"])
        outer_train = data[season].loc[data[season]["Date"] != held].copy().reset_index(drop=True)

        print(
            f"\n[Stage B {tracker.done+1}/{total}] {season} | F{outer_fold} | {cand['Model']} "
            f"| StageA rank={cand['StageA_rank']} trial={cand['StageA_trial_number']} "
            f"| StageA F1={cand['StageA_inner_Macro_F1']:.4f}"
        )
        t0 = time.time()
        result = evaluate_candidate_stageb(outer_train, cand, fresh_repeats)
        save_stageb_checkpoint(checkpoint_db, cand, result)
        merged = dict(cand)
        merged.update(result)
        results[ck] = merged
        duration = time.time() - t0
        tracker.add(duration)
        if grand_tracker is not None:
            grand_tracker.add_units(fresh_repeats * INNER_SPLITS + 3, duration)
        print(
            f"  Robust_Min={result['Robust_Min']:.4f} | Robust_Mean={result['Robust_Mean']:.4f} "
            f"| WorstDay={result['WorstDay_MacroF1']:.4f} "
            f"| WorstClass={result['Robust_WorstClassF1']:.4f} "
            f"| Stability_MaxSD={result['Stability_MaxSD']:.4f} "
            f"| Current candidate {fmt_seconds(duration)}"
        )
        tracker.print(prefix="  [Stage B Overall]")
        if grand_tracker is not None:
            grand_tracker.print(prefix="  [Full Workflow - Fit Equivalent]")

    tracker.print(prefix="[Stage B Complete]")
    return list(results.values())


# =============================================================================
# Deterministic Stage-B ranking, local winners, and inner-evidence common-family selection
# =============================================================================

def stageb_sort_df(df: pd.DataFrame):
    return df.sort_values(
        by=[
            "Robust_Min",
            "Robust_Mean",
            "WorstDay_MacroF1",
            "Robust_WorstClassF1",
            "Stability_MaxSD",
            "Robust_BalancedAccuracy",
            "Robust_Kappa",
            "StageA_inner_Macro_F1",
            "StageA_trial_number",
        ],
        ascending=[False, False, False, False, True, False, False, False, True],
        kind="mergesort",
    )


def stageb_results_dataframe(results):
    rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k not in {"Params", "InnerDayLODO_day_rows"}}
        row["Params_json"] = canonical_params(r["Params"])
        row["FreshSGKF_seed_MacroF1"] = json.dumps(r["FreshSGKF_seed_MacroF1"])
        row["InnerDayLODO_day_MacroF1"] = json.dumps(r["InnerDayLODO_day_MacroF1"])
        rows.append(row)
    return pd.DataFrame(rows)


def select_best_candidate_per_family(stageb_df: pd.DataFrame):
    selected = []
    for (season, outer_fold, model), g in stageb_df.groupby(["Season", "Outer_fold", "Model"], sort=True):
        best = stageb_sort_df(g).iloc[0].copy()
        best["Within_family_selected"] = True
        selected.append(best)
    return pd.DataFrame(selected).reset_index(drop=True)


def select_fold_winner_across_families(family_best_df: pd.DataFrame):
    winners = []
    for (season, outer_fold), g in family_best_df.groupby(["Season", "Outer_fold"], sort=True):
        best = stageb_sort_df(g).iloc[0].copy()
        best["Outer_fold_pipeline_winner"] = True
        winners.append(best)
    return pd.DataFrame(winners).reset_index(drop=True)


def build_final_common_family_table(family_best_df: pd.DataFrame):
    rows = []
    for model in CANDIDATE_MODELS:
        g = family_best_df[family_best_df["Model"] == model].copy()
        if g.empty:
            continue
        summer = g[g["Season"] == "Summer"]
        winter = g[g["Season"] == "Winter"]
        if len(summer) != 4 or len(winter) != 4:
            season_means = g.groupby("Season")["Robust_Min"].mean().to_dict()
            smean = season_means.get("Summer", np.nan)
            wmean = season_means.get("Winter", np.nan)
            valid_means = [x for x in [smean, wmean] if np.isfinite(x)]
            cross_min = min(valid_means) if valid_means else np.nan
            cross_mean = float(np.mean(valid_means)) if valid_means else np.nan
        else:
            smean = float(summer["Robust_Min"].mean())
            wmean = float(winter["Robust_Min"].mean())
            cross_min = min(smean, wmean)
            cross_mean = (smean + wmean) / 2
        rows.append({
            "Model": model,
            "Summer_mean_Robust_Min": smean,
            "Winter_mean_Robust_Min": wmean,
            "CrossSeason_Min_of_SeasonMean": cross_min,
            "CrossSeason_Mean_of_SeasonMean": cross_mean,
            "Across_tasks_WorstDay_min": float(g["WorstDay_MacroF1"].min()),
            "Across_tasks_WorstClass_min": float(g["Robust_WorstClassF1"].min()),
            "Across_tasks_Stability_MaxSD_max": float(g["Stability_MaxSD"].max()),
            "Across_tasks_Robust_BalAcc_mean": float(g["Robust_BalancedAccuracy"].mean()),
            "Across_tasks_Robust_Kappa_mean": float(g["Robust_Kappa"].mean()),
            "Tasks_N": len(g),
            "Model_order": CANDIDATE_MODELS.index(model),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(
        by=[
            "CrossSeason_Min_of_SeasonMean",
            "CrossSeason_Mean_of_SeasonMean",
            "Across_tasks_WorstDay_min",
            "Across_tasks_WorstClass_min",
            "Across_tasks_Stability_MaxSD_max",
            "Across_tasks_Robust_BalAcc_mean",
            "Across_tasks_Robust_Kappa_mean",
            "Model_order",
        ],
        ascending=[False, False, False, False, True, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    out["Final_common_family_rank"] = np.arange(1, len(out) + 1)
    # Retain original output field names for comparison with the archived results.
    out["Selected_for_Step12"] = out["Final_common_family_rank"] == 1
    return out


# =============================================================================
# Final outer-fold evaluation
# =============================================================================

def outer_fold_key(season, outer_fold):
    return f"{season}__F{outer_fold}"


def load_outer_checkpoint(path: Path):
    con = sqlite3.connect(path)
    try:
        fold_df = pd.read_sql_query("SELECT * FROM outer_fold_results", con)
        pred_df = pd.read_sql_query("SELECT * FROM outer_predictions", con)
    finally:
        con.close()
    return fold_df, pred_df


def save_outer_checkpoint(path: Path, winner: pd.Series, metrics: dict, pred_rows: list[dict]):
    fk = outer_fold_key(winner["Season"], int(winner["Outer_fold"]))
    con = sqlite3.connect(path)
    try:
        con.execute(
            """
            INSERT OR REPLACE INTO outer_fold_results
            (fold_key, season, outer_fold, held_out_date, selected_model, candidate_key,
             params_json, metrics_json, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fk, winner["Season"], int(winner["Outer_fold"]),
                pd.Timestamp(winner["Held_out_date"]).strftime("%Y-%m-%d"),
                winner["Model"], winner["candidate_key"], winner["Params_json"],
                json.dumps(metrics, ensure_ascii=False, sort_keys=True),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        con.execute("DELETE FROM outer_predictions WHERE fold_key=?", (fk,))
        con.executemany(
            """
            INSERT INTO outer_predictions
            (fold_key, row_id, y_true, y_pred, selected_model)
            VALUES (?, ?, ?, ?, ?)
            """,
            [(fk, int(r["row_id"]), int(r["y_true"]), int(r["y_pred"]), r["selected_model"])
             for r in pred_rows],
        )
        con.commit()
    finally:
        con.close()


def run_outer_exam(data, winners_df: pd.DataFrame, checkpoint_db: Path, run_dir: Path, grand_tracker=None):
    existing_fold, _ = load_outer_checkpoint(checkpoint_db)
    existing_keys = set(existing_fold["fold_key"].tolist()) if not existing_fold.empty else set()
    total = len(winners_df)
    tracker = ProgressTracker(total, len(existing_keys), "outer folds")
    model_dir = run_dir / "outer_selected_models"
    model_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 100)
    print("Stage 12B outer LODO evaluation")
    print("=" * 100)
    print("Local winners were selected from inner and stability evidence; do not retune using outer scores.")
    print(f"outer folds={total} | Completed checkpoints={len(existing_keys)}")

    for _, winner in winners_df.sort_values(["Season", "Outer_fold"]).iterrows():
        season = winner["Season"]
        fold = int(winner["Outer_fold"])
        fk = outer_fold_key(season, fold)
        if fk in existing_keys:
            continue

        held = pd.Timestamp(winner["Held_out_date"])
        train = data[season].loc[data[season]["Date"] != held].copy().reset_index(drop=True)
        test = data[season].loc[data[season]["Date"] == held].copy().reset_index(drop=True)
        raw = json.loads(winner["Params_json"])
        model = winner["Model"]
        _, weight_mode, _ = decode_config(raw, model)
        fit_seed = model_fit_seed(season, fold, model, phase_offset=900000)

        print(
            f"\n[Outer {tracker.done+1}/{total}] {season} F{fold} held={held.date()} "
            f"| selected={model} | StageB Robust_Min={winner['Robust_Min']:.4f}"
        )
        t0 = time.time()
        pipe = make_pipeline(model, raw, fit_seed)
        fit_pipeline(pipe, train[EP_FEATURES], train["y_model"].to_numpy(), weight_mode)
        pred = pipe.predict(test[EP_FEATURES]).astype(int)
        ytrue = test["y_model"].to_numpy()
        m = metrics_bundle(ytrue, pred)

        pred_rows = []
        for rid, yt, yp in zip(test["row_id"], ytrue, pred):
            pred_rows.append({
                "row_id": int(rid),
                "y_true": int(yt),
                "y_pred": int(yp),
                "selected_model": model,
            })
        save_outer_checkpoint(checkpoint_db, winner, m, pred_rows)
        joblib.dump(pipe, model_dir / f"12B_{season}_F{fold}_{model}.joblib")
        del pipe
        gc.collect()

        dur = time.time() - t0
        tracker.add(dur)
        if grand_tracker is not None:
            grand_tracker.add_units(1, dur)
        print(
            f"  Outer Macro-F1={m['Macro_F1']:.4f} | BalAcc={m['Balanced_Accuracy']:.4f} "
            f"| Kappa={m['Kappa']:.4f} | WorstClass={m['Worst_Class_F1']:.4f}"
        )
        tracker.print(prefix="  [Outer Overall]")
        if grand_tracker is not None:
            grand_tracker.print(prefix="  [Full Workflow - Fit Equivalent]")

    return load_outer_checkpoint(checkpoint_db)


# =============================================================================
# Aggregate pooled outer-fold OOF metrics
# =============================================================================

def build_outer_outputs(data, fold_ckpt: pd.DataFrame, pred_ckpt: pd.DataFrame):
    fold_rows = []
    if not fold_ckpt.empty:
        for _, r in fold_ckpt.iterrows():
            m = json.loads(r["metrics_json"])
            row = {
                "Season": r["season"],
                "Outer_fold": int(r["outer_fold"]),
                "Held_out_date": pd.Timestamp(r["held_out_date"]),
                "Selected_model": r["selected_model"],
                "candidate_key": r["candidate_key"],
            }
            row.update(m)
            fold_rows.append(row)
    fold_df = pd.DataFrame(fold_rows)

    pred_rows = []
    for _, r in pred_ckpt.iterrows():
        season = str(r["fold_key"]).split("__F")[0]
        fold = int(str(r["fold_key"]).split("__F")[1])
        row_id = int(r["row_id"])
        orig = data[season].iloc[row_id]
        pred_rows.append({
            "Season": season,
            "Outer_fold": fold,
            "Date": orig["Date"],
            "row_id": row_id,
            "y_true_model": int(r["y_true"]),
            "y_pred_model": int(r["y_pred"]),
            "y_true_raw": MODEL_TO_RAW[int(r["y_true"])],
            "y_pred_raw": MODEL_TO_RAW[int(r["y_pred"])],
            "Selected_model": r["selected_model"],
        })
    pred_df = (
        pd.DataFrame(pred_rows).sort_values(["Season", "row_id"]).reset_index(drop=True)
        if pred_rows else pd.DataFrame()
    )

    summary_rows, class_rows = [], []
    for season in ["Summer", "Winter"]:
        g = pred_df[pred_df["Season"] == season]
        if g.empty:
            continue
        y = g["y_true_model"].to_numpy()
        p = g["y_pred_model"].to_numpy()
        m = metrics_bundle(y, p)
        ff = fold_df[fold_df["Season"] == season]
        summary = {
            "Season": season,
            "N": len(g),
            "OOF_Macro_F1": m["Macro_F1"],
            "OOF_Balanced_Accuracy": m["Balanced_Accuracy"],
            "OOF_Kappa": m["Kappa"],
            "OOF_Accuracy": m["Accuracy"],
            "OOF_Worst_Class_F1": m["Worst_Class_F1"],
            "Outer_fold_MacroF1_mean": float(ff["Macro_F1"].mean()) if len(ff) else np.nan,
            "Outer_fold_MacroF1_SD_descriptive": float(ff["Macro_F1"].std(ddof=1)) if len(ff) > 1 else np.nan,
            "Outer_fold_MacroF1_worst": float(ff["Macro_F1"].min()) if len(ff) else np.nan,
            "Selected_family_sequence": " | ".join(ff.sort_values("Outer_fold")["Selected_model"].astype(str).tolist()),
        }
        summary_rows.append(summary)
        for raw_cls in [1, 2, 3]:
            class_rows.append({
                "Season": season,
                "Class": raw_cls,
                "Precision": m[f"Class{raw_cls}_Precision"],
                "Recall": m[f"Class{raw_cls}_Recall"],
                "F1": m[f"Class{raw_cls}_F1"],
                "Support": m[f"Class{raw_cls}_Support"],
            })
    return fold_df, pred_df, pd.DataFrame(summary_rows), pd.DataFrame(class_rows)


# =============================================================================
# Export Excel and JSON
# =============================================================================

def config_df(input_file, db12a, run_dir, smoke_test):
    return pd.DataFrame([
        {"Item": "Step", "Value": "12B Targeted Bayesian + Stability Selection (source-data diagnostic)"},
        {"Item": "Input", "Value": str(input_file)},
        {"Item": "12A DB", "Value": str(db12a)},
        {"Item": "Expected groups", "Value": "Summer=190; Winter=296"},
        {"Item": "Feature set", "Value": "EP only"},
        {"Item": "Candidate families", "Value": ", ".join(CANDIDATE_MODELS)},
        {"Item": "Outer", "Value": "4-fold Leave-One-Day-Out; Summer/Winter separately"},
        {"Item": "Stage A inner", "Value": "3-fold StratifiedGroupKFold(groups=measurement_group); shared splits within Season×Outer fold"},
        {"Item": "Stage A trials/study", "Value": 2 if smoke_test else STAGEA_TRIALS},
        {"Item": "Stage A pruner", "Value": "NopPruner"},
        {"Item": "12A targeted history", "Value": "current outer-fold's own 12A EP study only; first 80 COMPLETE trials"},
        {"Item": "Elite", "Value": "Top16 by 12A inner Macro-F1"},
        {"Item": "Target numeric rule", "Value": "Q10-Q90 + 10% Broad-span padding each side; min width=30% Broad span; log params in log space"},
        {"Item": "Categorical rule", "Value": "keep all 12A Broad choices"},
        {"Item": "Selector", "Value": str(SELECTOR_CHOICES)},
        {"Item": "Weight", "Value": str(WEIGHT_CHOICES)},
        {"Item": "Stage B TopK", "Value": 2 if smoke_test else TOP_K_STAGEB},
        {"Item": "Stage B fresh SGKF repeats", "Value": 1 if smoke_test else FRESH_SGKF_REPEATS},
        {"Item": "Stage B", "Value": "fresh SGKF + inner-day LODO; mechanisms combined 1:1 via Robust_Min/Robust_Mean"},
        {"Item": "Selection order", "Value": "Robust_Min > Robust_Mean > WorstDay > WorstClass > lower MaxSD > BalAcc > Kappa > StageA F1"},
        {"Item": "SMOTE", "Value": "NO"},
        {"Item": "Voting/Stacking", "Value": "NO"},
        {"Item": "Threshold post-processing", "Value": "NO"},
        {"Item": "Outer feedback tuning", "Value": "PROHIBITED"},
        {"Item": "Stage A resume", "Value": "Optuna SQLite trial-level"},
        {"Item": "Stage B resume", "Value": "SQLite candidate-level"},
        {"Item": "Outer resume", "Value": "SQLite fold-level + predictions"},
        {"Item": "Progress/ETA", "Value": "Stage-local + whole-pipeline fit-equivalent percentage, elapsed time, ETA, estimated finish time"},
        {"Item": "4-day SD", "Value": "descriptive day-to-day variability only"},
        {"Item": "Diagnostic role", "Value": "same frozen 11B methodology; data-version comparison only"},
        {"Item": "Smoke test", "Value": bool(smoke_test)},
        {"Item": "Run dir", "Value": str(run_dir)},
    ])


def write_excel(path: Path, sheets: dict[str, pd.DataFrame]):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

        # Formatting for readability only; it does not affect any analysis.
        wb = writer.book
        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            if ws.max_row >= 1 and ws.max_column >= 1:
                ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = cell.font.copy(bold=True)
            for col_cells in ws.columns:
                letter = col_cells[0].column_letter
                max_len = 0
                for cell in col_cells[:200]:
                    try:
                        max_len = max(max_len, len(str(cell.value)) if cell.value is not None else 0)
                    except Exception:
                        pass
                ws.column_dimensions[letter].width = min(max(max_len + 2, 10), 45)


# =============================================================================
# Main entry point
# =============================================================================

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    input_file, db12a, run_dir = resolve_paths(args)

    output_xlsx = run_dir / OUTPUT_XLSX_NAME
    log_path = run_dir / LOG_NAME
    stagea_db = run_dir / STAGEA_DB_NAME
    checkpoint_db = run_dir / CHECKPOINT_DB_NAME
    spaces_json = run_dir / SPACE_JSON_NAME
    config_json = run_dir / CONFIG_JSON_NAME

    init_checkpoint_db(checkpoint_db)

    # Append to logs to preserve historical progress when resuming.
    log_mode = "a" if log_path.exists() else "w"
    with open(log_path, log_mode, encoding="utf-8") as log_fp:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = Tee(sys.stdout, log_fp)
        sys.stderr = Tee(sys.stderr, log_fp)

        try:
            print("\n" + "=" * 100)
            print("12B: Targeted Bayesian optimization, stability selection, and nested LODO")
            print("=" * 100)
            print(f"Started: {datetime.now()}")
            print(f"Input: {input_file}")
            print(f"12A DB:{db12a}")
            print(f"Output directory: {run_dir}")
            print(f"Smoke test:{args.smoke_test}")
            print("This script executes directly and does not use archived 11A/11B databases.")
            print("=" * 100)

            data = load_data(input_file)
            for season in ["Summer", "Winter"]:
                d = data[season]
                print(
                    f"[Data validation] {season}: N={len(d)} | days={d['Date'].nunique()} | "
                    f"groups={d['measurement_group'].nunique()} | "
                    f"classes={d[TARGET].value_counts().sort_index().to_dict()}"
                )

            if args.smoke_test:
                models = ["GB"]
                seasons = ["Summer"]
                folds = [1]
                stagea_trials = 2
                top_k = 2
                fresh_repeats = 1
            else:
                models = list(CANDIDATE_MODELS)
                seasons = ["Summer", "Winter"]
                folds = [1, 2, 3, 4]
                stagea_trials = STAGEA_TRIALS
                top_k = TOP_K_STAGEB
                fresh_repeats = FRESH_SGKF_REPEATS

            # Targeted ranges are derived only from the matching outer-fold 12A EP history.
            spaces, space_audit_df = derive_all_targeted_spaces(
                data=data,
                db12a=db12a,
                models=models,
                seasons=seasons,
                folds=folds,
            )
            with open(spaces_json, "w", encoding="utf-8") as f:
                json.dump(spaces, f, ensure_ascii=False, indent=2, default=str)

            n_studies = len(seasons) * len(folds) * len(models)
            n_candidates = n_studies * top_k
            total_fit_units = (
                n_studies * stagea_trials * INNER_SPLITS
                + n_candidates * (fresh_repeats * INNER_SPLITS + 3)
                + len(seasons) * len(folds)
            )

            provisional_plans = []
            for _season in seasons:
                _dates = sorted(pd.unique(data[_season]["Date"]))
                for _fold in folds:
                    for _model in models:
                        provisional_plans.append({
                            "season": _season, "outer_fold": _fold,
                            "held_date": _dates[_fold - 1], "model": _model
                        })
            existing_stagea = stagea_existing_finished(stagea_db, provisional_plans, stagea_trials)
            existing_stageb = len(load_stageb_checkpoint(checkpoint_db))
            existing_outer_df, _ = load_outer_checkpoint(checkpoint_db)
            existing_outer = len(existing_outer_df)
            initial_fit_units = (
                existing_stagea * INNER_SPLITS
                + existing_stageb * (fresh_repeats * INNER_SPLITS + 3)
                + existing_outer
            )
            grand_tracker = ProgressTracker(
                total_units=total_fit_units,
                initial_done=min(initial_fit_units, total_fit_units),
                unit_name="fit-equivalent units",
            )
            print(
                f"\n[Full workflow plan] studies={n_studies} | StageA trials={n_studies*stagea_trials} | "
                f"StageB candidates={n_candidates} | Total fit-equivalent units={total_fit_units} | "
                f"Previously completed={min(initial_fit_units, total_fit_units)}"
            )
            if not args.smoke_test:
                print("Formal budget: 40 x 200 x 3 + 600 x 12 + 8 = 31,208 fit-equivalent units")
            grand_tracker.print(prefix="[Full Workflow - Fit Equivalent]")

            plans = run_stagea(
                data=data,
                spaces=spaces,
                run_dir=run_dir,
                stagea_db=stagea_db,
                models=models,
                seasons=seasons,
                folds=folds,
                trials_per_study=stagea_trials,
                smoke_test=args.smoke_test,
                grand_tracker=grand_tracker,
            )

            candidates, candidate_df = extract_stageb_candidates(
                stagea_db, plans, top_k=top_k, trials_per_study=stagea_trials
            )
            print(f"\nStage A -> Stage B candidates:{len(candidates)}")
            if not args.smoke_test:
                print("Formal plan: 40 studies x Top-15 = 600 candidates")

            stageb_results = run_stageb(
                data=data,
                candidates=candidates,
                checkpoint_db=checkpoint_db,
                fresh_repeats=fresh_repeats,
                grand_tracker=grand_tracker,
            )
            stageb_df = stageb_results_dataframe(stageb_results)

            family_best_df = select_best_candidate_per_family(stageb_df)
            winners_df = select_fold_winner_across_families(family_best_df)

            common_family_df = build_final_common_family_table(family_best_df)
            if not common_family_df.empty:
                selected_common = common_family_df.iloc[0]["Model"]
                print("\n" + "=" * 100)
                print(f"Common model family (Stage B inner evidence only): {selected_common}")
                print("Common-family selection supports downstream interpretation only; no retuning on outer results.")
                print("=" * 100)

            fold_ckpt, pred_ckpt = run_outer_exam(
                data=data,
                winners_df=winners_df,
                checkpoint_db=checkpoint_db,
                run_dir=run_dir,
                grand_tracker=grand_tracker,
            )

            fold_df, pred_df, oof_summary_df, class_oof_df = build_outer_outputs(
                data, fold_ckpt, pred_ckpt
            )

            tables = {
                "00_Run_Configuration": config_df(input_file, db12a, run_dir, args.smoke_test),
                "01_Targeted_Search_Spaces": space_audit_df,
                "02_StageA_TopK": candidate_df,
                "03_StageB_All_Candidates": stageb_df,
                "04_Best_Family_Configs": family_best_df,
                "05_Fold_Local_Winners": winners_df,
                "06_Common_Family_Selection": common_family_df,
                "07_Outer_Fold_Performance": fold_df,
                "08_Pooled_OOF_Summary": oof_summary_df,
                "09_Classwise_OOF": class_oof_df,
                "10_Outer_Predictions": pred_df,
            }
            write_excel(output_xlsx, tables)

            config_payload = {
                "step": "12B",
                "role": "source-data diagnostic rerun of frozen 11B methodology",
                "input": str(input_file),
                "broad_db": str(db12a),
                "broad_study_prefix": "12A__",
                "expected_groups": EXPECTED_GROUPS,
                "candidate_models": CANDIDATE_MODELS,
                "formal_models_this_run": models,
                "feature_set": "EP",
                "outer": "4-fold LODO",
                "stageA_inner": "3-fold StratifiedGroupKFold(groups=measurement_group)",
                "targeted_space": {
                    "source": "current outer-fold's own 12A EP study only",
                    "history_complete_trials": HISTORY_COMPLETE_TRIALS,
                    "elite_n": ELITE_N,
                    "q": [ELITE_Q_LOW, ELITE_Q_HIGH],
                    "padding_frac_of_broad_each_side": TARGET_PADDING_FRAC,
                    "min_width_frac_of_broad": TARGET_MIN_WIDTH_FRAC,
                    "min_active_elite": MIN_ACTIVE_ELITE,
                    "categorical": "keep all broad choices",
                },
                "stageA_trials_per_study": stagea_trials,
                "stageA_pruner": "NopPruner",
                "stageB_top_k": top_k,
                "stageB_fresh_sgkf_repeats": fresh_repeats,
                "stageB_inner_day_lodo": True,
                "selection_rule": [
                    "max Robust_Min", "max Robust_Mean", "max WorstDay MacroF1",
                    "max Robust_WorstClassF1", "min Stability_MaxSD",
                    "max Robust_BalancedAccuracy", "max Robust_Kappa",
                    "max StageA inner MacroF1", "min trial_number",
                ],
                "outer_feedback_tuning": False,
                "smote": False,
                "voting_stacking": False,
                "threshold_postprocessing": False,
                "smoke_test": bool(args.smoke_test),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }
            with open(config_json, "w", encoding="utf-8") as f:
                json.dump(config_payload, f, ensure_ascii=False, indent=2)

            print("\n" + "=" * 100)
            if args.smoke_test:
                print("Stage 12B smoke test completed. Do not use the results in the manuscript.")
                print("Use a new output directory and remove --smoke-test for the formal run.")
            else:
                print("Stage 12B complete.")
                print("Do not use outer-fold results to revise the frozen selection rules or search spaces.")
            print(f"Results workbook: {output_xlsx}")
            print(f"Targeted search-space JSON: {spaces_json}")
            print(f"Stage A DB:{stagea_db}")
            print(f"Checkpoint DB:{checkpoint_db}")
            print(f"Run configuration: {config_json}")
            print(f"Finished: {datetime.now()}")
            print("=" * 100)

        except Exception:
            print("\n" + "=" * 100)
            print("Stage 12B terminated with an error; completed trials and folds are resumable.")
            traceback.print_exc()
            print("=" * 100)
            raise
        finally:
            sys.stdout = old_out
            sys.stderr = old_err


if __name__ == "__main__":
    main()

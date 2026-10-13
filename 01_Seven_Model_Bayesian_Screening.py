from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import NopPruner
except ImportError as exc:
    raise ImportError("缺少Optuna，请先执行：pip install optuna") from exc


# ======================================================================
# 0. 可选第三方模型
# ======================================================================

OPTIONAL_MODEL_STATUS = {}

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
    OPTIONAL_MODEL_STATUS["XGB"] = "available"
except Exception as exc:
    HAS_XGB = False
    OPTIONAL_MODEL_STATUS["XGB"] = f"not available: {exc}"

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
    OPTIONAL_MODEL_STATUS["LGBM"] = "available"
except Exception as exc:
    HAS_LGBM = False
    OPTIONAL_MODEL_STATUS["LGBM"] = f"not available: {exc}"

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
    OPTIONAL_MODEL_STATUS["CAT"] = "available"
except Exception as exc:
    HAS_CAT = False
    OPTIONAL_MODEL_STATUS["CAT"] = f"not available: {exc}"


# ======================================================================
# 1. 路径与固定参数
# ======================================================================

DEFAULT_BASE_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT = (
    DEFAULT_BASE_DIR / "Analytical_Dataset_1001.xlsx"
)

RUN_DIR_NAME = "12A_BroadBayesian_NestedLODO_SevenModels"
OUTPUT_XLSX_NAME = "12A_BroadBayesian_NestedLODO_SevenModels.xlsx"
LOG_NAME = "12A_BroadBayesian_NestedLODO_SevenModels.log"
DB_NAME = "optuna_studies.db"
CONFIG_NAME = "12A_Run_Configuration.json"

SHEET_MAP = {
    "Summer": "summer",
    "Winter": "winter",
}

EXPECTED_N = {
    "Summer": 477,
    "Winter": 524,
}

EXPECTED_GROUPS = {
    # “完整数据合理性排查_源数据未改”版本的measurement_group数量。
    # 12A仅用于数据版本诊断，不修改/重建measurement_group。
    "Summer": 190,
    "Winter": 296,
}

TARGET = "Thermal_satisfaction_3"

RAW_TO_MODEL = {
    1: 0,
    2: 1,
    3: 2,
}

MODEL_TO_RAW = {
    0: 1,
    1: 2,
    2: 3,
}

CLASS_ORDER = [0, 1, 2]

RANDOM_STATE = 20260816
INNER_SPLITS = 3

N_TRIALS_PER_STUDY = 80

N_STARTUP_TRIALS = 12

# 正式11A不做trial pruning：每个trial完整评估3个inner folds。
# 原因：inner只有3折，完整评估更利于模型/参数之间公平比较，
# 避免因固定fold顺序导致的部分折早停偏差。

MODEL_N_JOBS = -1

# 每多少个trial更新一次详细打印。
PROGRESS_EVERY_TRIALS = 5

# 默认不保存全部11A模型，减少磁盘占用。
SAVE_OUTER_MODELS = False


# ======================================================================
# 2. 特征定义：与07A一致
# ======================================================================

OBJECTIVE = [
    "Temperature",
    "Wind_speed",
    "Humidity",
    "Globe_temperature",
    "Illumination",
]

SPATIAL = [
    "Location",
    "Scene",
]

SUBJECTIVE = [
    "TSV", "TCV", "TPV", "TAV",
    "SSV", "SCV", "SPV", "SAV",
    "HSV", "WSV", "HPV", "WPV",
]

FEATURE_SETS = {
    "E": {
        "label": "Microclimate + spatial context",
        "numeric": OBJECTIVE,
        "categorical": SPATIAL,
    },
    "P": {
        "label": "Subjective perceptions only",
        "numeric": SUBJECTIVE,
        "categorical": [],
    },
    "EP": {
        "label": "Microclimate + spatial context + subjective perceptions",
        "numeric": OBJECTIVE + SUBJECTIVE,
        "categorical": SPATIAL,
    },
}

FEATURE_SET_ORDER = ["E", "P", "EP"]


# ======================================================================
# 3. 固定七模型池
# ======================================================================

ALL_MODELS = [
    "RF",
    "ET",
    "GB",
    "HGB",
    "XGB",
    "LGBM",
    "CAT",
]

# 正式11A要求7个模型全部可用，避免不同机器得到不同模型池。
MISSING_OPTIONAL = []
if not HAS_XGB:
    MISSING_OPTIONAL.append("xgboost")
if not HAS_LGBM:
    MISSING_OPTIONAL.append("lightgbm")
if not HAS_CAT:
    MISSING_OPTIONAL.append("catboost")

if MISSING_OPTIONAL:
    raise ImportError(
        "正式11A固定使用7个模型，但当前缺少："
        + ", ".join(MISSING_OPTIONAL)
        + "\n请先安装：pip install xgboost lightgbm catboost"
    )


# ======================================================================
# 4. 命令行
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="12A 诊断 Broad Bayesian Search：严格Nested LODO（源数据未改）"
    )
    parser.add_argument(
        "--input",
        default=None,
        help="12A诊断输入xlsx；不填则使用“完整数据合理性排查_源数据未改”默认路径",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录；不填则在输入文件同目录创建11A结果目录",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "仅用于检查代码能否运行：每study 1 trial，"
            "只跑RF，E/EP，两季各第1个outer fold；"
            "不会污染正式11A数据库。"
            "不要把smoke-test结果用于论文。"
        ),
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="保存每个outer study最终模型；默认关闭以节省磁盘。",
    )
    return parser.parse_args()


# ======================================================================
# 5. 日志、进度、ETA
# ======================================================================

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


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
    frac = min(max(done / total, 0.0), 1.0)
    n = int(frac * width)
    if done >= total:
        return "[" + "=" * width + "]"
    return "[" + "=" * n + ">" + "." * max(width - n - 1, 0) + "]"


class GlobalProgress:
    """
    以“计划trial数”为权重估计总体进度。
    对MedianPruner提前结束的trial仍视作1个完成trial，
    因为Optuna study预算是按trial个数定义。
    """
    def __init__(self, total_trials):
        self.total_trials = int(total_trials)
        self.completed_trials = 0
        self.start = time.time()
        self.trial_durations = []

    def add_trial(self, duration):
        self.completed_trials += 1
        if duration is not None and np.isfinite(duration) and duration >= 0:
            self.trial_durations.append(float(duration))

    def eta_seconds(self):
        if self.completed_trials <= 0:
            return None
        elapsed = time.time() - self.start

        # 早期用累计速度；>=10 trial后混合近期20个trial均值，更适应模型速度变化。
        overall_per_trial = elapsed / max(self.completed_trials, 1)

        if len(self.trial_durations) >= 10:
            recent = np.mean(self.trial_durations[-20:])
            sec_per_trial = 0.65 * recent + 0.35 * overall_per_trial
        else:
            sec_per_trial = overall_per_trial

        remain = max(self.total_trials - self.completed_trials, 0)
        return sec_per_trial * remain

    def print(self, prefix="[总体]"):
        elapsed = time.time() - self.start
        eta = self.eta_seconds()
        finish = (
            datetime.now() + timedelta(seconds=float(eta))
            if eta is not None and np.isfinite(eta)
            else None
        )

        frac = (
            self.completed_trials / self.total_trials
            if self.total_trials > 0
            else 0
        )

        print(
            f"{prefix} {ascii_bar(self.completed_trials, self.total_trials)} "
            f"{self.completed_trials}/{self.total_trials} trials "
            f"({frac*100:.1f}%) "
            f"| 已耗时 {fmt_seconds(elapsed)} "
            f"| ETA {fmt_seconds(eta)} "
            f"| 预计完成 "
            f"{finish.strftime('%Y-%m-%d %H:%M:%S') if finish else '--'}"
        )


class StudyProgressCallback:
    def __init__(
        self,
        tracker: GlobalProgress,
        planned_trials: int,
        initial_existing_trials: int,
        label: str,
        print_every: int = 5,
    ):
        self.tracker = tracker
        self.planned_trials = int(planned_trials)
        self.initial_existing_trials = int(initial_existing_trials)
        self.label = label
        self.print_every = max(int(print_every), 1)
        self.study_start = time.time()
        self.last_callback_time = time.time()
        self.new_done = 0

    def __call__(self, study, trial):
        now = time.time()
        duration = now - self.last_callback_time
        self.last_callback_time = now
        self.new_done += 1
        self.tracker.add_trial(duration)

        if (
            self.new_done % self.print_every == 0
            or self.new_done >= self.planned_trials
        ):
            local_elapsed = now - self.study_start
            local_rate = local_elapsed / max(self.new_done, 1)
            local_remain = max(self.planned_trials - self.new_done, 0)
            local_eta = local_rate * local_remain

            completed = [
                t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
                and t.value is not None
            ]
            best = max((float(t.value) for t in completed), default=np.nan)

            print(
                f"\n  [Study] {self.label}\n"
                f"  {ascii_bar(self.new_done, self.planned_trials)} "
                f"{self.new_done}/{self.planned_trials} new trials "
                f"| best inner Macro-F1="
                f"{best:.4f}" if np.isfinite(best) else
                f"\n  [Study] {self.label}\n"
                f"  {ascii_bar(self.new_done, self.planned_trials)} "
                f"{self.new_done}/{self.planned_trials} new trials "
                f"| best inner Macro-F1=--"
            )
            print(
                f"  局部已耗时 {fmt_seconds(local_elapsed)} "
                f"| 局部ETA {fmt_seconds(local_eta)}"
            )
            self.tracker.print(prefix="  [总体]")


# ======================================================================
# 6. 数据读取与验证
# ======================================================================

def resolve_input(input_arg):
    p = Path(input_arg) if input_arg else DEFAULT_INPUT
    if not p.exists():
        raise FileNotFoundError(
            f"找不到输入文件：\n{p}\n"
            "当前12A诊断输入应为："
            "Analytical_Dataset_1001.xlsx"
        )
    return p


def load_data(input_file: Path):
    raw = pd.read_excel(
        input_file,
        sheet_name=list(SHEET_MAP.values()),
        engine="openpyxl",
    )

    required = list(dict.fromkeys(
        ["Date", "measurement_group", TARGET]
        + OBJECTIVE
        + SPATIAL
        + SUBJECTIVE
    ))

    out = {}

    for season, sheet in SHEET_MAP.items():
        df = raw[sheet].copy()
        df.columns = [str(c).strip() for c in df.columns]

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"{season}缺少字段：{missing}")

        if len(df) != EXPECTED_N[season]:
            raise ValueError(
                f"{season}: N={len(df)}，预期={EXPECTED_N[season]}"
            )

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

        for c in OBJECTIVE + SUBJECTIVE:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        for c in SPATIAL:
            df[c] = df[c].astype(str).str.strip()

        df["measurement_group"] = (
            df["measurement_group"].astype(str).str.strip()
        )

        if df[required].isna().any().any():
            bad = df[required].isna().sum()
            bad = bad[bad > 0].to_dict()
            raise ValueError(f"{season}存在缺失/非法字段：{bad}")

        df[TARGET] = df[TARGET].astype(int)

        if set(df[TARGET].unique()) != {1, 2, 3}:
            raise ValueError(f"{season}: Target不是完整1/2/3。")

        if df["Date"].nunique() != 4:
            raise ValueError(f"{season}: 调查日不是4天。")

        group_count = int(df["measurement_group"].nunique())
        if group_count != EXPECTED_GROUPS[season]:
            raise ValueError(
                f"{season}: measurement_group={group_count}，"
                f"冻结母表预期={EXPECTED_GROUPS[season]}"
            )

        cross = (
            df.groupby("measurement_group")["Date"]
            .nunique()
            .gt(1)
            .sum()
        )
        if cross:
            raise ValueError(
                f"{season}: 有{cross}个measurement_group跨日期。"
            )

        df["y_model"] = df[TARGET].map(RAW_TO_MODEL).astype(int)

        for d, g in df.groupby("Date", sort=True):
            if set(g["y_model"].unique()) != {0, 1, 2}:
                raise ValueError(
                    f"{season} {d.date()}: held-out day不含完整三类。"
                )

        df = df.reset_index(drop=True)
        df["row_id"] = np.arange(len(df))
        out[season] = df

    return out


# ======================================================================
# 7. 预处理
# ======================================================================

def make_onehot():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float64,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
            dtype=np.float64,
        )


def make_preprocessor(numeric_features, categorical_features, scale_numeric):
    num_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    transformers = [
        (
            "num",
            Pipeline(num_steps),
            numeric_features,
        )
    ]

    if categorical_features:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot()),
        ])
        transformers.append(
            ("cat", cat_pipe, categorical_features)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ======================================================================
# 8. 参数空间
# ======================================================================

def suggest_selector_and_weight(trial):
    selector_percentile = trial.suggest_categorical(
        "selector_percentile",
        [50, 65, 80, 100],
    )
    weight_mode = trial.suggest_categorical(
        "weight_mode",
        ["none", "balanced"],
    )
    return selector_percentile, weight_mode


def suggest_model_params(trial, model_name):
    p = {}

    if model_name == "LR":
        p["C"] = trial.suggest_float("C", 1e-3, 150.0, log=True)
        p["solver"] = trial.suggest_categorical(
            "solver", ["lbfgs", "saga"]
        )
        if p["solver"] == "saga":
            p["penalty"] = trial.suggest_categorical(
                "penalty", ["l1", "l2", "elasticnet"]
            )
            if p["penalty"] == "elasticnet":
                p["l1_ratio"] = trial.suggest_float(
                    "l1_ratio", 0.05, 0.95
                )
        else:
            p["penalty"] = "l2"

    elif model_name == "DT":
        p["max_depth_mode"] = trial.suggest_categorical(
            "max_depth_mode", ["none", "finite"]
        )
        if p["max_depth_mode"] == "finite":
            p["max_depth"] = trial.suggest_int("max_depth", 2, 20)
        else:
            p["max_depth"] = None
        p["min_samples_split"] = trial.suggest_int(
            "min_samples_split", 2, 20
        )
        p["min_samples_leaf"] = trial.suggest_int(
            "min_samples_leaf", 1, 12
        )
        p["max_features"] = trial.suggest_categorical(
            "max_features", [None, "sqrt", "log2", 0.5, 0.7, 1.0]
        )
        p["criterion"] = trial.suggest_categorical(
            "criterion", ["gini", "entropy", "log_loss"]
        )

    elif model_name in {"RF", "ET"}:
        p["n_estimators"] = trial.suggest_int(
            "n_estimators", 300, 1400, step=50
        )
        p["max_depth_mode"] = trial.suggest_categorical(
            "max_depth_mode", ["none", "finite"]
        )
        if p["max_depth_mode"] == "finite":
            p["max_depth"] = trial.suggest_int("max_depth", 3, 30)
        else:
            p["max_depth"] = None
        p["min_samples_split"] = trial.suggest_int(
            "min_samples_split", 2, 20
        )
        p["min_samples_leaf"] = trial.suggest_int(
            "min_samples_leaf", 1, 12
        )
        p["max_features"] = trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0]
        )
        p["criterion"] = trial.suggest_categorical(
            "criterion", ["gini", "entropy", "log_loss"]
        )
        p["bootstrap"] = trial.suggest_categorical(
            "bootstrap", [True, False]
        )
        if p["bootstrap"]:
            p["max_samples"] = trial.suggest_categorical(
                "max_samples", [None, 0.65, 0.80, 0.95]
            )
        else:
            p["max_samples"] = None

    elif model_name == "GB":
        p["n_estimators"] = trial.suggest_int(
            "n_estimators", 100, 800, step=25
        )
        p["learning_rate"] = trial.suggest_float(
            "learning_rate", 0.01, 0.25, log=True
        )
        p["max_depth"] = trial.suggest_int("max_depth", 1, 5)
        p["min_samples_split"] = trial.suggest_int(
            "min_samples_split", 2, 20
        )
        p["min_samples_leaf"] = trial.suggest_int(
            "min_samples_leaf", 1, 12
        )
        p["subsample"] = trial.suggest_float(
            "subsample", 0.60, 1.00
        )
        p["max_features"] = trial.suggest_categorical(
            "max_features", [None, "sqrt", "log2", 0.5, 0.7, 1.0]
        )

    elif model_name == "HGB":
        p["learning_rate"] = trial.suggest_float(
            "learning_rate", 0.01, 0.25, log=True
        )
        p["max_iter"] = trial.suggest_int(
            "max_iter", 100, 700, step=25
        )
        p["max_leaf_nodes"] = trial.suggest_int(
            "max_leaf_nodes", 7, 63
        )
        p["max_depth_mode"] = trial.suggest_categorical(
            "max_depth_mode", ["none", "finite"]
        )
        if p["max_depth_mode"] == "finite":
            p["max_depth"] = trial.suggest_int("max_depth", 2, 12)
        else:
            p["max_depth"] = None
        p["min_samples_leaf"] = trial.suggest_int(
            "min_samples_leaf", 5, 40
        )
        p["l2_regularization"] = trial.suggest_float(
            "l2_regularization", 1e-6, 20.0, log=True
        )
        p["max_bins"] = trial.suggest_categorical(
            "max_bins", [63, 127, 255]
        )

    elif model_name == "SVM_RBF":
        p["C"] = trial.suggest_float(
            "C", 1e-2, 150.0, log=True
        )
        p["gamma"] = trial.suggest_float(
            "gamma", 1e-4, 2.0, log=True
        )

    elif model_name == "XGB":
        p["n_estimators"] = trial.suggest_int(
            "n_estimators", 200, 1200, step=50
        )
        p["learning_rate"] = trial.suggest_float(
            "learning_rate", 0.01, 0.25, log=True
        )
        p["max_depth"] = trial.suggest_int("max_depth", 2, 9)
        p["min_child_weight"] = trial.suggest_float(
            "min_child_weight", 0.5, 15.0, log=True
        )
        p["subsample"] = trial.suggest_float(
            "subsample", 0.60, 1.00
        )
        p["colsample_bytree"] = trial.suggest_float(
            "colsample_bytree", 0.50, 1.00
        )
        p["gamma"] = trial.suggest_float(
            "gamma", 0.0, 5.0
        )
        p["reg_alpha"] = trial.suggest_float(
            "reg_alpha", 1e-7, 10.0, log=True
        )
        p["reg_lambda"] = trial.suggest_float(
            "reg_lambda", 1e-4, 50.0, log=True
        )

    elif model_name == "LGBM":
        p["n_estimators"] = trial.suggest_int(
            "n_estimators", 200, 1200, step=50
        )
        p["learning_rate"] = trial.suggest_float(
            "learning_rate", 0.01, 0.25, log=True
        )
        p["num_leaves"] = trial.suggest_int(
            "num_leaves", 7, 63
        )
        p["max_depth_mode"] = trial.suggest_categorical(
            "max_depth_mode", ["none", "finite"]
        )
        if p["max_depth_mode"] == "finite":
            p["max_depth"] = trial.suggest_int("max_depth", 3, 12)
        else:
            p["max_depth"] = -1
        p["min_child_samples"] = trial.suggest_int(
            "min_child_samples", 5, 50
        )
        p["colsample_bytree"] = trial.suggest_float(
            "colsample_bytree", 0.50, 1.00
        )
        p["reg_alpha"] = trial.suggest_float(
            "reg_alpha", 1e-7, 10.0, log=True
        )
        p["reg_lambda"] = trial.suggest_float(
            "reg_lambda", 1e-4, 50.0, log=True
        )
        p["bagging_mode"] = trial.suggest_categorical(
            "bagging_mode", ["off", "on"]
        )
        if p["bagging_mode"] == "on":
            p["subsample"] = trial.suggest_float(
                "subsample", 0.60, 0.95
            )
            p["subsample_freq"] = 1
        else:
            p["subsample"] = 1.0
            p["subsample_freq"] = 0

    elif model_name == "CAT":
        p["iterations"] = trial.suggest_int(
            "iterations", 200, 1000, step=50
        )
        p["depth"] = trial.suggest_int("depth", 3, 8)
        p["learning_rate"] = trial.suggest_float(
            "learning_rate", 0.01, 0.25, log=True
        )
        p["l2_leaf_reg"] = trial.suggest_float(
            "l2_leaf_reg", 1e-3, 30.0, log=True
        )
        p["random_strength"] = trial.suggest_float(
            "random_strength", 1e-4, 10.0, log=True
        )
        p["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature", 0.0, 5.0
        )
        p["border_count"] = trial.suggest_categorical(
            "border_count", [32, 64, 128]
        )

    else:
        raise KeyError(f"未知模型：{model_name}")

    return p


# ======================================================================
# 9. 建模
# ======================================================================

def model_requires_scaling(model_name):
    return model_name in {"LR", "SVM_RBF"}



class CatBoostSklearnAdapter(ClassifierMixin, BaseEstimator):
    """
    兼容新旧scikit-learn与部分CatBoost版本的轻量适配器。
    某些CatBoost版本缺少新版sklearn要求的__sklearn_tags__，
    直接作为Pipeline最后一步可能在predict/check_is_fitted时报错。
    本适配器只做API兼容，不改变CatBoost训练逻辑。
    """
    def __init__(self, params=None):
        self.params = {} if params is None else params

    def fit(self, X, y, sample_weight=None):
        self.model_ = CatBoostClassifier(**dict(self.params))
        self.model_.fit(
            X,
            y,
            sample_weight=sample_weight,
        )
        self.classes_ = np.asarray(
            getattr(self.model_, "classes_", np.unique(y))
        )
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        pred = self.model_.predict(X)
        return np.asarray(pred).reshape(-1).astype(int)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def make_estimator(model_name, params, seed):
    # 清理仅用于搜索结构的参数
    params = dict(params)
    params.pop("max_depth_mode", None)
    params.pop("bagging_mode", None)

    if model_name == "LR":
        return LogisticRegression(
            C=params["C"],
            solver=params["solver"],
            penalty=params["penalty"],
            l1_ratio=params.get("l1_ratio", None),
            max_iter=4000,
            random_state=seed,
        )

    if model_name == "DT":
        return DecisionTreeClassifier(
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            criterion=params["criterion"],
            random_state=seed,
        )

    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            criterion=params["criterion"],
            bootstrap=params["bootstrap"],
            max_samples=params.get("max_samples", None),
            random_state=seed,
            n_jobs=MODEL_N_JOBS,
        )

    if model_name == "ET":
        return ExtraTreesClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            criterion=params["criterion"],
            bootstrap=params["bootstrap"],
            max_samples=params.get("max_samples", None),
            random_state=seed,
            n_jobs=MODEL_N_JOBS,
        )

    if model_name == "GB":
        return GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            subsample=params["subsample"],
            max_features=params["max_features"],
            random_state=seed,
        )

    if model_name == "HGB":
        return HistGradientBoostingClassifier(
            learning_rate=params["learning_rate"],
            max_iter=params["max_iter"],
            max_leaf_nodes=params["max_leaf_nodes"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            l2_regularization=params["l2_regularization"],
            max_bins=params["max_bins"],
            random_state=seed,
        )

    if model_name == "SVM_RBF":
        return SVC(
            C=params["C"],
            gamma=params["gamma"],
            kernel="rbf",
            probability=False,
            random_state=seed,
        )

    if model_name == "XGB":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_child_weight=params["min_child_weight"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            gamma=params["gamma"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            random_state=seed,
            n_jobs=MODEL_N_JOBS,
            verbosity=0,
        )

    if model_name == "LGBM":
        return LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            max_depth=params["max_depth"],
            min_child_samples=params["min_child_samples"],
            colsample_bytree=params["colsample_bytree"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            subsample=params["subsample"],
            subsample_freq=params["subsample_freq"],
            random_state=seed,
            n_jobs=MODEL_N_JOBS,
            verbosity=-1,
        )

    if model_name == "CAT":
        cat_params = {
            "loss_function": "MultiClass",
            "iterations": params["iterations"],
            "depth": params["depth"],
            "learning_rate": params["learning_rate"],
            "l2_leaf_reg": params["l2_leaf_reg"],
            "random_strength": params["random_strength"],
            "bagging_temperature": params["bagging_temperature"],
            "border_count": params["border_count"],
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": MODEL_N_JOBS,
        }
        return CatBoostSklearnAdapter(
            params=cat_params,
        )

    raise KeyError(model_name)


def make_pipeline(feature_set, model_name, selector_percentile, params, seed):
    spec = FEATURE_SETS[feature_set]
    preprocessor = make_preprocessor(
        spec["numeric"],
        spec["categorical"],
        scale_numeric=model_requires_scaling(model_name),
    )

    selector = SelectPercentile(
        score_func=f_classif,
        percentile=int(selector_percentile),
    )

    model = make_estimator(
        model_name=model_name,
        params=params,
        seed=seed,
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("selector", selector),
        ("model", model),
    ])


def fit_pipeline(pipe, X, y, weight_mode):
    fit_kwargs = {}

    if weight_mode == "balanced":
        weights = compute_sample_weight(
            class_weight="balanced",
            y=y,
        )
        fit_kwargs["model__sample_weight"] = weights

    pipe.fit(X, y, **fit_kwargs)
    return pipe


def metrics_bundle(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        zero_division=0,
    )

    out = {
        "Macro_F1": float(
            f1_score(
                y_true,
                y_pred,
                labels=CLASS_ORDER,
                average="macro",
                zero_division=0,
            )
        ),
        "Balanced_Accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "Kappa": float(
            cohen_kappa_score(y_true, y_pred, labels=CLASS_ORDER)
        ),
        "Accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
    }

    for i, cls in enumerate(CLASS_ORDER):
        raw_cls = MODEL_TO_RAW[cls]
        out[f"Class{raw_cls}_Precision"] = float(precision[i])
        out[f"Class{raw_cls}_Recall"] = float(recall[i])
        out[f"Class{raw_cls}_F1"] = float(f1[i])
        out[f"Class{raw_cls}_Support"] = int(support[i])

    out["Worst_Class_F1"] = float(np.min(f1))
    return out


def build_inner_splits(train_df, seed):
    sgkf = StratifiedGroupKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    y = train_df["y_model"].to_numpy()
    groups = train_df["measurement_group"].astype(str).to_numpy()

    splits = list(
        sgkf.split(
            np.zeros((len(train_df), 1)),
            y,
            groups,
        )
    )

    for k, (itr, iva) in enumerate(splits, start=1):
        tr_y = y[itr]
        va_y = y[iva]
        tr_g = groups[itr]
        va_g = groups[iva]

        if set(tr_y) != {0, 1, 2}:
            raise RuntimeError(
                f"Inner{k}: train不含完整三类。"
            )
        if set(va_y) != {0, 1, 2}:
            raise RuntimeError(
                f"Inner{k}: validation不含完整三类。"
            )
        if set(tr_g) & set(va_g):
            raise RuntimeError(
                f"Inner{k}: measurement_group重叠。"
            )

    return splits


# ======================================================================
# 10. Optuna objective
# ======================================================================

def make_objective(
    train_df,
    feature_set,
    model_name,
    inner_splits,
    seed,
):
    feature_cols = (
        FEATURE_SETS[feature_set]["numeric"]
        + FEATURE_SETS[feature_set]["categorical"]
    )

    X_all = train_df[feature_cols].reset_index(drop=True)
    y_all = train_df["y_model"].to_numpy()

    def objective(trial):
        selector_percentile, weight_mode = (
            suggest_selector_and_weight(trial)
        )
        params = suggest_model_params(
            trial,
            model_name,
        )

        fold_scores = []
        fold_balacc = []
        fold_kappa = []
        fold_worst = []

        # 同一study内各trial使用相同模型随机种子，
        # 避免把trial.number混入random state造成额外随机自由度。
        trial_model_seed = int(seed)

        for fold_idx, (itr, iva) in enumerate(inner_splits, start=1):
            X_tr = X_all.iloc[itr]
            y_tr = y_all[itr]
            X_va = X_all.iloc[iva]
            y_va = y_all[iva]

            pipe = make_pipeline(
                feature_set=feature_set,
                model_name=model_name,
                selector_percentile=selector_percentile,
                params=params,
                seed=trial_model_seed,
            )

            fit_pipeline(
                pipe,
                X_tr,
                y_tr,
                weight_mode,
            )

            pred = pipe.predict(X_va).astype(int)
            m = metrics_bundle(y_va, pred)

            fold_scores.append(m["Macro_F1"])
            fold_balacc.append(m["Balanced_Accuracy"])
            fold_kappa.append(m["Kappa"])
            fold_worst.append(m["Worst_Class_F1"])

            running_mean = float(np.mean(fold_scores))
            trial.report(
                running_mean,
                step=fold_idx,
            )

            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr(
            "inner_macro_f1_sd",
            float(np.std(fold_scores, ddof=1))
            if len(fold_scores) > 1 else 0.0,
        )
        trial.set_user_attr(
            "inner_balacc_mean",
            float(np.mean(fold_balacc)),
        )
        trial.set_user_attr(
            "inner_kappa_mean",
            float(np.mean(fold_kappa)),
        )
        trial.set_user_attr(
            "inner_worst_class_f1_mean",
            float(np.mean(fold_worst)),
        )

        return float(np.mean(fold_scores))

    return objective


# ======================================================================
# 11. Trial恢复与best params
# ======================================================================

def completed_trial_count(study):
    return sum(
        t.state in {
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
        }
        for t in study.trials
    )


def recover_best_configuration(study, model_name):
    best = study.best_trial
    raw = dict(best.params)

    selector = int(raw.pop("selector_percentile"))
    weight_mode = raw.pop("weight_mode")

    # 还原条件参数的确定值
    if model_name in {"DT", "RF", "ET", "HGB", "LGBM"}:
        mode = raw.get("max_depth_mode")
        if mode == "none":
            raw["max_depth"] = -1 if model_name == "LGBM" else None
        raw.pop("max_depth_mode", None)

    if model_name == "LR":
        if raw.get("solver") == "lbfgs":
            raw["penalty"] = "l2"
            raw.pop("l1_ratio", None)

    if model_name in {"RF", "ET"}:
        if not bool(raw.get("bootstrap")):
            raw["max_samples"] = None

    if model_name == "LGBM":
        bag = raw.get("bagging_mode", "off")
        if bag == "off":
            raw["subsample"] = 1.0
            raw["subsample_freq"] = 0
        else:
            raw["subsample_freq"] = 1
        raw.pop("bagging_mode", None)

    return selector, weight_mode, raw, best


# ======================================================================
# 12. Nested LODO
# ======================================================================

def planned_trials_for_model(model_name):
    # 正式七模型版：所有模型统一80 trials/study。
    return N_TRIALS_PER_STUDY


def build_run_plan(data, smoke_test=False):
    plans = []

    if smoke_test:
        models = ["RF"]
        feature_sets = ["E", "EP"]
    else:
        models = ALL_MODELS
        feature_sets = FEATURE_SET_ORDER

    for season in ["Summer", "Winter"]:
        dates = sorted(pd.unique(data[season]["Date"]))

        if smoke_test:
            dates = dates[:1]

        for outer_fold, held_date in enumerate(dates, start=1):
            for feature_set in feature_sets:
                for model_name in models:
                    n_trials = 1 if smoke_test else planned_trials_for_model(model_name)
                    plans.append({
                        "Season": season,
                        "Outer_fold": outer_fold,
                        "Held_out_date": held_date,
                        "Feature_set": feature_set,
                        "Model": model_name,
                        "Planned_trials": n_trials,
                    })

    return plans


def run_nested_lodo(
    data,
    run_dir,
    db_path,
    smoke_test=False,
    save_models=False,
):
    plans = build_run_plan(
        data,
        smoke_test=smoke_test,
    )
    total_planned_trials = sum(p["Planned_trials"] for p in plans)
    tracker = GlobalProgress(total_planned_trials)

    trial_dir = run_dir / "optuna_trials"
    model_dir = run_dir / "models"
    trial_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    tuning_rows = []
    pred_rows = []

    print()
    print("=" * 100)
    print("11A运行计划")
    print("=" * 100)
    print(f"模型：{ALL_MODELS if not smoke_test else ['RF']}（正式固定7模型）")
    print(f"Feature sets：{FEATURE_SET_ORDER if not smoke_test else ['E','EP']}")
    print(f"Studies总数：{len(plans)}")
    print(f"计划新trial总数（首次运行）：{total_planned_trials}")
    print(
        "注意：如果DB中已有已完成trial，本次会自动减少需要新跑的trial；"
        "ETA会在运行时根据实际完成速度更新。"
    )
    print("=" * 100)

    study_done = 0

    for plan in plans:
        season = plan["Season"]
        outer_fold = plan["Outer_fold"]
        held_date = pd.Timestamp(plan["Held_out_date"])
        feature_set = plan["Feature_set"]
        model_name = plan["Model"]
        target_trials = int(plan["Planned_trials"])

        season_df = data[season]
        train_df = season_df.loc[
            season_df["Date"] != held_date
        ].copy()
        test_df = season_df.loc[
            season_df["Date"] == held_date
        ].copy()

        # Outer隔离检查
        train_groups = set(
            train_df["measurement_group"].astype(str)
        )
        test_groups = set(
            test_df["measurement_group"].astype(str)
        )
        overlap = len(train_groups & test_groups)

        if overlap:
            raise RuntimeError(
                f"{season} fold{outer_fold}: outer group overlap={overlap}"
            )

        if set(train_df["y_model"]) != {0, 1, 2}:
            raise RuntimeError(
                f"{season} fold{outer_fold}: outer train不含完整三类"
            )
        if set(test_df["y_model"]) != {0, 1, 2}:
            raise RuntimeError(
                f"{season} fold{outer_fold}: outer test不含完整三类"
            )

        # ------------------------------------------------------
        # 公平比较原则：
        # 同一 Season × Outer fold 下，所有 feature set 和所有模型
        # 共享完全相同的 inner SGKF 划分。
        # 这样E/P/EP和不同模型的inner证据是配对可比的。
        # 模型/Optuna本身仍使用独立且固定的study_seed。
        # ------------------------------------------------------
        split_seed = (
            RANDOM_STATE
            + (0 if season == "Summer" else 100000)
            + outer_fold * 1000
        )

        study_seed = (
            split_seed
            + FEATURE_SET_ORDER.index(feature_set) * 100
            + ALL_MODELS.index(model_name)
        )

        inner_splits = build_inner_splits(
            train_df.reset_index(drop=True),
            seed=split_seed,
        )

        date_tag = held_date.strftime("%Y%m%d")
        study_name = (
            f"12A__{season}__fold{outer_fold}__{date_tag}"
            f"__{feature_set}__{model_name}"
        )

        sampler = TPESampler(
            seed=study_seed,
            n_startup_trials=N_STARTUP_TRIALS,
            multivariate=True,
        )

        pruner = NopPruner()

        storage = f"sqlite:///{db_path.as_posix()}"

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        existing = completed_trial_count(study)
        remaining = max(target_trials - existing, 0)

        print()
        print("-" * 100)
        print(
            f"[{study_done+1}/{len(plans)} studies] "
            f"{season} | outer={outer_fold} | held-out={held_date.date()} "
            f"| {feature_set} | {model_name}"
        )
        print(
            f"目标trial={target_trials} | DB已有={existing} | 本次需新增={remaining}"
        )

        # 如果是resume，tracker总计划需要扣除已存在trial；
        # 第一次遇到study时把已有的数量视作“已完成”。
        if existing > 0:
            for _ in range(min(existing, target_trials)):
                tracker.add_trial(0.0)

        if remaining > 0:
            callback = StudyProgressCallback(
                tracker=tracker,
                planned_trials=remaining,
                initial_existing_trials=existing,
                label=(
                    f"{season}/F{outer_fold}/{feature_set}/{model_name}"
                ),
                print_every=(
                    1 if smoke_test else PROGRESS_EVERY_TRIALS
                ),
            )

            objective = make_objective(
                train_df=train_df.reset_index(drop=True),
                feature_set=feature_set,
                model_name=model_name,
                inner_splits=inner_splits,
                seed=study_seed,
            )

            study.optimize(
                objective,
                n_trials=remaining,
                callbacks=[callback],
                gc_after_trial=True,
                show_progress_bar=False,
                catch=(ValueError, FloatingPointError),
            )

        # 保存trial历史
        trials_df = study.trials_dataframe(
            attrs=(
                "number",
                "value",
                "datetime_start",
                "datetime_complete",
                "duration",
                "params",
                "user_attrs",
                "state",
            )
        )
        trials_csv = (
            trial_dir
            / f"{study_name}.csv"
        )
        trials_df.to_csv(
            trials_csv,
            index=False,
            encoding="utf-8-sig",
        )

        complete = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None
        ]
        if not complete:
            raise RuntimeError(
                f"{study_name}没有任何COMPLETE trial，无法outer预测。"
            )

        selector, weight_mode, best_params, best_trial = (
            recover_best_configuration(
                study,
                model_name,
            )
        )

        feature_cols = (
            FEATURE_SETS[feature_set]["numeric"]
            + FEATURE_SETS[feature_set]["categorical"]
        )

        X_train = train_df[feature_cols]
        y_train = train_df["y_model"].to_numpy()
        X_test = test_df[feature_cols]
        y_test = test_df["y_model"].to_numpy()

        final_pipe = make_pipeline(
            feature_set=feature_set,
            model_name=model_name,
            selector_percentile=selector,
            params=best_params,
            seed=study_seed,
        )

        t0 = time.time()
        fit_pipeline(
            final_pipe,
            X_train,
            y_train,
            weight_mode,
        )
        outer_fit_seconds = time.time() - t0

        y_pred = final_pipe.predict(X_test).astype(int)
        m = metrics_bundle(y_test, y_pred)

        fold_row = {
            "Season": season,
            "Outer_fold": outer_fold,
            "Held_out_date": held_date,
            "Feature_set": feature_set,
            "Model": model_name,
            "N_train": len(train_df),
            "N_test": len(test_df),
            "Train_groups": train_df["measurement_group"].nunique(),
            "Test_groups": test_df["measurement_group"].nunique(),
            "Group_overlap": overlap,
            "Best_inner_Macro_F1": float(best_trial.value),
            "Best_inner_Macro_F1_SD": float(
                best_trial.user_attrs.get(
                    "inner_macro_f1_sd", np.nan
                )
            ),
            "Best_inner_BalAcc": float(
                best_trial.user_attrs.get(
                    "inner_balacc_mean", np.nan
                )
            ),
            "Best_inner_Kappa": float(
                best_trial.user_attrs.get(
                    "inner_kappa_mean", np.nan
                )
            ),
            "Best_inner_Worst_Class_F1": float(
                best_trial.user_attrs.get(
                    "inner_worst_class_f1_mean", np.nan
                )
            ),
            "Best_trial_number": int(best_trial.number),
            "Selector_percentile": selector,
            "Weight_mode": weight_mode,
            "Outer_fit_seconds": outer_fit_seconds,
        }
        fold_row.update(m)
        fold_rows.append(fold_row)

        tuning_rows.append({
            "Season": season,
            "Outer_fold": outer_fold,
            "Held_out_date": held_date,
            "Feature_set": feature_set,
            "Model": model_name,
            "Study_name": study_name,
            "Planned_trials": target_trials,
            "Total_trials_in_DB": len(study.trials),
            "Complete_trials": len(complete),
            "Best_trial_number": int(best_trial.number),
            "Best_inner_Macro_F1": float(best_trial.value),
            "Best_inner_Macro_F1_SD": float(
                best_trial.user_attrs.get(
                    "inner_macro_f1_sd", np.nan
                )
            ),
            "Best_inner_BalAcc": float(
                best_trial.user_attrs.get(
                    "inner_balacc_mean", np.nan
                )
            ),
            "Best_inner_Kappa": float(
                best_trial.user_attrs.get(
                    "inner_kappa_mean", np.nan
                )
            ),
            "Best_inner_Worst_Class_F1": float(
                best_trial.user_attrs.get(
                    "inner_worst_class_f1_mean", np.nan
                )
            ),
            "Selector_percentile": selector,
            "Weight_mode": weight_mode,
            "Best_model_params_json": json.dumps(
                best_params,
                ensure_ascii=False,
                default=str,
            ),
        })

        for local_i, (_, r) in enumerate(
            test_df.reset_index(drop=True).iterrows()
        ):
            pred_rows.append({
                "Season": season,
                "Outer_fold": outer_fold,
                "Held_out_date": held_date,
                "Feature_set": feature_set,
                "Model": model_name,
                "row_id": int(r["row_id"]),
                "Date": r["Date"],
                "measurement_group": r["measurement_group"],
                "y_true": int(r["y_model"]),
                "y_true_raw": int(MODEL_TO_RAW[int(r["y_model"])]),
                "y_pred": int(y_pred[local_i]),
                "y_pred_raw": int(MODEL_TO_RAW[int(y_pred[local_i])]),
            })

        if save_models:
            joblib.dump(
                final_pipe,
                model_dir / f"{study_name}.joblib",
                compress=3,
            )

        study_done += 1
        print(
            f"[Study完成] outer Macro-F1={m['Macro_F1']:.4f} "
            f"| BalAcc={m['Balanced_Accuracy']:.4f} "
            f"| Kappa={m['Kappa']:.4f}"
        )
        tracker.print(prefix="[总体]")

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(tuning_rows),
        pd.DataFrame(pred_rows),
    )


# ======================================================================
# 13. OOF汇总
# ======================================================================

def build_oof_summary(fold_df, pred_df):
    summary_rows = []
    class_rows = []

    group_cols = [
        "Season",
        "Feature_set",
        "Model",
    ]

    for keys, g in pred_df.groupby(group_cols, sort=False):
        season, feature_set, model = keys

        y_true = g["y_true"].to_numpy(dtype=int)
        y_pred = g["y_pred"].to_numpy(dtype=int)

        expected_n = EXPECTED_N[season]
        if len(g) != expected_n:
            raise RuntimeError(
                f"{season}-{feature_set}-{model}: "
                f"OOF N={len(g)}，预期={expected_n}"
            )

        m = metrics_bundle(y_true, y_pred)

        folds = fold_df[
            (fold_df["Season"] == season)
            & (fold_df["Feature_set"] == feature_set)
            & (fold_df["Model"] == model)
        ].copy()

        summary_row = {
            "Season": season,
            "Feature_set": feature_set,
            "Model": model,
            "N_OOF": len(g),
            "Macro_F1": m["Macro_F1"],
            "Balanced_Accuracy": m["Balanced_Accuracy"],
            "Kappa": m["Kappa"],
            "Accuracy": m["Accuracy"],
            "Worst_Class_F1": m["Worst_Class_F1"],
            "Class1_F1": m["Class1_F1"],
            "Class2_F1": m["Class2_F1"],
            "Class3_F1": m["Class3_F1"],
            "Outer_fold_Macro_F1_mean": folds["Macro_F1"].mean(),
            "Outer_fold_Macro_F1_SD": folds["Macro_F1"].std(ddof=1),
            "Outer_fold_Macro_F1_min": folds["Macro_F1"].min(),
            "Outer_fold_Macro_F1_max": folds["Macro_F1"].max(),
            "Inner_best_Macro_F1_mean": folds["Best_inner_Macro_F1"].mean(),
            "Inner_best_Macro_F1_SD": folds["Best_inner_Macro_F1"].std(ddof=1),
        }
        summary_rows.append(summary_row)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=CLASS_ORDER,
            zero_division=0,
        )

        for i, cls in enumerate(CLASS_ORDER):
            raw_cls = MODEL_TO_RAW[cls]
            class_rows.append({
                "Season": season,
                "Feature_set": feature_set,
                "Model": model,
                "Class": raw_cls,
                "Precision": precision[i],
                "Recall": recall[i],
                "F1": f1[i],
                "Support": int(support[i]),
            })

    summary_df = pd.DataFrame(summary_rows)
    class_df = pd.DataFrame(class_rows)

    summary_df["Rank_overall_within_season"] = (
        summary_df.groupby("Season")["Macro_F1"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    summary_df["Rank_within_feature_set"] = (
        summary_df.groupby(["Season", "Feature_set"])["Macro_F1"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    summary_df = summary_df.sort_values(
        [
            "Season",
            "Rank_overall_within_season",
            "Feature_set",
            "Model",
        ]
    ).reset_index(drop=True)

    return summary_df, class_df


def build_feature_comparison(summary_df):
    """
    每季、每模型横向比较 E/P/EP。
    主要看EP-E和P-E，不用于outer反馈调参。
    """
    rows = []

    for season in ["Summer", "Winter"]:
        models = sorted(
            summary_df.loc[
                summary_df["Season"] == season,
                "Model",
            ].unique()
        )

        for model in models:
            sub = summary_df[
                (summary_df["Season"] == season)
                & (summary_df["Model"] == model)
            ].set_index("Feature_set")

            if not all(x in sub.index for x in ["E", "P", "EP"]):
                continue

            rows.append({
                "Season": season,
                "Model": model,
                "E_Macro_F1": sub.loc["E", "Macro_F1"],
                "P_Macro_F1": sub.loc["P", "Macro_F1"],
                "EP_Macro_F1": sub.loc["EP", "Macro_F1"],
                "Delta_EP_minus_E": (
                    sub.loc["EP", "Macro_F1"]
                    - sub.loc["E", "Macro_F1"]
                ),
                "Delta_P_minus_E": (
                    sub.loc["P", "Macro_F1"]
                    - sub.loc["E", "Macro_F1"]
                ),
            })

    return pd.DataFrame(rows)


def build_inner_family_table(tuning_df):
    """
    给12B提供inner-only证据。
    这里只汇总11A各outer task的best-inner表现。
    12B若缩小模型池/参数范围，应优先依据此表，
    而不是依据outer OOF高低进行“追分”。
    """
    rows = []

    for (
        season,
        feature_set,
        model,
    ), g in tuning_df.groupby(
        ["Season", "Feature_set", "Model"],
        sort=False,
    ):
        rows.append({
            "Season": season,
            "Feature_set": feature_set,
            "Model": model,
            "Inner_Macro_F1_mean_across_outer_tasks":
                g["Best_inner_Macro_F1"].mean(),
            "Inner_Macro_F1_SD_across_outer_tasks":
                g["Best_inner_Macro_F1"].std(ddof=1),
            "Inner_BalAcc_mean_across_outer_tasks":
                g["Best_inner_BalAcc"].mean(),
            "Inner_Kappa_mean_across_outer_tasks":
                g["Best_inner_Kappa"].mean(),
            "Inner_Worst_Class_F1_mean_across_outer_tasks":
                g["Best_inner_Worst_Class_F1"].mean(),
            "Outer_tasks_N": len(g),
            "Use_for_11B": (
                "YES: inner-only development evidence for 12B"
            ),
        })

    out = pd.DataFrame(rows)

    if not out.empty:
        out["Inner_rank_within_season_feature"] = (
            out.groupby(["Season", "Feature_set"])[
                "Inner_Macro_F1_mean_across_outer_tasks"
            ]
            .rank(method="min", ascending=False)
            .astype(int)
        )
        out = out.sort_values(
            [
                "Season",
                "Feature_set",
                "Inner_rank_within_season_feature",
            ]
        ).reset_index(drop=True)

    return out


def parameter_stability(tuning_df):
    rows = []

    for (
        season,
        feature_set,
        model,
    ), g in tuning_df.groupby(
        ["Season", "Feature_set", "Model"],
        sort=False,
    ):
        selectors = g["Selector_percentile"].tolist()
        weights = g["Weight_mode"].tolist()

        rows.append({
            "Season": season,
            "Feature_set": feature_set,
            "Model": model,
            "Outer_tasks_N": len(g),
            "Selector_values": " | ".join(map(str, selectors)),
            "Selector_mode": pd.Series(selectors).mode().iloc[0],
            "Weight_values": " | ".join(map(str, weights)),
            "Weight_mode": pd.Series(weights).mode().iloc[0],
            "Mean_best_inner_Macro_F1": g["Best_inner_Macro_F1"].mean(),
            "SD_best_inner_Macro_F1": g["Best_inner_Macro_F1"].std(ddof=1),
        })

    return pd.DataFrame(rows)


# ======================================================================
# 14. Excel输出
# ======================================================================

def model_availability_table():
    rows = []

    status_map = {
        "RF": "available (scikit-learn)",
        "ET": "available (scikit-learn)",
        "GB": "available (scikit-learn)",
        "HGB": "available (scikit-learn)",
        "XGB": OPTIONAL_MODEL_STATUS.get("XGB", "not available"),
        "LGBM": OPTIONAL_MODEL_STATUS.get("LGBM", "not available"),
        "CAT": OPTIONAL_MODEL_STATUS.get("CAT", "not available"),
    }

    for m in ALL_MODELS:
        rows.append({
            "Model": m,
            "Status": status_map[m],
            "Will_run": True,
            "Role": "11A broad-search candidate",
            "Trials_per_study": N_TRIALS_PER_STUDY,
        })

    return pd.DataFrame(rows)


def config_table(input_file, run_dir, smoke_test, save_models):
    return pd.DataFrame([
        {"Item": "Step", "Value": "12A Diagnostic Broad Bayesian Search (same logic as 11A)"},
        {"Item": "Input", "Value": str(input_file)},
        {"Item": "Target", "Value": TARGET},
        {"Item": "Target definition", "Value": "raw1-2 -> class1; raw3-4 -> class2; raw5 -> class3"},
        {"Item": "Seasons", "Value": "Summer / Winter separately"},
        {"Item": "Feature sets", "Value": "E / P / EP independent studies"},
        {"Item": "Outer CV", "Value": "4-fold Leave-One-Day-Out"},
        {"Item": "Inner CV", "Value": "3-fold StratifiedGroupKFold(groups=measurement_group)"},
        {"Item": "Inner split fairness", "Value": "Same inner SGKF splits shared by all E/P/EP feature sets and all model families within each Season × outer fold"},
        {"Item": "Primary metric", "Value": "Outer OOF Macro-F1"},
        {"Item": "Optuna objective", "Value": "mean inner 3-fold Macro-F1"},
        {"Item": "Sampler", "Value": "TPE; n_startup_trials=12"},
        {"Item": "Pruner", "Value": "NopPruner; every trial evaluates all 3 inner SGKF folds"},
        {"Item": "Trials/study", "Value": 1 if smoke_test else N_TRIALS_PER_STUDY},
        {"Item": "Models", "Value": ", ".join(ALL_MODELS if not smoke_test else ["RF"])},
        {"Item": "Formal model count", "Value": 7},
        {"Item": "Selector", "Value": "SelectPercentile f_classif: 50/65/80/100; fitted inside inner training fold"},
        {"Item": "Class weighting", "Value": "none / balanced; weights computed inside training fold only"},
        {"Item": "SMOTE", "Value": "NO"},
        {"Item": "Voting/Stacking", "Value": "NO"},
        {"Item": "Probability threshold correction", "Value": "NO"},
        {"Item": "Old 07 history warm-start", "Value": "NO; fresh independent 12A database"},
        {"Item": "Outer feedback tuning", "Value": "PROHIBITED"},
        {"Item": "LightGBM bagging", "Value": "explicit off/on; on => subsample_freq=1"},
        {"Item": "Resume", "Value": "YES; rerun same script resumes SQLite studies"},
        {"Item": "Save all outer models", "Value": bool(save_models)},
        {"Item": "Smoke test", "Value": bool(smoke_test)},
        {"Item": "Run directory", "Value": str(run_dir)},
    ])


def write_excel(
    output_file,
    config_df,
    availability_df,
    summary_df,
    fold_df,
    class_df,
    tuning_df,
    feature_cmp_df,
    inner_family_df,
    stability_df,
    pred_df,
):
    with pd.ExcelWriter(
        output_file,
        engine="openpyxl",
    ) as writer:
        config_df.to_excel(
            writer, sheet_name="00_Configuration", index=False
        )
        availability_df.to_excel(
            writer, sheet_name="01_Model_Availability", index=False
        )
        summary_df.to_excel(
            writer, sheet_name="02_OOF_Ranking", index=False
        )
        fold_df.to_excel(
            writer, sheet_name="03_Outer_Fold_Performance", index=False
        )
        class_df.to_excel(
            writer, sheet_name="04_Class_Performance", index=False
        )
        tuning_df.to_excel(
            writer, sheet_name="05_Best_Hyperparameters", index=False
        )
        feature_cmp_df.to_excel(
            writer, sheet_name="06_FeatureSet_Comparison", index=False
        )
        inner_family_df.to_excel(
            writer, sheet_name="07_12B_Inner_Evidence", index=False
        )
        stability_df.to_excel(
            writer, sheet_name="08_Parameter_Stability", index=False
        )
        pred_df.to_excel(
            writer, sheet_name="09_OOF_Predictions", index=False
        )

        wb = writer.book
        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

            for cell in ws[1]:
                cell.font = cell.font.copy(bold=True)

            # 适度列宽
            for col_cells in ws.columns:
                max_len = 0
                col_letter = col_cells[0].column_letter
                for cell in col_cells[:min(len(col_cells), 300)]:
                    v = "" if cell.value is None else str(cell.value)
                    max_len = max(max_len, len(v))
                ws.column_dimensions[col_letter].width = min(
                    max(max_len + 2, 10),
                    42,
                )


# ======================================================================
# 15. 主程序
# ======================================================================

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    args = parse_args()
    input_file = resolve_input(args.input)

    if args.output_dir:
        run_dir = Path(args.output_dir)
    elif args.smoke_test:
        # Smoke test必须与正式11A隔离，避免测试trial进入正式Optuna数据库。
        run_dir = input_file.parent / f"{RUN_DIR_NAME}_SMOKE_TEST"
    else:
        run_dir = input_file.parent / RUN_DIR_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    output_file = run_dir / OUTPUT_XLSX_NAME
    log_file = run_dir / LOG_NAME
    db_path = run_dir / DB_NAME
    config_json = run_dir / CONFIG_NAME

    save_models = bool(args.save_models or SAVE_OUTER_MODELS)

    with open(log_file, "a", encoding="utf-8") as log_fp:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(sys.stdout, log_fp)
        sys.stderr = Tee(sys.stderr, log_fp)

        try:
            print()
            print("=" * 100)
            print("12A 三级满意度 Broad Bayesian Search：严格Nested LODO（源数据未改诊断版）")
            print("=" * 100)
            print(f"开始时间：{datetime.now()}")
            print(f"输入：{input_file}")
            print(f"输出目录：{run_dir}")
            print(f"Optuna DB：{db_path}")
            print("Feature sets：E / P / EP（独立study）")
            print("Outer：4-fold Leave-One-Day-Out")
            print("Inner：3-fold StratifiedGroupKFold(groups=measurement_group)")
            print("正式主指标：完整4个held-out day拼接后的Outer OOF Macro-F1")
            print("旧07/07A/正式11A历史trial：均不读取；12A使用独立新DB")
            print("Voting / threshold correction / SMOTE：NO")
            print(f"Smoke test：{args.smoke_test}")
            print()

            print("模型可用性：")
            for _, r in model_availability_table().iterrows():
                print(
                    f"  {r['Model']}: {r['Status']} "
                    f"| will_run={r['Will_run']}"
                )

            data = load_data(input_file)

            print()
            print("数据检查：")
            for season in ["Summer", "Winter"]:
                print(
                    f"  {season}: N={len(data[season])}, "
                    f"days={data[season]['Date'].nunique()}, "
                    f"groups={data[season]['measurement_group'].nunique()}, "
                    f"class={data[season][TARGET].value_counts().sort_index().to_dict()}"
                )

            fold_df, tuning_df, pred_df = run_nested_lodo(
                data=data,
                run_dir=run_dir,
                db_path=db_path,
                smoke_test=args.smoke_test,
                save_models=save_models,
            )

            # Smoke test不覆盖全部4折，不能构建正式OOF排名
            if args.smoke_test:
                fold_df.to_excel(
                    run_dir / "12A_SMOKE_TEST_Outer_Results.xlsx",
                    index=False,
                )
                print()
                print("=" * 100)
                print("12A SMOKE TEST通过：代码能够完成Optuna + inner SGKF + outer预测。")
                print("不要使用smoke-test数值作为论文结果。")
                print("正式运行请去掉 --smoke-test。")
                print("=" * 100)
                return

            summary_df, class_df = build_oof_summary(
                fold_df,
                pred_df,
            )

            feature_cmp_df = build_feature_comparison(
                summary_df,
            )

            inner_family_df = build_inner_family_table(
                tuning_df,
            )

            stability_df = parameter_stability(
                tuning_df,
            )

            config_df = config_table(
                input_file=input_file,
                run_dir=run_dir,
                smoke_test=args.smoke_test,
                save_models=save_models,
            )

            availability_df = model_availability_table()

            write_excel(
                output_file=output_file,
                config_df=config_df,
                availability_df=availability_df,
                summary_df=summary_df,
                fold_df=fold_df,
                class_df=class_df,
                tuning_df=tuning_df,
                feature_cmp_df=feature_cmp_df,
                inner_family_df=inner_family_df,
                stability_df=stability_df,
                pred_df=pred_df,
            )

            config = {
                "input": str(input_file),
                "run_dir": str(run_dir),
                "models": ALL_MODELS,
                "feature_sets": FEATURE_SET_ORDER,
                "trials_per_study": N_TRIALS_PER_STUDY,
                "formal_model_count": 7,
                "outer": "4-fold LODO",
                "inner": "3-fold StratifiedGroupKFold; shared splits across models/features within same outer fold",
                "group": "measurement_group",
                "target": TARGET,
                "primary_metric": "Outer OOF Macro-F1",
                "old_history_warm_start": False,
                "run_role": "diagnostic rerun of 11A logic on source-data-unmodified dataset",
                "formal_results_replacement": False,
                "expected_groups": EXPECTED_GROUPS,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }

            with open(
                config_json,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    config,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print()
            print("=" * 100)
            print("12A 完成（源数据未改诊断复跑）")
            print("=" * 100)

            for season in ["Summer", "Winter"]:
                sub = (
                    summary_df[
                        summary_df["Season"] == season
                    ]
                    .sort_values(
                        "Rank_overall_within_season"
                    )
                    .head(10)
                )

                print()
                print(f"[{season}] Overall Top 10")
                for _, row in sub.iterrows():
                    print(
                        f"  #{int(row['Rank_overall_within_season'])} "
                        f"{row['Feature_set']}-{row['Model']}: "
                        f"Macro-F1={row['Macro_F1']:.4f} | "
                        f"BalAcc={row['Balanced_Accuracy']:.4f} | "
                        f"Kappa={row['Kappa']:.4f} | "
                        f"WorstClassF1={row['Worst_Class_F1']:.4f} | "
                        f"fold={row['Outer_fold_Macro_F1_mean']:.4f}"
                        f"±{row['Outer_fold_Macro_F1_SD']:.4f}"
                    )

            print()
            print("重要解释：")
            print("1) 02_OOF_Ranking用于描述泛化性能。")
            print("2) 07_12B_Inner_Evidence才是12B缩模型池/缩参数空间的主要开发证据。")
            print("3) 不允许因为某模型outer分数低而手工缩放参数去追held-out day。")
            print("4) 下一步12B必须读取本12A自己的Optuna DB，不能读取正式11A DB。")
            print()
            print(f"结果Excel：{output_file}")
            print(f"Optuna DB：{db_path}")
            print(f"结束时间：{datetime.now()}")
            print("=" * 100)

        except Exception:
            print()
            print("=" * 100)
            print("11A异常终止")
            print("=" * 100)
            traceback.print_exc()
            raise

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from src.config import (
    BASE_MODEL_FEATURES,
    RANDOM_STATE,
    RESAMPLING_STRATEGIES,
    SCOPE_COLUMN,
    TARGET_COLUMN,
    TEST_SIZE,
    VALIDATION_SIZE,
)


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def get_ticker_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("Ticker_")]


def get_model_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model feature columns, excluding target/result leakage fields."""
    return BASE_MODEL_FEATURES + get_ticker_feature_columns(df)


def filter_model_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "combined":
        return df.copy()
    if scope in {"intraday", "overnight"}:
        return df[df[SCOPE_COLUMN] == scope].copy()
    raise ValueError(f"Unknown model scope: {scope}")


def chronological_train_val_test_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    validation_size: float = VALIDATION_SIZE,
) -> SplitData:
    """Sort by signal time and split into train/validation/test without shuffling.

    This is one of the main leakage controls in the project. A random split would
    let the model train on future market/social regimes and then evaluate on
    earlier rows, which is not the real forecasting problem. The validation set
    is kept between train and test so model/resampling choices are made before
    looking at final held-out test performance.
    """
    ordered = df.sort_values(["Timestamp", "Ticker", SCOPE_COLUMN]).reset_index(drop=True)

    X = ordered[feature_cols]
    y = ordered[target_col].astype(int).to_numpy()

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    val_fraction_of_train_val = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_fraction_of_train_val, shuffle=False
    )

    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test)


def scale_split(split: SplitData) -> tuple[SplitData, StandardScaler]:
    """Fit StandardScaler on train only, then transform validation/test.

    The rubric calls out pre-split scaling as a leakage risk. Even though scaling
    does not use labels, fitting it on the full dataset would use future feature
    distributions. We therefore fit the scaler once on training rows and reuse it
    for validation, test, and dashboard inference.
    """
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(split.X_train),
        columns=split.X_train.columns,
        index=split.X_train.index,
    )
    X_val = pd.DataFrame(
        scaler.transform(split.X_val),
        columns=split.X_val.columns,
        index=split.X_val.index,
    )
    X_test = pd.DataFrame(
        scaler.transform(split.X_test),
        columns=split.X_test.columns,
        index=split.X_test.index,
    )
    return SplitData(X_train, X_val, X_test, split.y_train, split.y_val, split.y_test), scaler


def apply_resampling(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    strategy: str,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Apply train-only imbalance handling.

    Random upsampling and SMOTE are useful comparisons when class balance differs
    across time/scopes. They must be applied after splitting: duplicating or
    synthesizing observations before the split would leak training information
    into validation/test rows and inflate performance.
    """
    if strategy == "none":
        return X_train, y_train

    if strategy == "upsample":
        train_df = X_train.copy()
        train_df["_target"] = y_train
        counts = train_df["_target"].value_counts()
        max_count = counts.max()
        parts = []
        for target_value, group in train_df.groupby("_target"):
            parts.append(
                resample(
                    group,
                    replace=True,
                    n_samples=max_count,
                    random_state=random_state + int(target_value),
                )
            )
        balanced = pd.concat(parts).sample(frac=1, random_state=random_state)
        return balanced.drop(columns="_target"), balanced["_target"].to_numpy()

    if strategy == "smote":
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError as exc:
            raise ImportError(
                "SMOTE requires imbalanced-learn. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        smote = SMOTE(random_state=random_state, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return pd.DataFrame(X_resampled, columns=X_train.columns), y_resampled

    raise ValueError(f"Unknown resampling strategy: {strategy}")


def available_resampling_strategies() -> list[str]:
    strategies = []
    for strategy in RESAMPLING_STRATEGIES:
        if strategy != "smote":
            strategies.append(strategy)
            continue
        try:
            import imblearn  # noqa: F401
        except ImportError:
            continue
        strategies.append(strategy)
    return strategies


def base_model_specs(random_state: int = RANDOM_STATE) -> dict[str, object]:
    """Define the baseline and two more complex model families.

    Logistic Regression is the interpretable baseline. Random Forest and
    Histogram Gradient Boosting are included because they can model nonlinear
    interactions among ticker, sentiment, time, volume, and lag features. The
    final selection can still choose the simple baseline if validation evidence
    does not support extra complexity.
    """
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=200,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=random_state,
        ),
    }


def tuned_model_specs(random_state: int = RANDOM_STATE) -> dict[str, tuple[object, dict[str, list]]]:
    """Search spaces for the tree models.

    The grids are intentionally modest: they test depth/leaf-size/learning-rate
    tradeoffs that control overfitting without turning the project into an
    expensive brute-force search. Logistic Regression is not tuned here because
    it is used primarily as a stable baseline, while the rubric rewards tuning
    at least one stronger model family.
    """
    return {
        "random_forest_tuned": (
            RandomForestClassifier(
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
            {
                "n_estimators": [150, 250, 400],
                "max_depth": [6, 10, 14, None],
                "min_samples_leaf": [2, 5, 10],
                "max_features": ["sqrt", "log2", None],
            },
        ),
        "hist_gradient_boosting_tuned": (
            HistGradientBoostingClassifier(random_state=random_state),
            {
                "learning_rate": [0.03, 0.05, 0.08, 0.12],
                "max_iter": [100, 200, 300],
                "max_leaf_nodes": [15, 31, 63],
                "l2_regularization": [0.0, 0.05, 0.1, 0.5],
            },
        ),
    }


def run_randomized_search(
    estimator,
    param_distributions: dict[str, list],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_iter: int = 8,
    scoring: str = "f1",
    random_state: int = RANDOM_STATE,
):
    """Run randomized hyperparameter search with time-series cross-validation.

    RandomizedSearchCV gives a methodical tuning procedure without evaluating
    every parameter combination. TimeSeriesSplit keeps validation folds ordered
    in time, which better matches the forecasting objective than ordinary
    shuffled cross-validation.
    """
    cv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search

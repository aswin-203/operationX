import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "data/ml_large/final"
)

MODEL_DIR = Path(
    "models"
)

REPORT_DIR = Path(
    "reports"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# TARGET DATASETS
# ============================================================

TARGETS = {

    "pH": {
        "train": BASE / "pH_train.csv",
        "validation": BASE / "pH_validation.csv",
        "test": BASE / "pH_test.csv",
    },

    "temperature_kelvin": {
        "train": BASE / "temperature_kelvin_train.csv",
        "validation": BASE / "temperature_kelvin_validation.csv",
        "test": BASE / "temperature_kelvin_test.csv",
    },

    "matthews_coefficient": {
        "train": BASE / "matthews_coefficient_train.csv",
        "validation": BASE / "matthews_coefficient_validation.csv",
        "test": BASE / "matthews_coefficient_test.csv",
    },

    "solvent_percent": {
        "train": BASE / "solvent_percent_train.csv",
        "validation": BASE / "solvent_percent_validation.csv",
        "test": BASE / "solvent_percent_test.csv",
    },
}


# ============================================================
# CANDIDATE MODELS
# ============================================================

def get_models():

    return {

        # ----------------------------------------------------
        # Gradient Boosting
        # ----------------------------------------------------

        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=3,
            subsample=0.9,
            random_state=42,
            loss="huber",
        ),

        # ----------------------------------------------------
        # Gradient Boosting - squared error
        # ----------------------------------------------------

        "gradient_boosting_squared": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=3,
            subsample=0.9,
            random_state=42,
            loss="squared_error",
        ),

        # ----------------------------------------------------
        # Extra Trees
        # ----------------------------------------------------

        "extra_trees": ExtraTreesRegressor(
            n_estimators=400,
            max_features=0.8,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),

        # ----------------------------------------------------
        # Extra Trees - stronger regularization
        # ----------------------------------------------------

        "extra_trees_regularized": ExtraTreesRegressor(
            n_estimators=400,
            max_features=0.6,
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=42,
        ),

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        "random_forest": RandomForestRegressor(
            n_estimators=400,
            max_features=0.8,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),

        # ----------------------------------------------------
        # HistGradientBoosting
        # ----------------------------------------------------

        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=42,
        ),

        # ----------------------------------------------------
        # HistGradientBoosting - less regularization
        # ----------------------------------------------------

        "hist_gradient_boosting_flexible": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=15,
            l2_regularization=0.5,
            random_state=42,
        ),
    }


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_features(
    df,
    target,
    reference_columns=None,
):

    df = df.copy()

    # --------------------------------------------------------
    # Columns that must NOT be ML features.
    # --------------------------------------------------------

    drop_columns = [
        "pdb_id",
        "sequence_group",
        "status",
        target,
    ]

    X = df.drop(
        columns=[
            column
            for column in drop_columns
            if column in df.columns
        ],
        errors="ignore",
    )

    # --------------------------------------------------------
    # BLAST E-VALUE TRANSFORMATION
    # --------------------------------------------------------
    #
    # Convert:
    #
    # e-value -> -log10(e-value)
    #
    # Missing e-value indicates no usable BLAST hit.
    # blast_no_hit preserves that information.
    # --------------------------------------------------------

    if "best_evalue" in X.columns:

        evalue = pd.to_numeric(
            X["best_evalue"],
            errors="coerce",
        )

        no_hit = evalue.isna()

        safe_evalue = (
            evalue
            .fillna(1.0)
            .clip(lower=1e-300)
        )

        X["best_evalue"] = (
            -np.log10(
                safe_evalue
            )
        )

        if "blast_no_hit" not in X.columns:

            X["blast_no_hit"] = (
                no_hit.astype(int)
            )

        X.loc[
            no_hit,
            "best_evalue",
        ] = 0.0

    # --------------------------------------------------------
    # Convert everything to numeric.
    # --------------------------------------------------------

    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Replace infinity.
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # --------------------------------------------------------
    # TRAINING:
    #
    # Remove constant features.
    # --------------------------------------------------------

    if reference_columns is None:

        constant_columns = [
            column
            for column in X.columns
            if X[column].nunique(
                dropna=True
            ) <= 1
        ]

        if constant_columns:

            print()
            print(
                "Removing constant features:"
            )

            for column in constant_columns:

                print(
                    f"  - {column}"
                )

            X = X.drop(
                columns=constant_columns
            )

    else:

        # ----------------------------------------------------
        # Validation/test must use exactly the same features.
        # ----------------------------------------------------

        X = X.reindex(
            columns=reference_columns
        )

    return X


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


# ============================================================
# BASELINE
# ============================================================

def calculate_mean_baseline(
    y_train,
    y_test,
):

    mean_value = float(
        y_train.mean()
    )

    prediction = np.full(
        len(y_test),
        mean_value,
    )

    metrics = calculate_metrics(
        y_test,
        prediction,
    )

    metrics["prediction_value"] = (
        mean_value
    )

    return metrics


# ============================================================
# TRAIN ONE TARGET
# ============================================================

def train_target(
    target,
    paths,
):

    print()
    print("=" * 70)
    print(
        f"TARGET: {target}"
    )
    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    train_df = pd.read_csv(
        paths["train"]
    )

    validation_df = pd.read_csv(
        paths["validation"]
    )

    test_df = pd.read_csv(
        paths["test"]
    )

    print()
    print("DATASET")
    print("-" * 70)

    print(
        f"Train      : {len(train_df)}"
    )

    print(
        f"Validation : {len(validation_df)}"
    )

    print(
        f"Test       : {len(test_df)}"
    )

    # ========================================================
    # PREPARE FEATURES
    # ========================================================

    X_train = prepare_features(
        train_df,
        target,
    )

    feature_names = list(
        X_train.columns
    )

    X_validation = prepare_features(
        validation_df,
        target,
        reference_columns=feature_names,
    )

    X_test = prepare_features(
        test_df,
        target,
        reference_columns=feature_names,
    )

    # ========================================================
    # TARGETS
    # ========================================================

    y_train = pd.to_numeric(
        train_df[target],
        errors="coerce",
    )

    y_validation = pd.to_numeric(
        validation_df[target],
        errors="coerce",
    )

    y_test = pd.to_numeric(
        test_df[target],
        errors="coerce",
    )

    # ========================================================
    # SAFETY CHECKS
    # ========================================================

    if not y_train.notna().all():

        raise ValueError(
            f"Invalid training target values: {target}"
        )

    if not y_validation.notna().all():

        raise ValueError(
            f"Invalid validation target values: {target}"
        )

    if not y_test.notna().all():

        raise ValueError(
            f"Invalid test target values: {target}"
        )

    # ========================================================
    # FEATURE INFORMATION
    # ========================================================

    print()
    print(
        f"Final ML features: "
        f"{len(feature_names)}"
    )

    print()
    print("FEATURES")
    print("-" * 70)

    for index, feature in enumerate(
        feature_names,
        start=1,
    ):

        print(
            f"{index:2d}. {feature}"
        )

    # ========================================================
    # BASELINE
    # ========================================================

    baseline = calculate_mean_baseline(
        y_train,
        y_test,
    )

    print()
    print("MEAN BASELINE")
    print("-" * 70)

    print(
        f"MAE  : {baseline['MAE']:.4f}"
    )

    print(
        f"RMSE : {baseline['RMSE']:.4f}"
    )

    print(
        f"R2   : {baseline['R2']:.4f}"
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    print()
    print("CANDIDATE MODELS")
    print("-" * 70)

    validation_results = {}

    best_name = None
    best_model = None
    best_mae = float("inf")

    for model_name, model in get_models().items():

        pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),

                (
                    "model",
                    model,
                ),
            ]
        )

        print()
        print(
            f"Training {model_name}..."
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        validation_pred = (
            pipeline.predict(
                X_validation
            )
        )

        metrics = calculate_metrics(
            y_validation,
            validation_pred,
        )

        validation_results[
            model_name
        ] = metrics

        print(
            f"  MAE  : "
            f"{metrics['MAE']:.4f}"
        )

        print(
            f"  RMSE : "
            f"{metrics['RMSE']:.4f}"
        )

        print(
            f"  R2   : "
            f"{metrics['R2']:.4f}"
        )

        # ----------------------------------------------------
        # Select using validation MAE only.
        # ----------------------------------------------------

        if metrics["MAE"] < best_mae:

            best_mae = metrics["MAE"]

            best_name = model_name

            best_model = pipeline

    # ========================================================
    # SELECTED MODEL
    # ========================================================

    print()
    print("SELECTED MODEL")
    print("-" * 70)

    print(
        f"Model: {best_name}"
    )

    # ========================================================
    # TEST EVALUATION
    # ========================================================
    #
    # IMPORTANT:
    # Test data is used only once here.
    # ========================================================

    test_pred = best_model.predict(
        X_test
    )

    test_metrics = calculate_metrics(
        y_test,
        test_pred,
    )

    print()
    print("TEST PERFORMANCE")
    print("-" * 70)

    print(
        f"MAE  : "
        f"{test_metrics['MAE']:.4f}"
    )

    print(
        f"RMSE : "
        f"{test_metrics['RMSE']:.4f}"
    )

    print(
        f"R2   : "
        f"{test_metrics['R2']:.4f}"
    )

    # ========================================================
    # BASELINE COMPARISON
    # ========================================================

    print()
    print("BASELINE COMPARISON")
    print("-" * 70)

    mae_difference = (
        baseline["MAE"]
        -
        test_metrics["MAE"]
    )

    if mae_difference > 0:

        print(
            f"ML MAE improvement: "
            f"{mae_difference:.4f}"
        )

    else:

        print(
            f"ML MAE worse than baseline by: "
            f"{abs(mae_difference):.4f}"
        )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_filename = (
        target.lower()
        .replace(" ", "_")
        + "_model.joblib"
    )

    model_path = (
        MODEL_DIR /
        model_filename
    )

    joblib.dump(
        best_model,
        model_path,
    )

    print()
    print(
        f"Saved model: {model_path}"
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    feature_importance = []

    try:

        model_object = (
            best_model.named_steps[
                "model"
            ]
        )

        if hasattr(
            model_object,
            "feature_importances_",
        ):

            importance_values = (
                model_object
                .feature_importances_
            )

            for feature, importance in zip(
                feature_names,
                importance_values,
            ):

                feature_importance.append(
                    {
                        "feature": feature,

                        "importance":
                            float(
                                importance
                            ),
                    }
                )

            feature_importance.sort(
                key=lambda item:
                    item["importance"],
                reverse=True,
            )

            print()
            print(
                "TOP FEATURE IMPORTANCE"
            )

            print("-" * 70)

            for item in feature_importance[:15]:

                print(
                    f"{item['feature']:35s} "
                    f"{item['importance']:.6f}"
                )

    except Exception as error:

        print(
            "Could not calculate feature "
            f"importance: {error}"
        )

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "target": target,

        "train_records":
            len(train_df),

        "validation_records":
            len(validation_df),

        "test_records":
            len(test_df),

        "feature_count":
            len(feature_names),

        "feature_names":
            feature_names,

        "baseline":
            baseline,

        "selected_model":
            best_name,

        "validation":
            validation_results,

        "test":
            test_metrics,

        "feature_importance":
            feature_importance,

        "model_file":
            str(model_path),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "OPERATION X — ML MODEL TRAINING"
    )
    print("=" * 70)

    all_results = {}

    # ========================================================
    # TRAIN ALL FOUR TARGET MODELS
    # ========================================================

    for target, paths in TARGETS.items():

        result = train_target(
            target,
            paths,
        )

        all_results[target] = result

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report_path = (
        REPORT_DIR /
        "model_metrics.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            all_results,
            file,
            indent=2,
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "ALL MODELS TRAINED"
    )
    print("=" * 70)

    print()
    print(
        f"Metrics report: {report_path}"
    )

    print()

    print(
        f"{'TARGET':25s} "
        f"| {'MODEL':30s} "
        f"| {'MAE':10s} "
        f"| {'RMSE':10s} "
        f"| {'R2':10s}"
    )

    print("-" * 105)

    for target, result in all_results.items():

        metrics = result["test"]

        print(
            f"{target:25s} "
            f"| {result['selected_model']:30s} "
            f"| {metrics['MAE']:<10.4f} "
            f"| {metrics['RMSE']:<10.4f} "
            f"| {metrics['R2']:<10.4f}"
        )

    print()

    print(
        "MEAN BASELINE vs ML"
    )

    print("-" * 105)

    for target, result in all_results.items():

        baseline_mae = (
            result["baseline"]["MAE"]
        )

        ml_mae = (
            result["test"]["MAE"]
        )

        difference = (
            baseline_mae -
            ml_mae
        )

        print(
            f"{target:25s} "
            f"| baseline MAE={baseline_mae:.4f} "
            f"| ML MAE={ml_mae:.4f} "
            f"| difference={difference:+.4f}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
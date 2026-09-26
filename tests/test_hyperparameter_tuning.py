import pandas as pd
import pytest

from operation_x.ml.hyperparameter_tuning import (
    HyperparameterTuner,
)


def test_hyperparameter_tuner(tmp_path):

    data_file = tmp_path / "dataset.csv"

    rows = []

    for i in range(30):
        rows.append(
            {
                "query_pdb": f"PDB{i}",
                "feature_a": float(i),
                "feature_b": float(i % 5),
                "feature_c": float(i * 0.5),
                "pH": 5.0 + (i * 0.1),
            }
        )

    pd.DataFrame(rows).to_csv(
        data_file,
        index=False,
    )

    tuner = HyperparameterTuner(
        data_file=str(data_file),
        target="pH",
        model="random_forest",
        feature_count=3,
        n_splits=3,
        n_iter=2,
        random_state=42,
    )

    result = tuner.evaluate()

    assert result["target"] == "pH"
    assert result["model"] == "random_forest"
    assert result["rows"] == 30
    assert result["feature_count"] == 3

    assert "default" in result
    assert "tuned" in result

    assert "mae" in result["default"]
    assert "rmse" in result["default"]
    assert "r2" in result["default"]

    assert "mae" in result["tuned"]
    assert "rmse" in result["tuned"]
    assert "r2" in result["tuned"]

    assert "best_params" in result


def test_invalid_model(tmp_path):

    data_file = tmp_path / "dataset.csv"

    pd.DataFrame(
        {
            "query_pdb": ["A", "B", "C", "D"],
            "feature_a": [1, 2, 3, 4],
            "pH": [5, 6, 7, 8],
        }
    ).to_csv(
        data_file,
        index=False,
    )

    with pytest.raises(ValueError):

        HyperparameterTuner(
            data_file=str(data_file),
            target="pH",
            model="invalid_model",
            feature_count=1,
        )
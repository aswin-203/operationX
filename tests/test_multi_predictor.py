import pandas as pd
import pytest

from operation_x.ml.multi_predictor import (
    OperationXMultiPredictor,
)


TARGETS = [
    "pH",
    "temperature_kelvin",
    "matthews_coefficient",
    "solvent_percent",
]


def create_dataset(path, target):
    df = pd.DataFrame({
        "query_pdb": [f"PDB{i}" for i in range(20)],
        "feature_a": [float(i) for i in range(20)],
        "feature_b": [float(i * 2) for i in range(20)],
        "feature_c": [float(i * 3) for i in range(20)],
        target: [float(i) + 1.0 for i in range(20)],
    })

    df.to_csv(path, index=False)


def test_multi_predictor_fit(tmp_path):

    data_directory = tmp_path / "data"
    data_directory.mkdir()

    for target in TARGETS:
        create_dataset(
            data_directory / f"{target}_combined.csv",
            target,
        )

    predictor = OperationXMultiPredictor(
        data_directory=str(data_directory),
        model_name="gradient_boosting",
    )

    result = predictor.fit()

    assert result["targets"] == TARGETS

    for target in TARGETS:
        assert result["results"][target]["rows"] == 20
        assert result["results"][target]["feature_count"] == 3


def test_multi_predictor_returns_all_predictions(tmp_path):

    data_directory = tmp_path / "data"
    data_directory.mkdir()

    for target in TARGETS:
        create_dataset(
            data_directory / f"{target}_combined.csv",
            target,
        )

    predictor = OperationXMultiPredictor(
        data_directory=str(data_directory),
        model_name="gradient_boosting",
    )

    predictor.fit()

    result = predictor.predict({
        "feature_a": 10.0,
        "feature_b": 20.0,
        "feature_c": 30.0,
    })

    assert set(result.keys()) == set(TARGETS)

    for target in TARGETS:
        assert isinstance(result[target], float)


def test_multi_predictor_requires_fit(tmp_path):

    data_directory = tmp_path / "data"
    data_directory.mkdir()

    for target in TARGETS:
        create_dataset(
            data_directory / f"{target}_combined.csv",
            target,
        )

    predictor = OperationXMultiPredictor(
        data_directory=str(data_directory),
        model_name="gradient_boosting",
    )

    with pytest.raises(RuntimeError):
        predictor.predict({
            "feature_a": 10.0,
            "feature_b": 20.0,
            "feature_c": 30.0,
        })


def test_multi_predictor_detects_missing_features(tmp_path):

    data_directory = tmp_path / "data"
    data_directory.mkdir()

    for target in TARGETS:
        create_dataset(
            data_directory / f"{target}_combined.csv",
            target,
        )

    predictor = OperationXMultiPredictor(
        data_directory=str(data_directory),
        model_name="gradient_boosting",
    )

    predictor.fit()

    with pytest.raises(ValueError):
        predictor.predict({
            "feature_a": 10.0,
        })

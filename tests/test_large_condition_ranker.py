import pandas as pd

from operation_x.ml.crystallization_condition_ranker import (
    CrystallizationConditionRanker,
)


class FakeLargeDataset:

    def load(self):

        return pd.DataFrame(
            [
                {
                    "query_pdb": "10AF",
                    "crystallization_details": (
                        "3.0 M AMMONIUM SULFATE, "
                        "20 MM TRIS, 1MM EDTA, PH 9.0"
                    ),
                    "pH": 9.0,
                    "temperature_kelvin": 293.0,
                    "matthews_coefficient": 3.10,
                    "solvent_percent": 59.0,
                },
                {
                    "query_pdb": "10AH",
                    "crystallization_details": (
                        "2.0 M AMMONIUM SULFATE, PH 7.0"
                    ),
                    "pH": 7.0,
                    "temperature_kelvin": 277.0,
                    "matthews_coefficient": 2.05,
                    "solvent_percent": 40.0,
                },
                {
                    "query_pdb": "10AI",
                    "crystallization_details": (
                        "1.5 M PEG, PH 6.0"
                    ),
                    "pH": 6.0,
                    "temperature_kelvin": 298.0,
                    "matthews_coefficient": 2.0,
                    "solvent_percent": 40.0,
                },
            ]
        )


def test_large_dataset_ranker_returns_results():

    ranker = CrystallizationConditionRanker(
        FakeLargeDataset()
    )

    results = ranker.rank(
        predicted_ph=8.82,
        predicted_temperature=294.95,
        predicted_matthews=3.11,
        predicted_solvent=59.07,
        top_n=3,
    )

    assert len(results) == 3


def test_best_condition_is_ranked_first():

    ranker = CrystallizationConditionRanker(
        FakeLargeDataset()
    )

    results = ranker.rank(
        predicted_ph=8.82,
        predicted_temperature=294.95,
        predicted_matthews=3.11,
        predicted_solvent=59.07,
        top_n=3,
    )

    assert results[0]["query_pdb"] == "10AF"


def test_results_have_numeric_scores():

    ranker = CrystallizationConditionRanker(
        FakeLargeDataset()
    )

    results = ranker.rank(
        predicted_ph=8.82,
        predicted_temperature=294.95,
        predicted_matthews=3.11,
        predicted_solvent=59.07,
        top_n=3,
    )

    for result in results:
        assert isinstance(
            result["score"],
            float,
        )

        assert result["score"] >= 0.0
        assert result["score"] <= 1.0

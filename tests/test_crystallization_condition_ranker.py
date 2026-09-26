import pandas as pd

from operation_x.ml.crystallization_condition_ranker import (
    CrystallizationConditionRanker,
)


class FakeDataset:

    def load(self):

        return pd.DataFrame(
            [
                {
                    "query_pdb": "101M",
                    "condition_text": (
                        "3.0 M AMMONIUM SULFATE, "
                        "20 MM TRIS, 1MM EDTA, PH 9.0"
                    ),
                    "precipitant": "AMMONIUM SULFATE",
                    "buffer": "TRIS",
                    "pH": 9.0,
                    "temperature_kelvin": 294.0,
                    "matthews_coefficient": 3.09,
                    "solvent_percent": 60.2,
                },
                {
                    "query_pdb": "TEST",
                    "condition_text": (
                        "0.1 M HEPES, 20% PEG 3350"
                    ),
                    "precipitant": "PEG 3350",
                    "buffer": "HEPES",
                    "pH": 7.0,
                    "temperature_kelvin": 277.0,
                    "matthews_coefficient": 2.0,
                    "solvent_percent": 40.0,
                },
            ]
        )


def test_rank_returns_conditions():

    ranker = CrystallizationConditionRanker(
        FakeDataset()
    )

    result = ranker.rank(
        predicted_ph=8.82,
        predicted_temperature=294.95,
        predicted_matthews=3.11,
        predicted_solvent=59.07,
        top_n=2,
    )

    assert len(result) == 2


def test_best_matching_condition_is_first():

    ranker = CrystallizationConditionRanker(
        FakeDataset()
    )

    result = ranker.rank(
        predicted_ph=8.82,
        predicted_temperature=294.95,
        predicted_matthews=3.11,
        predicted_solvent=59.07,
        top_n=2,
    )

    assert result[0]["query_pdb"] == "101M"


def test_scores_are_sorted():

    ranker = CrystallizationConditionRanker(
        FakeDataset()
    )

    result = ranker.rank(
        predicted_ph=8.82,
        predicted_temperature=294.95,
        predicted_matthews=3.11,
        predicted_solvent=59.07,
        top_n=2,
    )

    assert (
        result[0]["score"]
        >= result[1]["score"]
    )

import math

import pandas as pd


class CrystallizationConditionRanker:

    def __init__(self, dataset):
        self.dataset = dataset

    @staticmethod
    def _distance(
        predicted_value,
        condition_value,
        scale,
    ):
        if (
            predicted_value is None
            or condition_value is None
        ):
            return None

        try:
            predicted_value = float(predicted_value)
            condition_value = float(condition_value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(predicted_value):
            return None

        if not math.isfinite(condition_value):
            return None

        return abs(
            predicted_value - condition_value
        ) / scale
    @staticmethod
    def _similarity(distance):
        if distance is None:
            return 0.0

        return math.exp(-distance)

    def rank(
        self,
        predicted_ph,
        predicted_temperature,
        predicted_matthews,
        predicted_solvent,
        top_n=5,
    ):

        df = self.dataset.load()

        if df.empty:
            return []

        rows = []

        weights = {
            "pH": 0.40,
            "temperature_kelvin": 0.20,
            "matthews_coefficient": 0.20,
            "solvent_percent": 0.20,
        }

        predictions = {
            "pH": predicted_ph,
            "temperature_kelvin": predicted_temperature,
            "matthews_coefficient": predicted_matthews,
            "solvent_percent": predicted_solvent,
        }

        scales = {
            "pH": 2.0,
            "temperature_kelvin": 20.0,
            "matthews_coefficient": 2.0,
            "solvent_percent": 20.0,
        }

        for _, row in df.iterrows():

            weighted_score = 0.0
            total_weight = 0.0

            for feature, weight in weights.items():

                distance = self._distance(
                    predictions[feature],
                    row.get(feature),
                    scales[feature],
                )

                if distance is None:
                    continue

                similarity = self._similarity(distance)

                weighted_score += (
                    weight * similarity
                )

                total_weight += weight

            if total_weight == 0:
                continue

            score = (
                weighted_score / total_weight
            )

            rows.append(
                {
                    "query_pdb": row["query_pdb"],
                    "condition_text": row.get(
                        "condition_text",
                        row.get("crystallization_details"),
                    ),
                    "precipitant": row.get(
                        "precipitant"
                    ),
                    "buffer": row.get("buffer"),
                    "pH": row.get("pH"),
                    "temperature_kelvin": row.get(
                        "temperature_kelvin"
                    ),
                    "matthews_coefficient": row.get(
                        "matthews_coefficient"
                    ),
                    "solvent_percent": row.get(
                        "solvent_percent"
                    ),
                    "score": score,
                }
            )
        result = pd.DataFrame(rows)

        result = result.sort_values(
            "score",
            ascending=False,
        )

        return result.head(top_n).to_dict(
            "records"
        )

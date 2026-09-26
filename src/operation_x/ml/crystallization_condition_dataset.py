from pathlib import Path

import pandas as pd

from operation_x.ml.crystallization_condition_parser import (
    CrystallizationConditionParser,
)


class CrystallizationConditionDataset:

    def __init__(self, data_file: str):
        self.data_file = Path(data_file)
        self.parser = CrystallizationConditionParser()

    def load(self) -> pd.DataFrame:

        if not self.data_file.exists():
            raise FileNotFoundError(self.data_file)

        df = pd.read_csv(self.data_file)

        required_columns = [
            "query_pdb",
            "crystallization_details",
            "pH",
            "temperature_kelvin",
            "matthews_coefficient",
            "solvent_percent",
        ]

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                "Missing required columns: "
                + ", ".join(missing)
            )

        records = []

        for _, row in df.iterrows():

            condition_text = str(
                row["crystallization_details"]
            ).strip()

            parsed = self.parser.parse(
                condition_text
            )

            record = {
                "query_pdb": row["query_pdb"],
                "condition_text": condition_text,

                "pH": row["pH"],
                "temperature_kelvin": (
                    row["temperature_kelvin"]
                ),
                "matthews_coefficient": (
                    row["matthews_coefficient"]
                ),
                "solvent_percent": (
                    row["solvent_percent"]
                ),

                "condition_parse_status": (
                    parsed["parse_status"]
                ),

                "protein_concentration": (
                    parsed["protein_concentration"]
                ),

                "inhibitor_concentration": (
                    parsed["inhibitor_concentration"]
                ),

                "inhibitor_unit": (
                    parsed["inhibitor_unit"]
                ),

                "precipitant": (
                    parsed["precipitant"]
                ),

                "precipitant_concentration": (
                    parsed[
                        "precipitant_concentration"
                    ]
                ),

                "precipitant_unit": (
                    parsed["precipitant_unit"]
                ),

                "buffer": parsed["buffer"],

                "buffer_concentration": (
                    parsed["buffer_concentration"]
                ),

                "buffer_unit": (
                    parsed["buffer_unit"]
                ),

                "additives": str(
                    parsed["additives"]
                ),
            }

            records.append(record)

        result = pd.DataFrame(records)

        # Stable condition identifier based on the
        # normalized original condition text.
        condition_map = {
            text: f"condition_{index + 1:03d}"
            for index, text in enumerate(
                sorted(
                    result["condition_text"].unique()
                )
            )
        }

        result["condition_id"] = (
            result["condition_text"]
            .map(condition_map)
        )

        return result

    def save(
        self,
        output_file: str,
    ) -> pd.DataFrame:

        result = self.load()

        output_path = Path(output_file)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        result.to_csv(
            output_path,
            index=False,
        )

        return result

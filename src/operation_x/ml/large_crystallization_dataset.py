import json
from pathlib import Path

import pandas as pd

from operation_x.ml.crystallization_condition_parser import (
    CrystallizationConditionParser,
)


class LargeCrystallizationDataset:

    COLUMNS = [
        "query_pdb",
        "sequence",
        "condition_text",
        "condition_parse_status",
        "precipitant",
        "precipitant_concentration",
        "precipitant_unit",
        "buffer",
        "buffer_concentration",
        "buffer_unit",
        "pH",
        "temperature_kelvin",
        "matthews_coefficient",
        "solvent_percent",
    ]

    def __init__(self, data_file: str):
        self.data_file = Path(data_file)
        self.parser = CrystallizationConditionParser()

    def load(self) -> pd.DataFrame:

        if not self.data_file.exists():
            raise FileNotFoundError(
                self.data_file
            )

        with self.data_file.open() as f:
            records = json.load(f)

        if not isinstance(records, dict):
            raise ValueError(
                "Large crystallization dataset must be a dictionary"
            )

        rows = []

        for pdb_id, record in records.items():

            if not isinstance(record, dict):
                continue

            condition_text = record.get(
                "crystallization_details"
            )

            parsed = {
                "parse_status": "unparsed",
                "precipitant": None,
                "precipitant_concentration": None,
                "precipitant_unit": None,
                "buffer": None,
                "buffer_concentration": None,
                "buffer_unit": None,
            }

            if isinstance(condition_text, str):
                try:
                    parsed = self.parser.parse(
                        condition_text
                    )
                except (TypeError, ValueError):
                    pass

            rows.append(
                {
                    "query_pdb": record.get(
                        "pdb_id",
                        pdb_id,
                    ),
                    "sequence": record.get(
                        "sequence"
                    ),
                    "condition_text": condition_text,
                    "condition_parse_status": parsed.get(
                        "parse_status",
                        "unparsed",
                    ),
                    "precipitant": parsed.get(
                        "precipitant"
                    ),
                    "precipitant_concentration": parsed.get(
                        "precipitant_concentration"
                    ),
                    "precipitant_unit": parsed.get(
                        "precipitant_unit"
                    ),
                    "buffer": parsed.get(
                        "buffer"
                    ),
                    "buffer_concentration": parsed.get(
                        "buffer_concentration"
                    ),
                    "buffer_unit": parsed.get(
                        "buffer_unit"
                    ),
                    "pH": record.get("pH"),
                    "temperature_kelvin": record.get(
                        "temperature_kelvin"
                    ),
                    "matthews_coefficient": record.get(
                        "matthews_coefficient"
                    ),
                    "solvent_percent": record.get(
                        "solvent_percent"
                    ),
                }
            )

        return pd.DataFrame(
            rows,
            columns=self.COLUMNS,
        )

    @staticmethod
    def unique_conditions(
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if "condition_text" not in df.columns:
            raise ValueError(
                "Dataset must contain condition_text"
            )

        columns = [
            "condition_text",
            "condition_parse_status",
            "precipitant",
            "precipitant_concentration",
            "precipitant_unit",
            "buffer",
            "buffer_concentration",
            "buffer_unit",
            "pH",
            "temperature_kelvin",
            "matthews_coefficient",
            "solvent_percent",
        ]

        available = [
            column
            for column in columns
            if column in df.columns
        ]

        return (
            df[
                df["condition_text"].notna()
            ][available]
            .drop_duplicates(
                subset=["condition_text"]
            )
            .reset_index(drop=True)
        )
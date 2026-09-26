from pathlib import Path
import json

import pandas as pd


class LargeFeatureBuilder:
    """
    Build target-specific ML datasets from the large
    Operation X similarity dataset.

    Each target is handled independently so that records
    missing one target are not discarded from the other
    target datasets.
    """

    TARGETS = [
        "pH",
        "temperature_kelvin",
        "matthews_coefficient",
        "solvent_percent",
    ]

    FEATURE_COLUMNS = [
        "best_identity",
        "best_coverage",
        "best_evalue",
        "best_bitscore",
        "sequence_strong_hit_count",
        "best_query_tm_score",
        "best_target_tm_score",
        "best_rmsd",
        "best_alignment_length",
        "structure_strong_hit_count",
        "domain_count",
        "pfam_count",
        "interpro_count",
    ]

    def __init__(
        self,
        input_file: str,
        output_directory: str = "data/ml_large",
    ):
        self.input_file = Path(input_file)
        self.output_directory = Path(output_directory)

        if not self.input_file.exists():
            raise FileNotFoundError(
                f"Input file not found: {self.input_file}"
            )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def load(self) -> list[dict]:
        with self.input_file.open() as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(
                "Similarity dataset must contain a list"
            )

        return data

    @staticmethod
    def _extract_features(record: dict) -> dict:
        evidence = record.get("evidence", {})

        sequence = evidence.get(
            "sequence_evidence",
            {},
        )

        structure = evidence.get(
            "structure_evidence",
            {},
        )

        domain = evidence.get(
            "domain_evidence",
            {},
        )

        return {
            "best_identity": sequence.get(
                "best_identity"
            ),
            "best_coverage": sequence.get(
                "best_coverage"
            ),
            "best_evalue": sequence.get(
                "best_evalue"
            ),
            "best_bitscore": sequence.get(
                "best_bitscore"
            ),
            "sequence_strong_hit_count": sequence.get(
                "strong_hit_count"
            ),
            "best_query_tm_score": structure.get(
                "best_query_tm_score"
            ),
            "best_target_tm_score": structure.get(
                "best_target_tm_score"
            ),
            "best_rmsd": structure.get(
                "best_rmsd"
            ),
            "best_alignment_length": structure.get(
                "best_alignment_length"
            ),
            "structure_strong_hit_count": structure.get(
                "strong_hit_count"
            ),
            "domain_count": domain.get(
                "domain_count"
            ),
            "pfam_count": domain.get(
                "pfam_count"
            ),
            "interpro_count": domain.get(
                "interpro_count"
            ),
        }

    def build_target(self, data: list[dict], target: str):
        rows = []
        metadata = []

        for record in data:
            crystallization = record.get(
                "crystallization",
                {},
            )

            target_value = crystallization.get(
                target
            )

            # Skip only this target if its label is missing.
            if target_value is None:
                continue

            features = self._extract_features(record)

            # Require all feature values to be present.
            if any(
                features[column] is None
                for column in self.FEATURE_COLUMNS
            ):
                continue

            rows.append(
                {
                    column: features[column]
                    for column in self.FEATURE_COLUMNS
                }
            )

            metadata.append(
                {
                    "query_pdb": record.get(
                        "pdb_id"
                    ),
                    "target": target,
                }
            )

        features_df = pd.DataFrame(
            rows,
            columns=self.FEATURE_COLUMNS,
        )

        targets_df = pd.DataFrame(
            {target: [
                data_record
                .get("crystallization", {})
                .get(target)
                for data_record in data
                if data_record
                .get("crystallization", {})
                .get(target) is not None
                and not any(
                    self._extract_features(data_record)[column]
                    is None
                    for column in self.FEATURE_COLUMNS
                )
            ]}
        )

        metadata_df = pd.DataFrame(
            metadata,
            columns=[
                "query_pdb",
                "target",
            ],
        )

        return (
            features_df,
            targets_df,
            metadata_df,
        )

    def save(self) -> dict:
        data = self.load()

        results = {}

        for target in self.TARGETS:
            (
                features,
                targets,
                metadata,
            ) = self.build_target(
                data,
                target,
            )

            safe_name = target.lower()

            features_file = (
                self.output_directory
                / f"{safe_name}_features.csv"
            )

            targets_file = (
                self.output_directory
                / f"{safe_name}_targets.csv"
            )

            metadata_file = (
                self.output_directory
                / f"{safe_name}_metadata.csv"
            )

            features.to_csv(
                features_file,
                index=False,
            )

            targets.to_csv(
                targets_file,
                index=False,
            )

            metadata.to_csv(
                metadata_file,
                index=False,
            )

            results[target] = {
                "features": str(features_file),
                "targets": str(targets_file),
                "metadata": str(metadata_file),
                "rows": len(features),
                "feature_count": len(
                    features.columns
                ),
            }

        return results

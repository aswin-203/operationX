from pathlib import Path

import pandas as pd


class CombinedFeatureBuilder:

    TARGETS = [
        "pH",
        "temperature_kelvin",
        "matthews_coefficient",
        "solvent_percent",
    ]

    FILE_PREFIXES = {
        "pH": "ph",
        "temperature_kelvin": "temperature_kelvin",
        "matthews_coefficient": "matthews_coefficient",
        "solvent_percent": "solvent_percent",
    }

    def __init__(
        self,
        similarity_features_directory: str,
        physicochemical_features_file: str,
        targets_directory: str,
        metadata_directory: str,
        output_directory: str,
    ):
        self.similarity_features_directory = Path(
            similarity_features_directory
        )

        self.physicochemical_features_file = Path(
            physicochemical_features_file
        )

        self.targets_directory = Path(
            targets_directory
        )

        self.metadata_directory = Path(
            metadata_directory
        )

        self.output_directory = Path(
            output_directory
        )

    def _load_similarity_features(
        self,
        target: str,
    ) -> pd.DataFrame:

        prefix = self.FILE_PREFIXES[target]

        similarity_file = (
            self.similarity_features_directory
            / f"{prefix}_features.csv"
        )

        if not similarity_file.exists():
            raise FileNotFoundError(
                similarity_file
            )

        return pd.read_csv(
            similarity_file
        )

    def _load_physicochemical_features(
        self,
    ) -> pd.DataFrame:

        if not self.physicochemical_features_file.exists():
            raise FileNotFoundError(
                self.physicochemical_features_file
            )

        physicochemical = pd.read_csv(
            self.physicochemical_features_file
        )

        if "pdb_id" not in physicochemical.columns:
            raise ValueError(
                "Physicochemical dataset must contain pdb_id"
            )

        return physicochemical

    def _load_target_data(
        self,
        target: str,
    ):

        prefix = self.FILE_PREFIXES[target]

        targets_file = (
            self.targets_directory
            / f"{prefix}_targets.csv"
        )

        metadata_file = (
            self.metadata_directory
            / f"{prefix}_metadata.csv"
        )

        if not targets_file.exists():
            raise FileNotFoundError(
                targets_file
            )

        if not metadata_file.exists():
            raise FileNotFoundError(
                metadata_file
            )

        targets = pd.read_csv(
            targets_file
        )

        metadata = pd.read_csv(
            metadata_file
        )

        return targets, metadata

    def build_target(
        self,
        target: str,
    ) -> pd.DataFrame:

        if target not in self.TARGETS:
            raise ValueError(
                f"Unsupported target: {target}"
            )

        similarity = self._load_similarity_features(
            target
        )

        physicochemical = (
            self._load_physicochemical_features()
        )

        targets, metadata = (
            self._load_target_data(target)
        )

        # --------------------------------------------------
        # Validate input row counts
        # --------------------------------------------------

        if len(similarity) != len(metadata):
            raise ValueError(
                f"{target}: similarity feature rows "
                f"({len(similarity)}) and metadata rows "
                f"({len(metadata)}) do not match"
            )

        if len(targets) != len(metadata):
            raise ValueError(
                f"{target}: target rows "
                f"({len(targets)}) and metadata rows "
                f"({len(metadata)}) do not match"
            )

        # --------------------------------------------------
        # Validate required metadata
        # --------------------------------------------------

        if "query_pdb" not in metadata.columns:
            raise ValueError(
                f"{target}: metadata must contain query_pdb"
            )

        # --------------------------------------------------
        # Validate target column
        # --------------------------------------------------

        if target not in targets.columns:
            raise ValueError(
                f"{target}: target dataset must contain "
                f"column '{target}'"
            )

        # --------------------------------------------------
        # Attach query PDB IDs to similarity features
        # --------------------------------------------------

        similarity_with_metadata = (
            similarity.copy()
        )

        similarity_with_metadata[
            "query_pdb"
        ] = metadata["query_pdb"].values

        # --------------------------------------------------
        # Prepare physicochemical features
        # --------------------------------------------------

        physicochemical = physicochemical.rename(
            columns={
                "pdb_id": "query_pdb"
            }
        )

        # Prevent accidental duplicate query IDs from
        # creating many-to-many merges.
        if physicochemical["query_pdb"].duplicated().any():
            raise ValueError(
                "Physicochemical dataset contains "
                "duplicate pdb_id values"
            )

        # --------------------------------------------------
        # Merge similarity + physicochemical features
        # --------------------------------------------------

        combined = similarity_with_metadata.merge(
            physicochemical,
            on="query_pdb",
            how="left",
            validate="many_to_one",
        )

        # --------------------------------------------------
        # Validate physicochemical merge
        # --------------------------------------------------

        physicochemical_columns = [
            column
            for column in physicochemical.columns
            if column != "query_pdb"
        ]

        if physicochemical_columns:
            unmatched = combined[
                physicochemical_columns
            ].isna().all(axis=1)

            if unmatched.any():
                missing_ids = (
                    combined.loc[
                        unmatched,
                        "query_pdb",
                    ]
                    .unique()
                    .tolist()
                )

                raise ValueError(
                    "Some rows could not be matched "
                    "to physicochemical features: "
                    f"{missing_ids}"
                )

        # --------------------------------------------------
        # Attach target
        # --------------------------------------------------

        combined[target] = targets[
            target
        ].values

        return combined

    def save(self) -> dict:

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        results = {}

        for target in self.TARGETS:

            combined = self.build_target(
                target
            )

            output_file = (
                self.output_directory
                / f"{target}_combined.csv"
            )

            combined.to_csv(
                output_file,
                index=False,
            )

            # Everything except identifiers and target
            # is considered an ML feature.
            feature_columns = [
                column
                for column in combined.columns
                if column not in {
                    "query_pdb",
                    target,
                }
            ]

            results[target] = {
                "output": str(output_file),
                "rows": len(combined),
                "feature_count": len(
                    feature_columns
                ),
            }

        return results
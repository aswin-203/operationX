from pathlib import Path

import pandas as pd


class FeatureBuilder:
    """
    Convert the Operation X similarity-condition dataset
    into an ML-ready feature/target dataset.

    This class does NOT train a model.
    It only prepares the data.
    """

    FEATURE_COLUMNS = [
        "sequence_identity",
        "sequence_coverage",
        "sequence_bitscore",
        "structure_query_tm_score",
        "structure_target_tm_score",
        "structure_rmsd",
        "structure_alignment_length",
        "structure_evalue",
        "pfam_count",
        "interpro_count",
    ]

    TARGET_COLUMNS = [
        "pH",
        "temperature_kelvin",
        "matthews_coefficient",
        "solvent_percent",
    ]

    def __init__(
        self,
        input_file: str,
        output_directory: str = "data/ml",
    ):
        self.input_file = Path(
            input_file
        ).expanduser()

        self.output_directory = Path(
            output_directory
        ).expanduser()

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def load(self) -> pd.DataFrame:
        """Load the similarity-condition dataset."""

        if not self.input_file.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.input_file}"
            )

        df = pd.read_csv(
            self.input_file
        )

        if df.empty:
            raise ValueError(
                "Input dataset is empty"
            )

        return df

    def build_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build numerical similarity features.

        Missing structure measurements are preserved
        and represented using explicit missing indicators.
        """

        missing_columns = [
            column
            for column in self.FEATURE_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing feature columns: "
                + ", ".join(missing_columns)
            )

        features = df[
            self.FEATURE_COLUMNS
        ].copy()

        # Explicit missingness indicators.
        #
        # These are important because missing structure
        # information is different from a real zero value.
        features[
            "structure_missing"
        ] = (
            df[
                "structure_query_tm_score"
            ].isna()
        ).astype(int)

        features[
            "structure_rmsd_missing"
        ] = (
            df[
                "structure_rmsd"
            ].isna()
        ).astype(int)

        # BLAST information is complete in the
        # current pilot dataset, but retain the
        # same principle for future datasets.
        features[
            "sequence_missing"
        ] = (
            df[
                "sequence_identity"
            ].isna()
        ).astype(int)

        return features

    def build_targets(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build crystallization targets."""

        missing_columns = [
            column
            for column in self.TARGET_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing target columns: "
                + ", ".join(missing_columns)
            )

        targets = df[
            self.TARGET_COLUMNS
        ].copy()

        return targets

    def build_metadata(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Preserve identifiers so each ML row can be traced
        back to the original similarity pair.
        """

        return df[
            [
                "query_pdb",
                "similar_pdb",
            ]
        ].copy()

    def build(self) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """
        Build metadata, features, and targets.
        """

        df = self.load()

        metadata = self.build_metadata(
            df
        )

        features = self.build_features(
            df
        )

        targets = self.build_targets(
            df
        )

        return (
            metadata,
            features,
            targets,
        )

    def save(self) -> dict:
        """
        Build and save the ML datasets.
        """

        (
            metadata,
            features,
            targets,
        ) = self.build()

        metadata_file = (
            self.output_directory
            / "metadata.csv"
        )

        features_file = (
            self.output_directory
            / "features.csv"
        )

        targets_file = (
            self.output_directory
            / "targets.csv"
        )

        metadata.to_csv(
            metadata_file,
            index=False,
        )

        features.to_csv(
            features_file,
            index=False,
        )

        targets.to_csv(
            targets_file,
            index=False,
        )

        return {
            "metadata": str(
                metadata_file
            ),
            "features": str(
                features_file
            ),
            "targets": str(
                targets_file
            ),
            "rows": len(features),
            "feature_count": len(
                features.columns
            ),
            "target_count": len(
                targets.columns
            ),
        }

class PredictionFeatureAssembler:

    PHYSICOCHEMICAL_FEATURES = [
        "sequence_length",
        "valid_sequence_length",
        "molecular_weight",
        "isoelectric_point",
        "gravy",
        "instability_index",
        "aromaticity",
        "charged_residue_fraction",
        "polar_residue_fraction",
        "hydrophobic_residue_fraction",
        "cysteine_fraction",
        "glycine_fraction",
        "proline_fraction",
        "ambiguous_residue_count",
        "ambiguous_residue_fraction",
    ]

    SIMILARITY_FEATURES = [
        "best_evalue",
        "best_bitscore",
        "best_identity",
        "best_coverage",
        "best_alignment_length",
        "sequence_strong_hit_count",
        "structure_strong_hit_count",
        "best_rmsd",
        "best_query_tm_score",
        "best_target_tm_score",
        "domain_count",
        "pfam_count",
        "interpro_count",
    ]

    @property
    def feature_columns(self):
        return (
            self.PHYSICOCHEMICAL_FEATURES
            + self.SIMILARITY_FEATURES
        )

    def assemble(
        self,
        physicochemical: dict,
        similarity: dict,
    ) -> dict:

        missing_physicochemical = [
            feature
            for feature in self.PHYSICOCHEMICAL_FEATURES
            if feature not in physicochemical
        ]

        missing_similarity = [
            feature
            for feature in self.SIMILARITY_FEATURES
            if feature not in similarity
        ]

        missing = (
            missing_physicochemical
            + missing_similarity
        )

        if missing:
            raise ValueError(
                "Missing required features: "
                + ", ".join(missing)
            )

        features = {}

        for feature in self.PHYSICOCHEMICAL_FEATURES:
            features[feature] = self._to_float(
                feature,
                physicochemical[feature],
            )

        for feature in self.SIMILARITY_FEATURES:
            features[feature] = self._to_float(
                feature,
                similarity[feature],
            )

        if len(features) != 28:
            raise ValueError(
                "Expected exactly 28 features, "
                f"got {len(features)}"
            )

        return features

    @staticmethod
    def _to_float(
        feature: str,
        value,
    ) -> float:

        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Feature '{feature}' must be numeric"
            ) from exc

        return result
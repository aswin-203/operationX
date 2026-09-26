class RealFeatureBuilder:

    SEQUENCE_FEATURES = [
        "best_evalue",
        "best_bitscore",
        "best_identity",
        "best_coverage",
        "best_alignment_length",
        "sequence_strong_hit_count",
    ]

    STRUCTURE_FEATURES = [
        "structure_strong_hit_count",
        "best_rmsd",
        "best_query_tm_score",
        "best_target_tm_score",
    ]

    DOMAIN_FEATURES = [
        "domain_count",
        "pfam_count",
        "interpro_count",
    ]

    @property
    def feature_columns(self):
        return (
            self.SEQUENCE_FEATURES
            + self.STRUCTURE_FEATURES
            + self.DOMAIN_FEATURES
        )

    def build(
        self,
        sequence_features: dict,
        structure_features: dict,
        domain_features: dict,
    ) -> dict:

        sources = {
            "sequence": (
                sequence_features,
                self.SEQUENCE_FEATURES,
            ),
            "structure": (
                structure_features,
                self.STRUCTURE_FEATURES,
            ),
            "domain": (
                domain_features,
                self.DOMAIN_FEATURES,
            ),
        }

        missing = []

        for _, (values, required) in sources.items():
            for feature in required:
                if feature not in values:
                    missing.append(feature)

        if missing:
            raise ValueError(
                "Missing required real features: "
                + ", ".join(missing)
            )

        result = {}

        for feature in self.feature_columns:

            if feature in sequence_features:
                value = sequence_features[feature]

            elif feature in structure_features:
                value = structure_features[feature]

            else:
                value = domain_features[feature]

            try:
                result[feature] = float(value)

            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Feature '{feature}' must be numeric"
                ) from exc

        return result

    def build_from_evidence(
        self,
        evidence: dict,
    ) -> dict:

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

        sequence_alignment_length = (
            sequence.get("best_alignment_length")
        )

        if sequence_alignment_length is None:
            sequence_alignment_length = (
                structure.get("best_alignment_length")
            )

        sequence_features = {
            "best_evalue": sequence.get(
                "best_evalue"
            ),
            "best_bitscore": sequence.get(
                "best_bitscore"
            ),
            "best_identity": sequence.get(
                "best_identity"
            ),
            "best_coverage": sequence.get(
                "best_coverage"
            ),
            "best_alignment_length": (
                sequence_alignment_length
            ),
            "sequence_strong_hit_count": (
                sequence.get("strong_hit_count")
            ),
        }

        structure_features = {
            "structure_strong_hit_count": (
                structure.get("strong_hit_count")
            ),
            "best_rmsd": structure.get(
                "best_rmsd"
            ),
            "best_query_tm_score": (
                structure.get("best_query_tm_score")
            ),
            "best_target_tm_score": (
                structure.get("best_target_tm_score")
            ),
        }

        domain_features = {
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

        return self.build(
            sequence_features=sequence_features,
            structure_features=structure_features,
            domain_features=domain_features,
        )

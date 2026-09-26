class DomainSimilarity:
    """Extract Pfam and InterPro domain annotations from RCSB entities."""

    DOMAIN_TYPES = {
        "Pfam",
        "InterPro",
    }

    def extract(self, entity: dict) -> list[dict]:
        annotations = entity.get(
            "rcsb_polymer_entity_annotation",
            [],
        )

        features = entity.get(
            "rcsb_polymer_entity_feature",
            [],
        )

        domains = []

        feature_map = {}

        for feature in features:
            feature_id = feature.get("feature_id")

            if feature_id:
                feature_map[feature_id] = feature

        for annotation in annotations:
            annotation_type = annotation.get("type")

            if annotation_type not in self.DOMAIN_TYPES:
                continue

            annotation_id = annotation.get(
                "annotation_id"
            )

            feature = feature_map.get(
                annotation_id,
                {},
            )

            positions = feature.get(
                "feature_positions",
                [],
            )

            start = None
            end = None

            if positions:
                first_position = positions[0]

                start = first_position.get(
                    "beg_seq_id"
                )

                end = first_position.get(
                    "end_seq_id"
                )

            domains.append(
                {
                    "domain_id": annotation_id,
                    "domain_name": annotation.get(
                        "name"
                    ),
                    "domain_type": annotation_type,
                    "provenance": annotation.get(
                        "provenance_source"
                    ),
                    "start": start,
                    "end": end,
                }
            )

        return domains

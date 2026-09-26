import pandas as pd


class SimilarityEvidence:
    """Map BLAST hits to known Operation X records."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    @staticmethod
    def _parse_subject_id(subject_id: str) -> tuple[str, str]:
        """
        Convert BLAST subject identifiers into
        PDB ID and entity ID.

        Supported examples:

        pdb|101M|1
        101M_1
        """

        if not isinstance(subject_id, str):
            raise TypeError(
                "subject_id must be a string"
            )

        subject_id = subject_id.strip()

        if subject_id.startswith("pdb|"):

            parts = subject_id.split("|")

            if len(parts) >= 3:
                return parts[1], parts[2]

        if "_" in subject_id:

            pdb_id, entity_id = (
                subject_id.split("_", 1)
            )

            return pdb_id, entity_id

        raise ValueError(
            f"Unsupported BLAST subject ID: "
            f"{subject_id}"
        )

    def collect(
        self,
        hits: list[dict],
    ) -> list[dict]:

        dataset = pd.read_csv(
            self.dataset_path
        )

        evidence = []

        for hit in hits:

            pdb_id, entity_id = (
                self._parse_subject_id(
                    hit["subject_id"]
                )
            )

            matches = dataset[
                (
                    dataset["pdb_id"]
                    .astype(str)
                    .str.upper()
                    == pdb_id.upper()
                )
                &
                (
                    dataset["entity_id"]
                    .astype(str)
                    == str(entity_id)
                )
            ]

            if matches.empty:
                continue

            record = matches.iloc[0]

            evidence.append(
                {
                    "pdb_id": pdb_id,
                    "entity_id": entity_id,

                    "identity": hit[
                        "identity"
                    ],

                    "coverage": hit[
                        "coverage"
                    ],

                    "evalue": hit[
                        "evalue"
                    ],

                    "bitscore": hit[
                        "bitscore"
                    ],

                    "crystallization_method":
                        record.get(
                            "crystallization_method"
                        ),

                    "pH":
                        record.get("pH"),

                    "temperature_kelvin":
                        record.get(
                            "temperature_kelvin"
                        ),

                    "pressure":
                        record.get(
                            "pressure"
                        ),

                    "crystallization_time":
                        record.get(
                            "crystallization_time"
                        ),

                    "crystallization_details":
                        record.get(
                            "crystallization_details"
                        ),

                    "expression_host":
                        record.get(
                            "expression_host"
                        ),

                    "expression_system":
                        record.get(
                            "expression_system"
                        ),
                }
            )

        return evidence

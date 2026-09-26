import json
from pathlib import Path

import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis


class PhysicochemicalFeatureBuilder:

    VALID_AMINO_ACIDS = set(
        "ACDEFGHIKLMNPQRSTVWY"
    )

    def __init__(
        self,
        records_file: str,
        output_file: str,
    ):
        self.records_file = Path(records_file)
        self.output_file = Path(output_file)

    def _calculate_features(
        self,
        pdb_id: str,
        sequence: str,
    ) -> dict:

        sequence = "".join(sequence.split())
        sequence = sequence.upper()

        if not sequence:
            raise ValueError(
                f"Empty sequence for {pdb_id}"
            )

        original_length = len(sequence)

        clean_sequence = "".join(
            aa
            for aa in sequence
            if aa in self.VALID_AMINO_ACIDS
        )

        if not clean_sequence:
            raise ValueError(
                f"No valid amino acids for {pdb_id}"
            )

        ambiguous_count = (
            original_length - len(clean_sequence)
        )

        ambiguous_fraction = (
            ambiguous_count / original_length
        )

        analysis = ProteinAnalysis(
            clean_sequence
        )

        length = len(clean_sequence)

        charged_residues = sum(
            clean_sequence.count(aa)
            for aa in "DEKR"
        )

        hydrophobic_residues = sum(
            clean_sequence.count(aa)
            for aa in "AILMFWV"
        )

        polar_residues = sum(
            clean_sequence.count(aa)
            for aa in "STNQ"
        )

        return {
            "pdb_id": pdb_id,

            "sequence_length": original_length,

            "valid_sequence_length": length,

            "ambiguous_residue_count": ambiguous_count,

            "ambiguous_residue_fraction": (
                ambiguous_fraction
            ),

            "molecular_weight": (
                analysis.molecular_weight()
            ),

            "isoelectric_point": (
                analysis.isoelectric_point()
            ),

            "aromaticity": (
                analysis.aromaticity()
            ),

            "instability_index": (
                analysis.instability_index()
            ),

            "gravy": (
                analysis.gravy()
            ),

            "charged_residue_fraction": (
                charged_residues / length
            ),

            "hydrophobic_residue_fraction": (
                hydrophobic_residues / length
            ),

            "polar_residue_fraction": (
                polar_residues / length
            ),

            "glycine_fraction": (
                clean_sequence.count("G") / length
            ),

            "proline_fraction": (
                clean_sequence.count("P") / length
            ),

            "cysteine_fraction": (
                clean_sequence.count("C") / length
            ),
        }

    def build(self) -> pd.DataFrame:

        if not self.records_file.exists():
            raise FileNotFoundError(
                self.records_file
            )

        with self.records_file.open() as f:
            records = json.load(f)

        rows = []

        for pdb_id, record in records.items():

            sequence = record.get(
                "sequence",
                "",
            )

            if not sequence:
                continue

            rows.append(
                self._calculate_features(
                    pdb_id=pdb_id,
                    sequence=sequence,
                )
            )

        return pd.DataFrame(rows)

    def save(self) -> dict:

        df = self.build()

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            self.output_file,
            index=False,
        )

        return {
            "output": str(self.output_file),
            "rows": len(df),
            "feature_count": len(df.columns) - 1,
        }

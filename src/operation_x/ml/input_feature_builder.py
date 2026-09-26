from Bio.SeqUtils.ProtParam import ProteinAnalysis


class InputFeatureBuilder:

    VALID_AMINO_ACIDS = set(
        "ACDEFGHIKLMNPQRSTVWY"
    )

    AMBIGUOUS_AMINO_ACIDS = set(
        "XBZJUO"
    )

    def __init__(self):
        pass

    def _clean_sequence(
        self,
        sequence: str,
    ) -> str:

        if not isinstance(sequence, str):
            raise ValueError(
                "Sequence must be a string"
            )

        sequence = (
            sequence
            .strip()
            .upper()
            .replace(" ", "")
            .replace("\n", "")
            .replace("\r", "")
        )

        if not sequence:
            raise ValueError(
                "Sequence cannot be empty"
            )

        invalid = set(sequence) - (
            self.VALID_AMINO_ACIDS
            | self.AMBIGUOUS_AMINO_ACIDS
        )

        if invalid:
            raise ValueError(
                "Invalid amino acid characters: "
                + ", ".join(sorted(invalid))
            )

        return sequence

    def build(
        self,
        sequence: str,
    ) -> dict:

        sequence = self._clean_sequence(
            sequence
        )

        valid_sequence = "".join(
            residue
            for residue in sequence
            if residue in self.VALID_AMINO_ACIDS
        )

        ambiguous_count = sum(
            residue in self.AMBIGUOUS_AMINO_ACIDS
            for residue in sequence
        )

        sequence_length = len(sequence)

        valid_sequence_length = len(
            valid_sequence
        )

        ambiguous_fraction = (
            ambiguous_count / sequence_length
            if sequence_length
            else 0.0
        )

        # Biopython's ProteinAnalysis does not
        # accept ambiguous residues such as X.
        if not valid_sequence:
            raise ValueError(
                "Sequence contains no valid "
                "amino acid residues"
            )

        analysis = ProteinAnalysis(
            valid_sequence
        )

        charged_count = sum(
            residue in "DEKR"
            for residue in valid_sequence
        )

        polar_count = sum(
            residue in "STNQ"
            for residue in valid_sequence
        )

        hydrophobic_count = sum(
            residue in "AVILMFWY"
            for residue in valid_sequence
        )

        cysteine_count = valid_sequence.count(
            "C"
        )

        glycine_count = valid_sequence.count(
            "G"
        )

        proline_count = valid_sequence.count(
            "P"
        )

        length = len(valid_sequence)

        return {
            "sequence_length": float(
                sequence_length
            ),

            "valid_sequence_length": float(
                valid_sequence_length
            ),

            "molecular_weight": float(
                analysis.molecular_weight()
            ),

            "isoelectric_point": float(
                analysis.isoelectric_point()
            ),

            "gravy": float(
                analysis.gravy()
            ),

            "instability_index": float(
                analysis.instability_index()
            ),

            "aromaticity": float(
                analysis.aromaticity()
            ),

            "charged_residue_fraction": (
                charged_count / length
            ),

            "polar_residue_fraction": (
                polar_count / length
            ),

            "hydrophobic_residue_fraction": (
                hydrophobic_count / length
            ),

            "cysteine_fraction": (
                cysteine_count / length
            ),

            "glycine_fraction": (
                glycine_count / length
            ),

            "proline_fraction": (
                proline_count / length
            ),

            "ambiguous_residue_count": float(
                ambiguous_count
            ),

            "ambiguous_residue_fraction": (
                ambiguous_fraction
            ),
        }
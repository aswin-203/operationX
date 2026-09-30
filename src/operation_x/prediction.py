"""
Operation X — Prediction Script

Input:
    Protein amino-acid sequence

Output:
    - BLAST evidence
    - Predicted pH
    - Predicted temperature
    - Predicted Matthews coefficient
    - Predicted solvent percentage
    - Top experimental crystallization conditions
    - Experimental reagent/condition text

Important:
    Predictions are ranked experimental-condition estimates,
    not guarantees of crystallization success.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from Bio.SeqUtils.ProtParam import ProteinAnalysis

from operation_x.ml.large_crystallization_dataset import (
    LargeCrystallizationDataset,
)

from operation_x.ml.crystallization_condition_ranker import (
    CrystallizationConditionRanker,
)


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = Path("models")

BLAST_DB = Path(
    "data/blast/large/pdb_sequences"
)

LARGE_DATASET = Path(
    "data/structures/large/records.json"
)


# ============================================================
# TARGETS
# ============================================================

TARGETS = [
    "pH",
    "temperature_kelvin",
    "matthews_coefficient",
    "solvent_percent",
]


MODEL_FILES = {
    "pH": (
        MODEL_DIR / "ph_model.joblib"
    ),

    "temperature_kelvin": (
        MODEL_DIR / "temperature_kelvin_model.joblib"
    ),

    "matthews_coefficient": (
        MODEL_DIR / "matthews_coefficient_model.joblib"
    ),

    "solvent_percent": (
        MODEL_DIR / "solvent_percent_model.joblib"
    ),
}


# ============================================================
# SEQUENCE
# ============================================================

VALID_AMINO_ACIDS = set(
    "ACDEFGHIKLMNPQRSTVWY"
)


def clean_sequence(
    sequence: str,
) -> str:

    sequence = "".join(
        sequence.split()
    ).upper()

    if not sequence:

        raise ValueError(
            "Protein sequence cannot be empty."
        )

    invalid = (
        set(sequence)
        - VALID_AMINO_ACIDS
    )

    if invalid:

        raise ValueError(
            "Invalid amino acid characters: "
            + ", ".join(
                sorted(invalid)
            )
        )

    return sequence


# ============================================================
# PHYSICOCHEMICAL FEATURES
# ============================================================

def build_physicochemical_features(
    sequence: str,
) -> dict:

    sequence = clean_sequence(
        sequence
    )

    analysis = ProteinAnalysis(
        sequence
    )

    length = len(sequence)

    charged_count = sum(
        sequence.count(residue)
        for residue in "DEKR"
    )

    polar_count = sum(
        sequence.count(residue)
        for residue in "STNQ"
    )

    hydrophobic_count = sum(
        sequence.count(residue)
        for residue in "AVILMFWY"
    )

    return {
        "sequence_length": float(
            length
        ),

        "valid_sequence_length": float(
            length
        ),

        "ambiguous_residue_count": 0.0,

        "ambiguous_residue_fraction": 0.0,

        "molecular_weight": float(
            analysis.molecular_weight()
        ),

        "isoelectric_point": float(
            analysis.isoelectric_point()
        ),

        "aromaticity": float(
            analysis.aromaticity()
        ),

        "instability_index": float(
            analysis.instability_index()
        ),

        "gravy": float(
            analysis.gravy()
        ),

        "charged_residue_fraction": (
            charged_count / length
        ),

        "hydrophobic_residue_fraction": (
            hydrophobic_count / length
        ),

        "polar_residue_fraction": (
            polar_count / length
        ),

        "glycine_fraction": (
            sequence.count("G") / length
        ),

        "proline_fraction": (
            sequence.count("P") / length
        ),

        "cysteine_fraction": (
            sequence.count("C") / length
        ),
    }


# ============================================================
# BLAST
# ============================================================

def run_blast(
    sequence: str,
    max_hits: int = 100,
) -> list[dict]:

    # BLAST databases are stored using a common prefix:
    #
    # data/blast/large/pdb_sequences
    #
    # with files such as:
    #   pdb_sequences.phr
    #   pdb_sequences.pin
    #   pdb_sequences.psq
    #
    # Therefore we check the .pin index file.

    blast_index = Path(
        str(BLAST_DB) + ".pin"
    )

    if not blast_index.exists():

        raise FileNotFoundError(
            "BLAST database not found: "
            f"{BLAST_DB}\n"
            f"Expected database index: "
            f"{blast_index}"
        )

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(
            temp_dir
        )

        query_file = (
            temp_dir / "query.fasta"
        )

        output_file = (
            temp_dir / "blast.tsv"
        )

        query_file.write_text(
            ">query\n"
            + sequence
            + "\n",
            encoding="utf-8",
        )

        command = [
            "blastp",

            "-query",
            str(query_file),

            "-db",
            str(BLAST_DB),

            "-out",
            str(output_file),

            "-outfmt",
            "6 qseqid sseqid pident "
            "length qlen evalue bitscore",

            "-max_target_seqs",
            str(max_hits),

            "-max_hsps",
            "1",
        ]

        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if not output_file.exists():

            return []

        hits = []

        for line in output_file.read_text(
            encoding="utf-8"
        ).splitlines():

            parts = line.split("\t")

            if len(parts) < 7:

                continue

            try:

                identity = float(
                    parts[2]
                )

                alignment_length = int(
                    parts[3]
                )

                query_length = int(
                    parts[4]
                )

                evalue = float(
                    parts[5]
                )

                bitscore = float(
                    parts[6]
                )

            except ValueError:

                continue

            coverage = (
                alignment_length
                / query_length
                if query_length
                else 0.0
            )

            hits.append(
                {
                    "subject_id": parts[1],
                    "identity": identity,
                    "coverage": coverage,
                    "evalue": evalue,
                    "bitscore": bitscore,
                }
            )

        return hits


# ============================================================
# BLAST FEATURES
# ============================================================

def build_blast_features(
    hits: list[dict],
) -> dict:

    if not hits:

        return {
            "best_identity": 0.0,
            "best_coverage": 0.0,
            "best_evalue": 1.0,
            "best_bitscore": 0.0,
            "sequence_strong_hit_count": 0.0,
        }

    best = sorted(
        hits,
        key=lambda hit: (
            -hit["bitscore"],
            -hit["identity"],
            hit["evalue"],
        ),
    )[0]

    strong_hits = [
        hit
        for hit in hits
        if hit["identity"] >= 70.0
        and hit["coverage"] >= 0.80
        and hit["evalue"] <= 1e-5
    ]

    return {
        "best_identity": (
            best["identity"]
        ),

        "best_coverage": (
            best["coverage"]
        ),

        "best_evalue": (
            best["evalue"]
        ),

        "best_bitscore": (
            best["bitscore"]
        ),

        "sequence_strong_hit_count": float(
            len(strong_hits)
        ),
    }


# ============================================================
# MODEL INPUT
# ============================================================

def prepare_model_input(
    features: dict,
    model,
) -> pd.DataFrame:

    feature_names = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if feature_names is None:

        raise ValueError(
            "Model does not contain "
            "feature_names_in_."
        )

    row = {}

    for feature in feature_names:

        row[feature] = features.get(
            feature,
            np.nan,
        )

    frame = pd.DataFrame(
        [row],
        columns=feature_names,
    )

    frame = frame.apply(
        pd.to_numeric,
        errors="coerce",
    )

    frame = frame.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if frame.isna().any().any():

        frame = frame.fillna(0.0)

    return frame


# ============================================================
# PREDICTION
# ============================================================

def predict(
    sequence: str,
) -> dict:

    sequence = clean_sequence(
        sequence
    )

    print()

    print(
        "Running BLAST similarity search..."
    )

    hits = run_blast(
        sequence
    )

    print(
        f"BLAST hits      : "
        f"{len(hits)}"
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    features = {}

    features.update(
        build_physicochemical_features(
            sequence
        )
    )

    features.update(
        build_blast_features(
            hits
        )
    )

    # --------------------------------------------------------
    # BLAST evidence
    # --------------------------------------------------------

    print()

    print(
        "Building experimental-condition evidence..."
    )

    print()

    print(
        "BLAST EVIDENCE"
    )

    print(
        "-" * 70
    )

    if hits:

        best = sorted(
            hits,
            key=lambda hit: (
                -hit["bitscore"],
                -hit["identity"],
            ),
        )[0]

        print(
            f"Best identity     : "
            f"{best['identity']:.2f}%"
        )

        print(
            f"Best coverage     : "
            f"{best['coverage']:.3f}"
        )

        print(
            f"Best E-value      : "
            f"{best['evalue']:.2e}"
        )

        print(
            f"Best bitscore     : "
            f"{best['bitscore']:.2f}"
        )

        print(
            f"Strong hits       : "
            f"{features['sequence_strong_hit_count']:.0f}"
        )

    else:

        print(
            "No BLAST hits found."
        )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = {}

    print()

    print(
        "PREDICTIONS"
    )

    print(
        "-" * 70
    )

    for target in TARGETS:

        model_file = MODEL_FILES[
            target
        ]

        if not model_file.exists():

            raise FileNotFoundError(
                "Model not found: "
                f"{model_file}"
            )

        model = joblib.load(
            model_file
        )

        row = prepare_model_input(
            features,
            model,
        )

        prediction = float(
            model.predict(row)[0]
        )

        predictions[target] = (
            prediction
        )

    print()

    print(
        f"Predicted pH             : "
        f"{predictions['pH']:.2f}"
    )

    print(
        f"Predicted temperature    : "
        f"{predictions['temperature_kelvin']:.2f} K"
    )

    print(
        f"Predicted Matthews coeff : "
        f"{predictions['matthews_coefficient']:.3f}"
    )

    print(
        f"Predicted solvent        : "
        f"{predictions['solvent_percent']:.2f}%"
    )

    # ========================================================
    # EXPERIMENTAL CONDITION RANKING
    # ========================================================

    print()

    print(
        "Loading experimental "
        "crystallization dataset..."
    )

    if not LARGE_DATASET.exists():

        raise FileNotFoundError(
            "Experimental dataset not found: "
            f"{LARGE_DATASET}"
        )

    dataset = (
        LargeCrystallizationDataset(
            LARGE_DATASET
        )
    )

    ranker = (
        CrystallizationConditionRanker(
            dataset
        )
    )

    print()

    print(
        "Ranking experimental "
        "crystallization conditions..."
    )

    recommendations = ranker.rank(
        predicted_ph=predictions[
            "pH"
        ],

        predicted_temperature=(
            predictions[
                "temperature_kelvin"
            ]
        ),

        predicted_matthews=(
            predictions[
                "matthews_coefficient"
            ]
        ),

        predicted_solvent=(
            predictions[
                "solvent_percent"
            ]
        ),

        top_n=5,
    )

    # ========================================================
    # DISPLAY CONDITIONS
    # ========================================================

    print()

    print(
        "TOP EXPERIMENTAL CONDITIONS"
    )

    print(
        "-" * 70
    )

    if not recommendations:

        print(
            "No experimental conditions "
            "were available."
        )

    for index, recommendation in enumerate(
        recommendations,
        start=1,
    ):

        print()

        print(
            f"{index}. PDB ID       : "
            f"{recommendation.get('query_pdb')}"
        )

        score = recommendation.get(
            "score"
        )

        if score is not None:

            print(
                f"   Score        : "
                f"{float(score):.4f}"
            )

        print(
            f"   pH           : "
            f"{recommendation.get('pH')}"
        )

        print(
            f"   Temperature  : "
            f"{recommendation.get('temperature_kelvin')} K"
        )

        print(
            f"   Matthews     : "
            f"{recommendation.get('matthews_coefficient')}"
        )

        print(
            f"   Solvent      : "
            f"{recommendation.get('solvent_percent')}%"
        )

        print(
            "   Reagents / condition:"
        )

        condition = recommendation.get(
            "condition_text"
        )

        if condition:

            print(
                f"   {condition}"
            )

        else:

            details = recommendation.get(
                "crystallization_details"
            )

            if details:

                print(
                    f"   {details}"
                )

            else:

                print(
                    "   Not available"
                )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "These are predicted/ranked experimental "
        "condition estimates."
    )

    print(
        "They are not guarantees of crystallization success."
    )

    print(
        "=" * 70
    )

    return predictions


# ============================================================
# FASTA
# ============================================================

def load_fasta(
    path: str,
) -> str:

    fasta_path = Path(
        path
    )

    if not fasta_path.exists():

        raise FileNotFoundError(
            f"FASTA file not found: "
            f"{fasta_path}"
        )

    lines = fasta_path.read_text(
        encoding="utf-8"
    ).splitlines()

    sequence = "".join(
        line.strip()
        for line in lines
        if line.strip()
        and not line.startswith(">")
    )

    return sequence


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Operation X protein "
            "crystallization condition prediction"
        )
    )

    parser.add_argument(
        "--sequence",
        type=str,
        help="Protein amino-acid sequence",
    )

    parser.add_argument(
        "--fasta",
        type=str,
        help=(
            "FASTA file containing "
            "one protein sequence"
        ),
    )

    args = parser.parse_args()

    if args.sequence and args.fasta:

        raise ValueError(
            "Use either --sequence or "
            "--fasta, not both."
        )

    if args.sequence:

        sequence = args.sequence

    elif args.fasta:

        sequence = load_fasta(
            args.fasta
        )

    else:

        parser.error(
            "Provide either --sequence "
            "or --fasta."
        )

    predict(
        sequence
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
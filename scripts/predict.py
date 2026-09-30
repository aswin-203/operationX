"""
Operation X — Prediction Script

Input:
    Protein amino-acid sequence

Output:
    Predicted:
        - pH
        - temperature (K)
        - Matthews coefficient
        - solvent percentage

The prediction uses:
    1. Physicochemical features
    2. BLAST similarity against the local PDB sequence database
    3. BLAST-derived experimental-condition evidence
    4. Expression features
    5. Trained Operation X models

Important:
    Predictions are ranked experimental-condition estimates,
    not guarantees of crystallization success.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from Bio.SeqUtils.ProtParam import ProteinAnalysis


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = Path("models")

BLAST_DB = Path(
    "data/blast/large/pdb_sequences"
)

FINAL_DIR = Path(
    "data/ml_large/final"
)

TARGET_TRAIN_FILES = {
    "pH": FINAL_DIR / "pH_train.csv",
    "temperature_kelvin": FINAL_DIR / "temperature_kelvin_train.csv",
    "matthews_coefficient": FINAL_DIR / "matthews_coefficient_train.csv",
    "solvent_percent": FINAL_DIR / "solvent_percent_train.csv",
}


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
    "pH": MODEL_DIR / "ph_model.joblib",
    "temperature_kelvin": MODEL_DIR / "temperature_kelvin_model.joblib",
    "matthews_coefficient": MODEL_DIR / "matthews_coefficient_model.joblib",
    "solvent_percent": MODEL_DIR / "solvent_percent_model.joblib",
}


# ============================================================
# FEATURES
# ============================================================

PHYSICOCHEMICAL_FEATURES = [
    "sequence_length",
    "valid_sequence_length",
    "ambiguous_residue_count",
    "ambiguous_residue_fraction",
    "molecular_weight",
    "isoelectric_point",
    "aromaticity",
    "instability_index",
    "gravy",
    "charged_residue_fraction",
    "hydrophobic_residue_fraction",
    "polar_residue_fraction",
    "glycine_fraction",
    "proline_fraction",
    "cysteine_fraction",
]


BLAST_FEATURES = [
    "best_identity",
    "best_coverage",
    "best_evalue",
    "best_bitscore",
    "sequence_strong_hit_count",
]


DOMAIN_FEATURES = [
    "domain_count",
    "pfam_count",
    "interpro_count",
]


EXPRESSION_FEATURES = [
    "host_ecoli",
    "host_insect",
    "host_yeast",
    "host_other",
    "host_unknown",
    "system_bacterial",
    "system_insect",
    "system_yeast",
    "system_mammalian",
    "system_other",
    "system_unknown",
    "strain_present",
    "expression_evidence_count",
]


BLAST_CONDITION_FEATURES = [
    "blast_condition_neighbor_count",
    "blast_condition_strong_neighbor_count",
    "blast_condition_mean_identity_top5",
    "blast_condition_mean_coverage_top5",
    "blast_condition_weighted_value",
    "blast_condition_top_neighbor_identity",
    "blast_condition_top_neighbor_coverage",
    "blast_condition_top_neighbor_bitscore",
]


# ============================================================
# SEQUENCE CLEANING
# ============================================================

VALID_AMINO_ACIDS = set(
    "ACDEFGHIKLMNPQRSTVWY"
)


def clean_sequence(sequence: str) -> str:
    """
    Remove whitespace and validate the protein sequence.
    """

    if not isinstance(sequence, str):
        raise TypeError(
            "Protein sequence must be a string."
        )

    sequence = "".join(
        sequence.split()
    ).upper()

    if not sequence:
        raise ValueError(
            "Protein sequence cannot be empty."
        )

    invalid = sorted(
        set(sequence) - VALID_AMINO_ACIDS - set("BXZJUO")
    )

    if invalid:
        raise ValueError(
            "Invalid amino-acid characters found: "
            + ", ".join(invalid)
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

    valid_sequence_length = sum(
        residue in VALID_AMINO_ACIDS
        for residue in sequence
    )

    ambiguous_count = (
        length
        - valid_sequence_length
    )

    ambiguous_fraction = (
        ambiguous_count / length
        if length
        else 0.0
    )

    # ProteinAnalysis can fail for some
    # unusual sequences. Keep prediction
    # pipeline robust.
    try:
        molecular_weight = (
            analysis.molecular_weight()
        )
    except Exception:
        molecular_weight = np.nan

    try:
        isoelectric_point = (
            analysis.isoelectric_point()
        )
    except Exception:
        isoelectric_point = np.nan

    try:
        aromaticity = (
            analysis.aromaticity()
        )
    except Exception:
        aromaticity = np.nan

    try:
        instability_index = (
            analysis.instability_index()
        )
    except Exception:
        instability_index = np.nan

    try:
        gravy = (
            analysis.gravy()
        )
    except Exception:
        gravy = np.nan

    # Residue fractions
    def fraction(
        residues: str,
    ) -> float:

        return sum(
            sequence.count(residue)
            for residue in residues
        ) / length

    charged_fraction = fraction(
        "DEKR"
    )

    hydrophobic_fraction = fraction(
        "AVILMFWY"
    )

    polar_fraction = fraction(
        "STNQ"
    )

    glycine_fraction = fraction(
        "G"
    )

    proline_fraction = fraction(
        "P"
    )

    cysteine_fraction = fraction(
        "C"
    )

    return {
        "sequence_length": length,
        "valid_sequence_length": valid_sequence_length,
        "ambiguous_residue_count": ambiguous_count,
        "ambiguous_residue_fraction": ambiguous_fraction,
        "molecular_weight": molecular_weight,
        "isoelectric_point": isoelectric_point,
        "aromaticity": aromaticity,
        "instability_index": instability_index,
        "gravy": gravy,
        "charged_residue_fraction": charged_fraction,
        "hydrophobic_residue_fraction": hydrophobic_fraction,
        "polar_residue_fraction": polar_fraction,
        "glycine_fraction": glycine_fraction,
        "proline_fraction": proline_fraction,
        "cysteine_fraction": cysteine_fraction,
    }


# ============================================================
# BLAST SEARCH
# ============================================================

def run_blast(
    sequence: str,
    max_hits: int = 100,
) -> list[dict]:

    if not BLAST_DB.with_suffix(".pin").exists():
        raise FileNotFoundError(
            "BLAST database was not found.\n"
            f"Expected: {BLAST_DB}.pin"
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
            ">operation_x_query\n"
            f"{sequence}\n",
            encoding="utf-8",
        )

        command = [
            "blastp",
            "-query",
            str(query_file),
            "-db",
            str(BLAST_DB),
            "-max_target_seqs",
            str(max_hits),
            "-outfmt",
            (
                "6 qseqid sseqid pident "
                "length qlen slen evalue bitscore"
            ),
            "-evalue",
            "1e-5",
            "-out",
            str(output_file),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "BLAST search failed:\n"
                + result.stderr
            )

        hits = []

        if not output_file.exists():
            return hits

        for line in output_file.read_text(
            encoding="utf-8"
        ).splitlines():

            if not line.strip():
                continue

            fields = line.split("\t")

            if len(fields) != 8:
                continue

            (
                qseqid,
                sseqid,
                identity,
                alignment_length,
                qlen,
                slen,
                evalue,
                bitscore,
            ) = fields

            qlen = int(qlen)
            alignment_length = int(
                alignment_length
            )

            coverage = (
                alignment_length / qlen
                if qlen > 0
                else 0.0
            )

            hits.append(
                {
                    "query_id": qseqid,
                    "subject_id": sseqid,
                    "identity": float(identity),
                    "alignment_length": alignment_length,
                    "query_length": qlen,
                    "subject_length": int(slen),
                    "coverage": coverage,
                    "evalue": float(evalue),
                    "bitscore": float(bitscore),
                }
            )

    return hits


# ============================================================
# BASIC BLAST FEATURES
# ============================================================

def build_blast_features(
    hits: list[dict],
) -> dict:

    if not hits:

        return {
            "best_identity": np.nan,
            "best_coverage": np.nan,
            "best_evalue": np.nan,
            "best_bitscore": np.nan,
            "sequence_strong_hit_count": 0,
        }

    # Same rule used by the training
    # evidence-fusion logic:
    # best sequence hit = highest bitscore.
    best_hit = max(
        hits,
        key=lambda hit: hit["bitscore"],
    )

    strong_hits = [
        hit
        for hit in hits
        if (
            hit["identity"] >= 70.0
            and hit["coverage"] >= 0.70
        )
    ]

    return {
        "best_identity": best_hit["identity"],
        "best_coverage": best_hit["coverage"],
        "best_evalue": best_hit["evalue"],
        "best_bitscore": best_hit["bitscore"],
        "sequence_strong_hit_count": len(
            strong_hits
        ),
    }


# ============================================================
# BLAST CONDITION EVIDENCE
# ============================================================

def load_training_conditions(
    target: str,
) -> dict:

    file = TARGET_TRAIN_FILES[target]

    if not file.exists():
        raise FileNotFoundError(
            f"Training file not found: {file}"
        )

    df = pd.read_csv(
        file
    )

    if "pdb_id" not in df.columns:
        raise ValueError(
            f"{file} does not contain pdb_id."
        )

    if target not in df.columns:
        raise ValueError(
            f"{file} does not contain {target}."
        )

    result = {}

    for _, row in df.iterrows():

        pdb_id = str(
            row["pdb_id"]
        ).strip()

        value = pd.to_numeric(
            row[target],
            errors="coerce",
        )

        if pd.isna(value):
            continue

        result[pdb_id.upper()] = float(
            value
        )

    return result


def build_condition_features(
    hits: list[dict],
    target: str,
) -> dict:
    """
    Estimate the target condition using only
    experimental values from the TRAINING split.

    This prevents test/validation target values
    from entering prediction.
    """

    condition_map = load_training_conditions(
        target
    )

    usable = []

    for hit in hits:

        subject = str(
            hit["subject_id"]
        ).strip().upper()

        # Only use homologs that belong to
        # this target's training split.
        if subject not in condition_map:
            continue

        identity = hit["identity"]
        coverage = hit["coverage"]

        # Same broad evidence thresholds used
        # when constructing condition evidence.
        if identity < 20.0:
            continue

        if coverage < 0.80:
            continue

        target_value = condition_map[
            subject
        ]

        weight = (
            identity / 100.0
        ) * (
            coverage / 100.0
        )

        usable.append(
            {
                "pdb_id": subject,
                "identity": identity,
                "coverage": coverage,
                "bitscore": hit["bitscore"],
                "value": target_value,
                "weight": weight,
            }
        )

    if not usable:

        return {
            "blast_condition_neighbor_count": 0,
            "blast_condition_strong_neighbor_count": 0,
            "blast_condition_mean_identity_top5": 0.0,
            "blast_condition_mean_coverage_top5": 0.0,
            "blast_condition_weighted_value": np.nan,
            "blast_condition_top_neighbor_identity": 0.0,
            "blast_condition_top_neighbor_coverage": 0.0,
            "blast_condition_top_neighbor_bitscore": 0.0,
        }

    # Sort in the same evidence-first order:
    # identity -> coverage -> bitscore.
    usable.sort(
        key=lambda item: (
            item["identity"],
            item["coverage"],
            item["bitscore"],
        ),
        reverse=True,
    )

    top20 = usable[:20]

    top5 = top20[:5]

    strong = [
        item
        for item in top20
        if (
            item["identity"] >= 50.0
            and item["coverage"] >= 0.80
        )
    ]

    weights = np.array(
        [
            item["weight"]
            for item in top20
        ],
        dtype=float,
    )

    values = np.array(
        [
            item["value"]
            for item in top20
        ],
        dtype=float,
    )

    if weights.sum() > 0:

        weighted_value = float(
            np.average(
                values,
                weights=weights,
            )
        )

    else:

        weighted_value = np.nan

    return {
        "blast_condition_neighbor_count": len(
            top20
        ),
        "blast_condition_strong_neighbor_count": len(
            strong
        ),
        "blast_condition_mean_identity_top5": float(
            np.mean(
                [
                    item["identity"]
                    for item in top5
                ]
            )
        ),
        "blast_condition_mean_coverage_top5": float(
            np.mean(
                [
                    item["coverage"]
                    for item in top5
                ]
            )
        ),
        "blast_condition_weighted_value": weighted_value,
        "blast_condition_top_neighbor_identity": float(
            top20[0]["identity"]
        ),
        "blast_condition_top_neighbor_coverage": float(
            top20[0]["coverage"]
        ),
        "blast_condition_top_neighbor_bitscore": float(
            top20[0]["bitscore"]
        ),
    }


# ============================================================
# EXPRESSION FEATURES
# ============================================================

def build_expression_features() -> dict:
    """
    No expression metadata is provided for a new
    sequence, so represent expression as unknown.

    The training dataset contains explicit unknown
    categories, so this is preferable to pretending
    the protein was expressed in E. coli.
    """

    return {
        "host_ecoli": 0,
        "host_insect": 0,
        "host_yeast": 0,
        "host_other": 0,
        "host_unknown": 1,

        "system_bacterial": 0,
        "system_insect": 0,
        "system_yeast": 0,
        "system_mammalian": 0,
        "system_other": 0,
        "system_unknown": 1,

        "strain_present": 0,

        "expression_evidence_count": 0,
    }


# ============================================================
# DOMAIN FEATURES
# ============================================================

def build_domain_features() -> dict:
    """
    No domain annotation is inferred locally for a new
    sequence in this prediction script.

    Missing domain evidence is represented as zero
    and the trained pipeline's imputer handles missing
    values where appropriate.
    """

    return {
        "domain_count": np.nan,
        "pfam_count": np.nan,
        "interpro_count": np.nan,
    }


# ============================================================
# COMPLETE FEATURE ROW
# ============================================================

def build_feature_row(
    sequence: str,
    hits: list[dict],
) -> dict:

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

    features.update(
        build_domain_features()
    )

    features.update(
        build_expression_features()
    )

    features["blast_no_hit"] = (
        1 if not hits else 0
    )

    return features


# ============================================================
# MODEL INPUT PREPARATION
# ============================================================

def prepare_model_input(
    features: dict,
    model,
) -> pd.DataFrame:
    """
    Match the input row to the feature names
    stored inside the trained sklearn pipeline.

    The current models were saved as sklearn
    Pipelines containing median imputation + model.
    """

    # Determine feature names from the underlying
    # estimator/pipeline.
    feature_names = None

    if hasattr(
        model,
        "feature_names_in_",
    ):
        feature_names = list(
            model.feature_names_in_
        )

    elif hasattr(
        model,
        "named_steps",
    ):

        for step in model.named_steps.values():

            if hasattr(
                step,
                "feature_names_in_",
            ):
                feature_names = list(
                    step.feature_names_in_
                )
                break

    if feature_names is None:

        raise RuntimeError(
            "Could not determine the trained "
            "model feature names."
        )

    row = pd.DataFrame(
        [
            {
                feature: features.get(
                    feature,
                    np.nan,
                )
                for feature in feature_names
            }
        ]
    )

    # Same numeric conversion used during training.
    for column in row.columns:

        row[column] = pd.to_numeric(
            row[column],
            errors="coerce",
        )

    row = row.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Same BLAST e-value transformation used
    # during training.
    if "best_evalue" in row.columns:

        evalue = pd.to_numeric(
            row["best_evalue"],
            errors="coerce",
        )

        no_hit = evalue.isna()

        safe_evalue = (
            evalue.fillna(1.0)
            .clip(lower=1e-300)
        )

        row["best_evalue"] = (
            -np.log10(safe_evalue)
        )

        if "blast_no_hit" in row.columns:

            # If BLAST explicitly returned no hit,
            # preserve that flag.
            row.loc[
                no_hit,
                "blast_no_hit",
            ] = 1

        row.loc[
            no_hit,
            "best_evalue",
        ] = 0.0

    return row


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
    print("=" * 70)
    print("OPERATION X — PREDICTION")
    print("=" * 70)

    print()
    print(
        f"Sequence length : {len(sequence)}"
    )

    # --------------------------------------------------------
    # 1. BLAST
    # --------------------------------------------------------

    print()
    print("Running BLAST similarity search...")

    hits = run_blast(
        sequence,
        max_hits=100,
    )

    print(
        f"BLAST hits      : {len(hits)}"
    )

    # --------------------------------------------------------
    # 2. Basic features
    # --------------------------------------------------------

    features = build_feature_row(
        sequence,
        hits,
    )

    # --------------------------------------------------------
    # 3. Condition evidence
    # --------------------------------------------------------

    condition_features = {}

    print()
    print(
        "Building experimental-condition evidence..."
    )

    for target in TARGETS:

        condition_features.update(
            build_condition_features(
                hits,
                target,
            )
        )

    features.update(
        condition_features
    )

    # --------------------------------------------------------
    # 4. Print evidence summary
    # --------------------------------------------------------

    blast_features = build_blast_features(
        hits
    )

    print()
    print("BLAST EVIDENCE")
    print("-" * 70)

    if hits:

        print(
            "Best identity     : "
            f"{blast_features['best_identity']:.2f}%"
        )

        print(
            "Best coverage     : "
            f"{blast_features['best_coverage']:.3f}"
        )

        print(
            "Best E-value      : "
            f"{blast_features['best_evalue']:.3g}"
        )

        print(
            "Best bitscore     : "
            f"{blast_features['best_bitscore']:.2f}"
        )

        print(
            "Strong hits       : "
            f"{blast_features['sequence_strong_hit_count']}"
        )

    else:

        print(
            "No BLAST evidence found."
        )

    # --------------------------------------------------------
    # 5. Load models and predict
    # --------------------------------------------------------

    predictions = {}

    print()
    print("PREDICTIONS")
    print("-" * 70)

    for target in TARGETS:

        model_file = MODEL_FILES[target]

        if not model_file.exists():

            raise FileNotFoundError(
                f"Model not found: {model_file}"
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

        predictions[target] = prediction

    # --------------------------------------------------------
    # 6. Display results
    # --------------------------------------------------------

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

    print()
    print("=" * 70)
    print(
        "These are predicted/ranked experimental "
        "condition estimates."
    )
    print(
        "They are not guarantees of crystallization success."
    )
    print("=" * 70)

    return predictions


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Operation X protein crystallization "
            "condition prediction"
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
        help="FASTA file containing one protein sequence",
    )

    args = parser.parse_args()

    if args.sequence:

        sequence = args.sequence

    elif args.fasta:

        fasta_file = Path(
            args.fasta
        )

        if not fasta_file.exists():

            raise FileNotFoundError(
                f"FASTA file not found: {fasta_file}"
            )

        lines = fasta_file.read_text(
            encoding="utf-8"
        ).splitlines()

        sequence = "".join(
            line.strip()
            for line in lines
            if line.strip()
            and not line.startswith(">")
        )

    else:

        print(
            "Enter the protein sequence."
        )

        sequence = input(
            "> "
        )

    predict(
        sequence
    )


if __name__ == "__main__":
    main()
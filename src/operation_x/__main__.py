import argparse
import json
from pathlib import Path

from operation_x.prediction import predict

from operation_x.ml.large_crystallization_dataset import (
    LargeCrystallizationDataset,
)

from operation_x.ml.crystallization_condition_ranker import (
    CrystallizationConditionRanker,
)


# ============================================================
# PATHS
# ============================================================

LARGE_DATASET = (
    "data/structures/large/records.json"
)

RECORDS_FILE = Path(
    "data/structures/large/records.json"
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Operation X - "
            "Protein Crystallization Prediction"
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
        help="Path to a FASTA file",
    )

    return parser.parse_args()


# ============================================================
# FASTA
# ============================================================

def load_fasta(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        lines = file.readlines()

    sequence_parts = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if line.startswith(">"):
            continue

        sequence_parts.append(line)

    return "".join(sequence_parts)


# ============================================================
# GET SEQUENCE
# ============================================================

def get_sequence(args):

    if args.sequence and args.fasta:

        raise ValueError(
            "Use either --sequence or --fasta, "
            "not both."
        )

    if args.sequence:

        sequence = args.sequence

    elif args.fasta:

        sequence = load_fasta(
            args.fasta
        )

    else:

        raise ValueError(
            "Please provide a protein sequence "
            "using --sequence or --fasta."
        )

    sequence = "".join(
        sequence.split()
    ).upper()

    if not sequence:

        raise ValueError(
            "Protein sequence cannot be empty."
        )

    valid_amino_acids = set(
        "ACDEFGHIKLMNPQRSTVWY"
    )

    invalid = (
        set(sequence)
        - valid_amino_acids
    )

    if invalid:

        raise ValueError(
            "Invalid amino-acid characters: "
            + ", ".join(
                sorted(invalid)
            )
        )

    return sequence


# ============================================================
# HEADER
# ============================================================

def print_header():

    print()
    print("=" * 70)

    print(
        "OPERATION X - "
        "PROTEIN CRYSTALLIZATION PREDICTION"
    )

    print("=" * 70)


# ============================================================
# PRINT PREDICTIONS
# ============================================================

def print_predictions(
    predictions
):

    print()
    print("=" * 70)
    print("FINAL PREDICTIONS")
    print("=" * 70)

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


# ============================================================
# EXPRESSION DETAILS
# ============================================================

def load_expression_details(
    pdb_id,
):
    """
    Load documented expression metadata for
    an experimental PDB record.

    No expression information is inferred.

    If a field is unavailable in the dataset,
    it is displayed as 'Not reported'.
    """

    default_result = {
        "host": "Not reported",
        "strain": "Not reported",
        "system": "Not reported",
        "inducer": "Not reported",
        "evidence": "Not reported",
    }

    if not RECORDS_FILE.exists():

        return default_result

    try:

        with RECORDS_FILE.open(
            "r",
            encoding="utf-8",
        ) as handle:

            records = json.load(
                handle
            )

    except (
        OSError,
        json.JSONDecodeError,
    ):

        return default_result

    if not isinstance(
        records,
        dict,
    ):

        return default_result

    normalized_id = str(
        pdb_id
    ).strip().upper()

    # Handle possible entity suffixes.
    normalized_id = normalized_id.split(
        "_"
    )[0]

    record = records.get(
        normalized_id
    )

    if record is None:

        # Fallback search in case the JSON
        # keys are formatted differently.

        for key, value in records.items():

            key_normalized = str(
                key
            ).strip().upper().split(
                "_"
            )[0]

            if key_normalized == normalized_id:

                record = value
                break

    if not isinstance(
        record,
        dict,
    ):

        return default_result

    def get_value(
        field_name,
    ):

        value = record.get(
            field_name
        )

        if value is None:

            return "Not reported"

        if isinstance(
            value,
            str,
        ):

            value = value.strip()

            if not value:
                return "Not reported"

        return value

    return {
        "host": get_value(
            "expression_host"
        ),

        "strain": get_value(
            "expression_strain"
        ),

        "system": get_value(
            "expression_system"
        ),

        "inducer": get_value(
            "inducer"
        ),

        "evidence": get_value(
            "expression_evidence"
        ),
    }


# ============================================================
# PRINT RECOMMENDATIONS
# ============================================================

def print_recommendations(
    recommendations
):

    print()
    print("=" * 70)

    print(
        "TOP EXPERIMENTAL CONDITION "
        "RECOMMENDATIONS"
    )

    print("=" * 70)

    if not recommendations:

        print()

        print(
            "No experimental recommendations found."
        )

        return

    print()

    for index, recommendation in enumerate(
        recommendations,
        start=1,
    ):

        print(
            f"Recommendation #{index}"
        )

        print("-" * 70)

        if isinstance(
            recommendation,
            dict,
        ):

            # ------------------------------------------------
            # PDB ID
            # ------------------------------------------------

            pdb_id = recommendation.get(
                "query_pdb"
            )

            if pdb_id:

                print(
                    f"PDB ID: {pdb_id}"
                )

            # ------------------------------------------------
            # Ranking score
            # ------------------------------------------------

            if "score" in recommendation:

                try:

                    print(
                        f"Score: "
                        f"{float(recommendation['score']):.4f}"
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    print(
                        f"Score: "
                        f"{recommendation['score']}"
                    )

            # ------------------------------------------------
            # Crystallization properties
            # ------------------------------------------------

            if "pH" in recommendation:

                print(
                    f"pH: "
                    f"{recommendation['pH']}"
                )

            if "temperature_kelvin" in recommendation:

                print(
                    f"Temperature (K): "
                    f"{recommendation['temperature_kelvin']}"
                )

            if "matthews_coefficient" in recommendation:

                print(
                    f"Matthews coefficient: "
                    f"{recommendation['matthews_coefficient']}"
                )

            if "solvent_percent" in recommendation:

                print(
                    f"Solvent (%): "
                    f"{recommendation['solvent_percent']}"
                )

            # ------------------------------------------------
            # Experimental condition text
            # ------------------------------------------------

            if "condition_text" in recommendation:

                print()

                print(
                    "Condition:"
                )

                print(
                    recommendation[
                        "condition_text"
                    ]
                )

            # ------------------------------------------------
            # Expression details
            # ------------------------------------------------

            if pdb_id:

                expression = (
                    load_expression_details(
                        pdb_id
                    )
                )

                print()

                print(
                    "EXPRESSION DETAILS"
                )

                print("-" * 40)

                print(
                    f"Host              : "
                    f"{expression['host']}"
                )

                print(
                    f"Strain            : "
                    f"{expression['strain']}"
                )

                print(
                    f"Expression system : "
                    f"{expression['system']}"
                )

                print(
                    f"Inducer           : "
                    f"{expression['inducer']}"
                )

                print(
                    f"Evidence           : "
                    f"{expression['evidence']}"
                )

            # ------------------------------------------------
            # Additional recommendation fields
            # ------------------------------------------------

            displayed = {
                "query_pdb",
                "score",
                "pH",
                "temperature_kelvin",
                "matthews_coefficient",
                "solvent_percent",
                "condition_text",
            }

            for key, value in recommendation.items():

                if key in displayed:

                    continue

                print(
                    f"{key}: {value}"
                )

        else:

            print(
                recommendation
            )

        print()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    sequence = get_sequence(
        args
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print_header()

    print()

    print(
        "Input type                  : "
        "Protein sequence"
    )

    print(
        f"Sequence length             : "
        f"{len(sequence)} aa"
    )

    print()

    print(
        "Structure required          : NO"
    )

    print(
        "Target structure used       : NO"
    )

    # --------------------------------------------------------
    # 1. OPERATION X PREDICTION
    # --------------------------------------------------------

    predictions = predict(
        sequence
    )

    # --------------------------------------------------------
    # 2. DISPLAY PREDICTIONS
    # --------------------------------------------------------

    print_predictions(
        predictions
    )

    # --------------------------------------------------------
    # 3. LOAD EXPERIMENTAL DATASET
    # --------------------------------------------------------

    print()

    print(
        "Loading experimental "
        "crystallization dataset..."
    )

    dataset = LargeCrystallizationDataset(
        LARGE_DATASET
    )

    # --------------------------------------------------------
    # 4. CREATE CONDITION RANKER
    # --------------------------------------------------------

    ranker = CrystallizationConditionRanker(
        dataset
    )

    # --------------------------------------------------------
    # 5. RANK EXPERIMENTAL CONDITIONS
    # --------------------------------------------------------

    print()

    print(
        "Ranking experimental "
        "crystallization conditions..."
    )

    recommendations = ranker.rank(
        predicted_ph=predictions[
            "pH"
        ],

        predicted_temperature=predictions[
            "temperature_kelvin"
        ],

        predicted_matthews=predictions[
            "matthews_coefficient"
        ],

        predicted_solvent=predictions[
            "solvent_percent"
        ],

        top_n=5,
    )

    # --------------------------------------------------------
    # 6. DISPLAY RECOMMENDATIONS
    # --------------------------------------------------------

    print_recommendations(
        recommendations
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "Operation X analysis complete."
    )

    print("=" * 70)

    print()

    print(
        "Note: predictions are ranked "
        "experimental-condition estimates, "
        "not guarantees of crystallization success."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
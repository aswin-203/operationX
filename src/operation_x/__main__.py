import argparse

from operation_x.ml.real_pipeline_factory import (
    RealPipelineFactory,
)

from operation_x.ml.large_crystallization_dataset import (
    LargeCrystallizationDataset,
)

from operation_x.ml.crystallization_condition_ranker import (
    CrystallizationConditionRanker,
)


BLAST_DATABASE = "data/blast/pilot/pdb_sequences"

STRUCTURE_DATABASE = "data/foldseek/pilot/structures"

FOLDSEEK_PATH = (
    "/home/aswin/Downloads/foldseek/bin/foldseek"
)

BLASTP_PATH = "/usr/bin/blastp"

LARGE_DATASET = (
    "data/structures/large/records.json"
)


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


def load_fasta(path):

    with open(path, "r") as file:

        lines = file.readlines()

    sequence_parts = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if line.startswith(">"):
            continue

        sequence_parts.append(line)

    sequence = "".join(sequence_parts)

    return sequence


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

    invalid = set(sequence) - valid_amino_acids

    if invalid:

        raise ValueError(
            "Invalid amino-acid characters: "
            + ", ".join(sorted(invalid))
        )

    return sequence


def print_header():

    print()
    print("=" * 70)
    print(
        " " * 20 +
        "OPERATION X"
    )
    print(
        " " * 12 +
        "PROTEIN CRYSTALLIZATION ANALYSIS"
    )
    print("=" * 70)


def print_similarity(evidence):

    sequence = evidence.get(
        "sequence_evidence",
        {},
    )

    structure = evidence.get(
        "structure_evidence",
        {},
    )

    print()
    print("SIMILARITY")
    print("-" * 70)

    # ---------------------------------
    # Sequence similarity
    # ---------------------------------

    identity = sequence.get(
        "best_identity"
    )

    coverage = sequence.get(
        "best_coverage"
    )

    strong_sequence_hits = sequence.get(
        "strong_hit_count",
        0,
    )

    if identity is None:
        print(
            "Best sequence identity      : "
            "No significant hit"
        )
    else:
        print(
            f"Best sequence identity      : "
            f"{float(identity):.2f} %"
        )

    if coverage is None:
        print(
            "Best sequence coverage      : "
            "No significant hit"
        )
    else:
        print(
            f"Best sequence coverage      : "
            f"{float(coverage) * 100:.2f} %"
        )

    print(
        f"Strong sequence hits        : "
        f"{strong_sequence_hits}"
    )

    print()

    # ---------------------------------
    # Structure similarity
    # ---------------------------------

    tm_score = structure.get(
        "best_query_tm_score"
    )

    rmsd = structure.get(
        "best_rmsd"
    )

    strong_structure_hits = structure.get(
        "strong_hit_count",
        0,
    )

    if tm_score is None:
        print(
            "Best structure TM-score     : "
            "Not available"
        )
    else:
        print(
            f"Best structure TM-score     : "
            f"{float(tm_score):.4f}"
        )

    if rmsd is None:
        print(
            "Best structure RMSD         : "
            "Not available"
        )
    else:
        print(
            f"Best structure RMSD         : "
            f"{float(rmsd):.4f} Å"
        )

    print(
        f"Strong structure hits       : "
        f"{strong_structure_hits}"
    )

def print_predictions(predictions):

    print()
    print("PREDICTED CRYSTAL PROPERTIES")
    print("-" * 70)

    print(
        f"pH                          : "
        f"{predictions['pH']:.4f}"
    )

    print(
        f"Temperature                 : "
        f"{predictions['temperature_kelvin']:.4f} K"
    )

    print(
        f"Matthews coefficient        : "
        f"{predictions['matthews_coefficient']:.4f}"
    )

    print(
        f"Solvent percentage          : "
        f"{predictions['solvent_percent']:.4f} %"
    )


def print_recommendations(results):

    print()
    print("RECOMMENDED EXPERIMENTAL")
    print("CRYSTALLIZATION CONDITIONS")
    print("-" * 70)

    if not results:

        print(
            "No compatible experimental "
            "conditions found."
        )

        return

    for index, result in enumerate(
        results,
        start=1,
    ):

        print()

        print(
            f"{index}. Reference PDB: "
            f"{result.get('query_pdb', 'unknown')}"
        )

        print(
            f"   Compatibility score      : "
            f"{result['score']:.4f}"
        )

        temperature = result.get(
            "temperature_kelvin"
        )

        if temperature is not None:

            print(
                f"   Temperature              : "
                f"{float(temperature):.2f} K"
            )

        ph = result.get("pH")

        if ph is not None:

            print(
                f"   pH                       : "
                f"{float(ph):.2f}"
            )

        print()

        print(
            "   Condition:"
        )

        condition = result.get(
            "condition_text"
        )

        if condition:

            print(
                f"   {condition}"
            )


def main():

    args = parse_arguments()

    sequence = get_sequence(args)

    # ---------------------------------------------
    # Target identity
    #
    # This is NOT a PDB ID.
    # It is only an internal label for this input.
    # ---------------------------------------------

    target_id = "INPUT"

    print_header()

    print()
    print(
        "Input type                  : Protein sequence"
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

    # ---------------------------------------------
    # 1. Create the real Operation X pipeline
    # ---------------------------------------------

    factory = RealPipelineFactory(
        blast_database=BLAST_DATABASE,
        structure_database=STRUCTURE_DATABASE,
        foldseek_path=FOLDSEEK_PATH,
        blastp_path=BLASTP_PATH,
    )

    pipeline = factory.create()

    # ---------------------------------------------
    # 2. Run ML prediction
    # ---------------------------------------------

    predictions = pipeline.predict(
        sequence=sequence,
        pdb_id=target_id,
        entity={},
        max_hits=10,
    )

    # ---------------------------------------------
    # 3. Get similarity evidence
    # ---------------------------------------------

    evidence_result = (
        pipeline.similarity_pipeline.analyze(
            pdb_id=target_id,
            sequence=sequence,
            entity={},
            max_hits=10,
        )
    )

    print_similarity(
        evidence_result["evidence"]
    )

    # ---------------------------------------------
    # 4. Print ML predictions
    # ---------------------------------------------

    print_predictions(
        predictions
    )

    # ---------------------------------------------
    # 5. Rank experimental conditions
    # ---------------------------------------------

    dataset = LargeCrystallizationDataset(
        LARGE_DATASET
    )

    ranker = CrystallizationConditionRanker(
        dataset
    )

    recommendations = ranker.rank(
        predicted_ph=predictions["pH"],
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

    print_recommendations(
        recommendations
    )

    print()
    print("=" * 70)
    print(
        "Operation X analysis complete."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()

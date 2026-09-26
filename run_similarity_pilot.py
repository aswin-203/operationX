import json
from pathlib import Path

import pandas as pd

from operation_x.clients.pdb_client import PDBClient

from operation_x.similarity.sequence_search import (
    SequenceSimilarity,
)

from operation_x.similarity.structure_search import (
    StructureSimilarity,
)

from operation_x.similarity.domain_search import (
    DomainSimilarity,
)

from operation_x.similarity.similarity_pipeline import (
    SimilarityPipeline,
)


# ============================================================
# Configuration
# ============================================================

INPUT_DATASET = (
    "data/processed/operation_x_pilot.csv"
)

BLAST_DATABASE = (
    "data/blast/pilot/pdb_sequences"
)

FOLDSEEK_EXECUTABLE = (
    "/home/aswin/Downloads/foldseek/bin/foldseek"
)

FOLDSEEK_DATABASE = (
    "data/foldseek/pilot/structures"
)

STRUCTURE_DIRECTORY = (
    Path("data/structures/pilot")
)

OUTPUT_DIRECTORY = (
    Path("data/similarity/pilot")
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Create clients
# ============================================================

client = PDBClient()

sequence_search = SequenceSimilarity(
    database=BLAST_DATABASE,
)

structure_search = StructureSimilarity(
    executable=FOLDSEEK_EXECUTABLE,
)

domain_similarity = DomainSimilarity()


pipeline = SimilarityPipeline(
    sequence_search=sequence_search,
    structure_search=structure_search,
    domain_similarity=domain_similarity,
    structure_database=FOLDSEEK_DATABASE,
    output_directory=str(
        OUTPUT_DIRECTORY
    ),
)


# ============================================================
# Load pilot dataset
# ============================================================

df = pd.read_csv(
    INPUT_DATASET
)

pdb_ids = (
    df["pdb_id"]
    .dropna()
    .astype(str)
    .tolist()
)


print("=" * 60)
print("OPERATION X — PILOT SIMILARITY DATASET")
print("=" * 60)

print()
print(
    f"Proteins to process: {len(pdb_ids)}"
)

print()


# ============================================================
# Process proteins
# ============================================================

results = []

for index, pdb_id in enumerate(
    pdb_ids,
    start=1,
):

    print("=" * 60)

    print(
        f"[{index}/{len(pdb_ids)}] "
        f"Processing {pdb_id}"
    )

    print("=" * 60)

    structure_path = (
        STRUCTURE_DIRECTORY
        / f"{pdb_id}.cif"
    )

    if not structure_path.exists():

        print(
            f"{pdb_id} -> ERROR: "
            f"structure file not found"
        )

        continue


    try:

        # ----------------------------------------
        # Get polymer entity
        # ----------------------------------------

        entry = client.get_entry(
            pdb_id
        )

        identifiers = entry.get(
            "rcsb_entry_container_identifiers",
            {},
        )

        polymer_ids = identifiers.get(
            "polymer_entity_ids",
            [],
        )

        if not polymer_ids:

            raise ValueError(
                "No polymer entity found"
            )

        entity_id = polymer_ids[0]

        entity = (
            client.get_polymer_entity(
                pdb_id,
                entity_id,
            )
        )


        # ----------------------------------------
        # Extract sequence
        # ----------------------------------------

        entity_poly = entity.get(
            "entity_poly",
            {},
        )

        sequence = (
            entity_poly.get(
                "pdbx_seq_one_letter_code_can"
            )
            or entity_poly.get(
                "pdbx_seq_one_letter_code"
            )
            or ""
        )

        if not sequence:

            raise ValueError(
                "No protein sequence found"
            )


        # ----------------------------------------
        # Run similarity pipeline
        # ----------------------------------------

        result = pipeline.analyze(
            pdb_id=pdb_id,
            sequence=sequence,
            structure_path=str(
                structure_path
            ),
            entity=entity,
            max_hits=10,
        )


        # ----------------------------------------
        # Save individual result
        # ----------------------------------------

        output_file = (
            OUTPUT_DIRECTORY
            / f"{pdb_id}.json"
        )

        output_file.write_text(
            json.dumps(
                result,
                indent=2,
            )
        )


        results.append(result)


        print()
        print(
            f"{pdb_id} -> SUCCESS"
        )

        print(
            f"Sequence hits : "
            f"{len(result['sequence_hits'])}"
        )

        print(
            f"Structure hits: "
            f"{len(result['structure_hits'])}"
        )

        print(
            f"Domains       : "
            f"{len(result['domains'])}"
        )


    except Exception as error:

        print()
        print(
            f"{pdb_id} -> ERROR: "
            f"{error}"
        )


# ============================================================
# Save combined dataset
# ============================================================

combined_file = (
    OUTPUT_DIRECTORY
    / "pilot_similarity.json"
)

combined_file.write_text(
    json.dumps(
        results,
        indent=2,
    )
)


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 60)
print("PILOT SIMILARITY DATASET")
print("=" * 60)

print(
    f"Requested : {len(pdb_ids)}"
)

print(
    f"Successful: {len(results)}"
)

print(
    f"Failed    : "
    f"{len(pdb_ids) - len(results)}"
)

print()

print(
    f"Output directory: "
    f"{OUTPUT_DIRECTORY}"
)

print(
    f"Combined file: "
    f"{combined_file}"
)

print()
print(
    "Pilot similarity analysis completed."
)

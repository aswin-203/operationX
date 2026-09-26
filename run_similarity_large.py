import json
from pathlib import Path

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
# CONFIGURATION
# ============================================================

RECORDS_FILE = (
    "data/structures/large/records.json"
)

STRUCTURE_DIRECTORY = (
    "data/structures/large"
)

BLAST_DATABASE = (
    "data/blast/large/pdb_sequences"
)

FOLDSEEK_EXECUTABLE = (
    "/home/aswin/Downloads/foldseek/bin/foldseek"
)

FOLDSEEK_DATABASE = (
    "data/foldseek/large/structures"
)

OUTPUT_DIRECTORY = (
    "data/similarity/large"
)

OUTPUT_FILE = (
    "data/similarity/large/large_similarity.json"
)

MAX_HITS = 10


# ============================================================
# LOAD RECORDS
# ============================================================

with open(RECORDS_FILE) as f:
    records = json.load(f)

pdb_ids = list(records.keys())


print("=" * 60)
print("OPERATION X — LARGE SIMILARITY DATASET")
print("=" * 60)

print()
print(f"Proteins to process: {len(pdb_ids)}")
print()


# ============================================================
# CREATE COMPONENTS
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
    output_directory=OUTPUT_DIRECTORY,
)


# ============================================================
# PROCESS DATASET
# ============================================================

results = []

successful = 0
failed = 0

Path(OUTPUT_DIRECTORY).mkdir(
    parents=True,
    exist_ok=True,
)


for index, pdb_id in enumerate(
    pdb_ids,
    start=1,
):

    print("=" * 60)

    print(
        f"[{index}/{len(pdb_ids)}] "
        f"Processing {pdb_id}"
    )

    print()

    try:

        record = records[pdb_id]

        sequence = record.get(
            "sequence"
        )

        if not sequence:

            print(
                f"{pdb_id} -> ERROR: "
                "No sequence"
            )

            failed += 1
            continue


        structure_path = (
            Path(STRUCTURE_DIRECTORY)
            / f"{pdb_id}.cif"
        )


        if not structure_path.exists():

            print(
                f"{pdb_id} -> ERROR: "
                "Structure file missing"
            )

            failed += 1
            continue


        entity = client.get_polymer_entity(
            pdb_id,
            "1",
        )


        result = pipeline.analyze(
            pdb_id=pdb_id,
            sequence=sequence,
            structure_path=str(
                structure_path
            ),
            entity=entity,
            max_hits=MAX_HITS,
        )


        # Add crystallization information
        # from the collected records.

        result["crystallization"] = {
            "pH": record.get("pH"),
            "temperature_kelvin": record.get(
                "temperature_kelvin"
            ),
            "pressure": record.get(
                "pressure"
            ),
            "crystallization_method": record.get(
                "crystallization_method"
            ),
            "crystallization_time": record.get(
                "crystallization_time"
            ),
            "matthews_coefficient": record.get(
                "matthews_coefficient"
            ),
            "solvent_percent": record.get(
                "solvent_percent"
            ),
            "crystallization_details": record.get(
                "crystallization_details"
            ),
        }


        # Save individual result

        individual_file = (
            Path(OUTPUT_DIRECTORY)
            / f"{pdb_id}.json"
        )

        with open(
            individual_file,
            "w",
        ) as f:

            json.dump(
                result,
                f,
                indent=2,
            )


        results.append(result)

        successful += 1


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

        failed += 1

        print(
            f"{pdb_id} -> ERROR: "
            f"{error}"
        )


# ============================================================
# SAVE COMBINED DATASET
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
) as f:

    json.dump(
        results,
        f,
        indent=2,
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("LARGE SIMILARITY DATASET")
print("=" * 60)

print(
    f"Requested : {len(pdb_ids)}"
)

print(
    f"Successful: {successful}"
)

print(
    f"Failed    : {failed}"
)

print(
    f"Results   : {len(results)}"
)

print()
print(
    f"Output directory: "
    f"{OUTPUT_DIRECTORY}"
)

print(
    f"Combined file: "
    f"{OUTPUT_FILE}"
)

print()
print(
    "Large similarity analysis completed."
)
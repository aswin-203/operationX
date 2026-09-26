import json
from pathlib import Path

import pandas as pd


SIMILARITY_FILE = (
    "data/similarity/pilot/pilot_similarity.json"
)

CRYSTALLIZATION_FILE = (
    "data/processed/operation_x_pilot.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "operation_x_similarity_conditions.csv"
)


# ============================================================
# Load data
# ============================================================

with open(SIMILARITY_FILE) as f:
    similarity_data = json.load(f)


crystallization_df = pd.read_csv(
    CRYSTALLIZATION_FILE
)


# ============================================================
# Build PDB lookup
# ============================================================

condition_lookup = {}

for _, row in crystallization_df.iterrows():

    pdb_id = str(
        row["pdb_id"]
    ).strip()

    condition_lookup[pdb_id] = row.to_dict()


# ============================================================
# Extract target PDB ID
# ============================================================

def normalize_pdb_id(value):

    if value is None:
        return None

    value = str(value).strip()

    # BLAST format:
    # pdb|101M|1

    if value.startswith("pdb|"):

        parts = value.split("|")

        if len(parts) >= 2:
            return parts[1]

    # Foldseek may return:
    # 101M
    # 10AI_B

    if "_" in value:

        return value.split("_")[0]

    return value


# ============================================================
# Build condition-level records
# ============================================================

records = []


for result in similarity_data:

    query_pdb = str(
        result["pdb_id"]
    ).strip()


    sequence_hits = (
        result.get(
            "sequence_hits",
            [],
        )
    )

    structure_hits = (
        result.get(
            "structure_hits",
            [],
        )
    )

    domains = (
        result.get(
            "domains",
            [],
        )
    )


    # --------------------------------------------------------
    # Index structure hits by PDB
    # --------------------------------------------------------

    structure_by_pdb = {}

    for hit in structure_hits:

        target = normalize_pdb_id(
            hit.get("target")
        )

        if not target:
            continue

        structure_by_pdb[target] = hit


    # --------------------------------------------------------
    # Index domains
    # --------------------------------------------------------

    pfam_ids = {
        domain["domain_id"]
        for domain in domains
        if domain.get("domain_type")
        == "Pfam"
    }

    interpro_ids = {
        domain["domain_id"]
        for domain in domains
        if domain.get("domain_type")
        == "InterPro"
    }


    # --------------------------------------------------------
    # Process BLAST hits
    # --------------------------------------------------------

    for sequence_hit in sequence_hits:

        subject = normalize_pdb_id(
            sequence_hit.get(
                "subject_id"
            )
        )

        if not subject:
            continue


        # Ignore self-hit

        if subject == query_pdb:
            continue


        condition = (
            condition_lookup.get(
                subject
            )
        )

        if condition is None:
            continue


        structure_hit = (
            structure_by_pdb.get(
                subject
            )
        )


        # ----------------------------------------------------
        # Structure evidence
        # ----------------------------------------------------

        if structure_hit:

            query_tm_score = (
                structure_hit.get(
                    "query_tm_score"
                )
            )

            target_tm_score = (
                structure_hit.get(
                    "target_tm_score"
                )
            )

            structure_rmsd = (
                structure_hit.get(
                    "rmsd"
                )
            )

            alignment_length = (
                structure_hit.get(
                    "alignment_length"
                )
            )

            structure_evalue = (
                structure_hit.get(
                    "evalue"
                )
            )

        else:

            query_tm_score = None
            target_tm_score = None
            structure_rmsd = None
            alignment_length = None
            structure_evalue = None


        # ----------------------------------------------------
        # Create condition record
        # ----------------------------------------------------

        record = {

            # Identity
            "query_pdb": query_pdb,

            "similar_pdb": subject,


            # Sequence similarity
            "sequence_identity": (
                sequence_hit.get(
                    "identity"
                )
            ),

            "sequence_coverage": (
                sequence_hit.get(
                    "coverage"
                )
            ),

            "sequence_evalue": (
                sequence_hit.get(
                    "evalue"
                )
            ),

            "sequence_bitscore": (
                sequence_hit.get(
                    "bitscore"
                )
            ),


            # Structure similarity
            "structure_query_tm_score": (
                query_tm_score
            ),

            "structure_target_tm_score": (
                target_tm_score
            ),

            "structure_rmsd": (
                structure_rmsd
            ),

            "structure_alignment_length": (
                alignment_length
            ),

            "structure_evalue": (
                structure_evalue
            ),


            # Domain evidence
            "pfam_count": len(
                pfam_ids
            ),

            "interpro_count": len(
                interpro_ids
            ),

            "pfam_ids": "|".join(
                sorted(pfam_ids)
            ),

            "interpro_ids": "|".join(
                sorted(interpro_ids)
            ),


            # Experimental conditions
            "crystallization_available": (
                condition.get(
                    "crystallization_available"
                )
            ),

            "crystallization_method": (
                condition.get(
                    "crystallization_method"
                )
            ),

            "pH": (
                condition.get("pH")
            ),

            "temperature_kelvin": (
                condition.get(
                    "temperature_kelvin"
                )
            ),

            "pressure": (
                condition.get(
                    "pressure"
                )
            ),

            "crystallization_time": (
                condition.get(
                    "crystallization_time"
                )
            ),

            "crystallization_details": (
                condition.get(
                    "crystallization_details"
                )
            ),

            "matthews_coefficient": (
                condition.get(
                    "matthews_coefficient"
                )
            ),

            "solvent_percent": (
                condition.get(
                    "solvent_percent"
                )
            ),


            # Expression
            "expression_host": (
                condition.get(
                    "expression_host"
                )
            ),

            "expression_strain": (
                condition.get(
                    "expression_strain"
                )
            ),

            "expression_system": (
                condition.get(
                    "expression_system"
                )
            ),

            "inducer": (
                condition.get(
                    "inducer"
                )
            ),
        }


        records.append(record)


# ============================================================
# Create dataframe
# ============================================================

output_df = pd.DataFrame(
    records
)


# ============================================================
# Save
# ============================================================

Path(
    OUTPUT_FILE
).parent.mkdir(
    parents=True,
    exist_ok=True,
)


output_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# Summary
# ============================================================

print("=" * 60)
print("OPERATION X — CONDITION DATASET")
print("=" * 60)

print(
    f"Records: {len(output_df)}"
)

print(
    f"Columns: {len(output_df.columns)}"
)

print()

print(
    f"Output: {OUTPUT_FILE}"
)

print()

if not output_df.empty:

    print(
        "Query proteins:",
        output_df[
            "query_pdb"
        ].nunique(),
    )

    print(
        "Similar proteins:",
        output_df[
            "similar_pdb"
        ].nunique(),
    )

    print()

    print(
        "Crystallization available:"
    )

    print(
        output_df[
            "crystallization_available"
        ].value_counts(
            dropna=False
        )
    )

print()

print(
    output_df[
        [
            "query_pdb",
            "similar_pdb",
            "sequence_identity",
            "sequence_coverage",
            "structure_query_tm_score",
            "pH",
            "temperature_kelvin",
            "crystallization_method",
        ]
    ].head(20).to_string(
        index=False
    )
)

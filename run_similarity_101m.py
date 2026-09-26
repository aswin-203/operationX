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


PDB_ID = "101M"

BLAST_DATABASE = (
    "data/blast/pilot/pdb_sequences"
)

FOLDSEEK_EXECUTABLE = (
    "/home/aswin/Downloads/foldseek/bin/foldseek"
)

FOLDSEEK_DATABASE = (
    "data/foldseek/pilot/structures"
)

STRUCTURE_PATH = (
    f"data/structures/pilot/{PDB_ID}.cif"
)


# -----------------------------------------
# 1. Create clients
# -----------------------------------------

client = PDBClient()

sequence_search = SequenceSimilarity(
    database=BLAST_DATABASE,
)

structure_search = StructureSimilarity(
    executable=FOLDSEEK_EXECUTABLE,
)

domain_similarity = DomainSimilarity()


# -----------------------------------------
# 2. Create pipeline
# -----------------------------------------

pipeline = SimilarityPipeline(
    sequence_search=sequence_search,
    structure_search=structure_search,
    domain_similarity=domain_similarity,
    structure_database=FOLDSEEK_DATABASE,
    output_directory="data/similarity/pilot",
)


# -----------------------------------------
# 3. Get protein information
# -----------------------------------------

print("=" * 60)
print("OPERATION X — REAL SIMILARITY TEST")
print("=" * 60)
print()

print(f"PDB: {PDB_ID}")
print()

entity = client.get_polymer_entity(
    PDB_ID,
    "1",
)

sequence = (
    entity["entity_poly"]
    .get(
        "pdbx_seq_one_letter_code_can"
    )
    or entity["entity_poly"]
    .get(
        "pdbx_seq_one_letter_code"
    )
)

if not sequence:
    raise ValueError(
        f"No sequence found for {PDB_ID}"
    )


# -----------------------------------------
# 4. Run complete pipeline
# -----------------------------------------

result = pipeline.analyze(
    pdb_id=PDB_ID,
    sequence=sequence,
    structure_path=STRUCTURE_PATH,
    entity=entity,
    max_hits=10,
)


# -----------------------------------------
# 5. Print BLAST results
# -----------------------------------------

print("=" * 60)
print("SEQUENCE SIMILARITY — BLAST")
print("=" * 60)

for hit in result["sequence_hits"]:

    print(
        f"{hit['subject_id']} | "
        f"identity: {hit['identity']:.3f}% | "
        f"coverage: {hit['coverage']:.3f} | "
        f"evalue: {hit['evalue']} | "
        f"bitscore: {hit['bitscore']}"
    )


# -----------------------------------------
# 6. Print Foldseek results
# -----------------------------------------

print()
print("=" * 60)
print("STRUCTURE SIMILARITY — FOLDSEEK")
print("=" * 60)

for hit in result["structure_hits"]:

    print(
        f"{hit['target']} | "
        f"query TM-score: "
        f"{hit['query_tm_score']:.4f} | "
        f"target TM-score: "
        f"{hit['target_tm_score']:.4f} | "
        f"RMSD: {hit['rmsd']:.3f} | "
        f"alignment: {hit['alignment_length']} | "
        f"evalue: {hit['evalue']}"
    )


# -----------------------------------------
# 7. Print domain annotations
# -----------------------------------------

print()
print("=" * 60)
print("DOMAIN ANNOTATIONS")
print("=" * 60)

for domain in result["domains"]:

    start = domain["start"]
    end = domain["end"]

    location = ""

    if start is not None and end is not None:
        location = (
            f" | residues: {start}-{end}"
        )

    print(
        f"{domain['domain_type']} | "
        f"{domain['domain_id']} | "
        f"{domain['domain_name']}"
        f"{location}"
    )


# -----------------------------------------
# 8. Print fused evidence
# -----------------------------------------

evidence = result["evidence"]

sequence_evidence = (
    evidence["sequence_evidence"]
)

structure_evidence = (
    evidence["structure_evidence"]
)

domain_evidence = (
    evidence["domain_evidence"]
)


print()
print("=" * 60)
print("EVIDENCE FUSION")
print("=" * 60)


# -----------------------------------------
# Sequence evidence
# -----------------------------------------

print()
print("SEQUENCE EVIDENCE")
print("-" * 60)

print(
    f"Best identity       : "
    f"{sequence_evidence['best_identity']:.3f}%"
)

print(
    f"Best coverage       : "
    f"{sequence_evidence['best_coverage']:.3f}"
)

print(
    f"Best E-value        : "
    f"{sequence_evidence['best_evalue']}"
)

print(
    f"Best bitscore       : "
    f"{sequence_evidence['best_bitscore']}"
)

print(
    f"Strong sequence hits: "
    f"{sequence_evidence['strong_hit_count']}"
)


# -----------------------------------------
# Structure evidence
# -----------------------------------------

print()
print("STRUCTURE EVIDENCE")
print("-" * 60)

print(
    f"Best query TM-score : "
    f"{structure_evidence['best_query_tm_score']:.4f}"
)

print(
    f"Best target TM-score: "
    f"{structure_evidence['best_target_tm_score']:.4f}"
)

print(
    f"Best RMSD           : "
    f"{structure_evidence['best_rmsd']:.3f}"
)

print(
    f"Best alignment      : "
    f"{structure_evidence['best_alignment_length']}"
)

print(
    f"Strong structure hits: "
    f"{structure_evidence['strong_hit_count']}"
)


# -----------------------------------------
# Domain evidence
# -----------------------------------------

print()
print("DOMAIN EVIDENCE")
print("-" * 60)

print(
    f"Total domains       : "
    f"{domain_evidence['domain_count']}"
)

print(
    f"Pfam domains        : "
    f"{domain_evidence['pfam_count']}"
)

print(
    f"InterPro domains    : "
    f"{domain_evidence['interpro_count']}"
)

print(
    f"Pfam IDs            : "
    f"{', '.join(domain_evidence['pfam_ids']) or 'None'}"
)

print(
    f"InterPro IDs        : "
    f"{', '.join(domain_evidence['interpro_ids']) or 'None'}"
)


# -----------------------------------------
# 9. Final summary
# -----------------------------------------

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)

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

print(
    f"Strong seq. hits: "
    f"{sequence_evidence['strong_hit_count']}"
)

print(
    f"Strong struct. hits: "
    f"{structure_evidence['strong_hit_count']}"
)

print()
print("Similarity analysis completed.")

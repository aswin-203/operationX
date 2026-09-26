from operation_x.similarity.evidence_fusion import (
    EvidenceFusion,
)


def test_evidence_fusion_summarizes_all_sources():

    fusion = EvidenceFusion()

    sequence_hits = [
        {
            "identity": 100.0,
            "coverage": 1.0,
            "evalue": 1e-100,
            "bitscore": 300.0,
        },
        {
            "identity": 95.0,
            "coverage": 0.95,
            "evalue": 1e-80,
            "bitscore": 250.0,
        },
    ]

    structure_hits = [
        {
            "query_tm_score": 1.0,
            "target_tm_score": 1.0,
            "rmsd": 0.0,
            "alignment_length": 154,
        },
        {
            "query_tm_score": 0.85,
            "target_tm_score": 0.82,
            "rmsd": 1.2,
            "alignment_length": 120,
        },
    ]

    domains = [
        {
            "domain_id": "PF00042",
            "domain_name": "Globin",
            "domain_type": "Pfam",
        },
        {
            "domain_id": "IPR002335",
            "domain_name": "Myoglobin",
            "domain_type": "InterPro",
        },
    ]

    result = fusion.summarize(
        sequence_hits,
        structure_hits,
        domains,
    )

    assert (
        result["sequence_evidence"][
            "best_identity"
        ]
        == 100.0
    )

    assert (
        result["sequence_evidence"][
            "strong_hit_count"
        ]
        == 2
    )

    assert (
        result["structure_evidence"][
            "best_query_tm_score"
        ]
        == 1.0
    )

    assert (
        result["structure_evidence"][
            "strong_hit_count"
        ]
        == 2
    )

    assert (
        result["domain_evidence"][
            "pfam_count"
        ]
        == 1
    )

    assert (
        result["domain_evidence"][
            "interpro_count"
        ]
        == 1
    )


def test_evidence_fusion_handles_empty_results():

    fusion = EvidenceFusion()

    result = fusion.summarize(
        [],
        [],
        [],
    )

    assert (
        result["sequence_evidence"][
            "strong_hit_count"
        ]
        == 0
    )

    assert (
        result["structure_evidence"][
            "strong_hit_count"
        ]
        == 0
    )

    assert (
        result["domain_evidence"][
            "domain_count"
        ]
        == 0
    )

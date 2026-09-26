import pytest

from operation_x.ml.real_feature_builder import (
    RealFeatureBuilder,
)


def test_build_real_features():

    builder = RealFeatureBuilder()

    result = builder.build(
        sequence_features={
            "best_evalue": 1e-20,
            "best_bitscore": 200.0,
            "best_identity": 95.0,
            "best_coverage": 0.90,
            "best_alignment_length": 100.0,
            "sequence_strong_hit_count": 3.0,
        },
        structure_features={
            "structure_strong_hit_count": 2.0,
            "best_rmsd": 1.2,
            "best_query_tm_score": 0.92,
            "best_target_tm_score": 0.90,
        },
        domain_features={
            "domain_count": 2.0,
            "pfam_count": 3.0,
            "interpro_count": 4.0,
        },
    )

    assert isinstance(result, dict)

    assert len(result) == 13

    assert result["best_evalue"] == 1e-20
    assert result["best_bitscore"] == 200.0
    assert result["best_identity"] == 95.0
    assert result["best_coverage"] == 0.90
    assert result["best_alignment_length"] == 100.0
    assert result["sequence_strong_hit_count"] == 3.0

    assert result["structure_strong_hit_count"] == 2.0
    assert result["best_rmsd"] == 1.2
    assert result["best_query_tm_score"] == 0.92
    assert result["best_target_tm_score"] == 0.90

    assert result["domain_count"] == 2.0
    assert result["pfam_count"] == 3.0
    assert result["interpro_count"] == 4.0


def test_missing_features_are_rejected():

    builder = RealFeatureBuilder()

    with pytest.raises(ValueError):

        builder.build(
            sequence_features={},
            structure_features={},
            domain_features={},
        )


def test_feature_count():

    builder = RealFeatureBuilder()

    result = builder.build(
        sequence_features={
            "best_evalue": 1e-10,
            "best_bitscore": 100.0,
            "best_identity": 80.0,
            "best_coverage": 0.8,
            "best_alignment_length": 80.0,
            "sequence_strong_hit_count": 1.0,
        },
        structure_features={
            "structure_strong_hit_count": 1.0,
            "best_rmsd": 2.0,
            "best_query_tm_score": 0.8,
            "best_target_tm_score": 0.8,
        },
        domain_features={
            "domain_count": 1.0,
            "pfam_count": 1.0,
            "interpro_count": 1.0,
        },
    )

    assert len(result) == 13


def test_build_from_evidence():

    builder = RealFeatureBuilder()

    evidence = {
        "sequence_evidence": {
            "best_evalue": 1e-20,
            "best_bitscore": 200.0,
            "best_identity": 95.0,
            "best_coverage": 0.9,
            "best_alignment_length": 100.0,
            "strong_hit_count": 3,
        },
        "structure_evidence": {
            "strong_hit_count": 2,
            "best_rmsd": 1.2,
            "best_query_tm_score": 0.92,
            "best_target_tm_score": 0.90,
        },
        "domain_evidence": {
            "domain_count": 2,
            "pfam_count": 3,
            "interpro_count": 4,
        },
    }

    result = builder.build_from_evidence(evidence)

    assert len(result) == 13
    assert result["best_identity"] == 95.0
    assert result["best_rmsd"] == 1.2
    assert result["pfam_count"] == 3.0

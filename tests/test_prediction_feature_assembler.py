import pytest

from operation_x.ml.prediction_feature_assembler import (
    PredictionFeatureAssembler,
)


def test_assembler_returns_28_features():

    assembler = PredictionFeatureAssembler()

    physicochemical = {
        "sequence_length": 100.0,
        "valid_sequence_length": 100.0,
        "molecular_weight": 11000.0,
        "isoelectric_point": 7.2,
        "gravy": 0.1,
        "instability_index": 35.0,
        "aromaticity": 0.08,
        "charged_residue_fraction": 0.20,
        "polar_residue_fraction": 0.25,
        "hydrophobic_residue_fraction": 0.40,
        "cysteine_fraction": 0.02,
        "glycine_fraction": 0.08,
        "proline_fraction": 0.05,
        "ambiguous_residue_count": 0.0,
        "ambiguous_residue_fraction": 0.0,
    }

    similarity = {
        "best_evalue": 1e-20,
        "best_bitscore": 200.0,
        "best_identity": 95.0,
        "best_coverage": 0.9,
        "best_alignment_length": 100.0,
        "sequence_strong_hit_count": 3.0,
        "structure_strong_hit_count": 2.0,
        "best_rmsd": 1.2,
        "best_query_tm_score": 0.92,
        "best_target_tm_score": 0.90,
        "domain_count": 2.0,
        "pfam_count": 3.0,
        "interpro_count": 4.0,
    }

    result = assembler.assemble(
        physicochemical=physicochemical,
        similarity=similarity,
    )

    assert isinstance(result, dict)
    assert len(result) == 28


def test_assembler_contains_exact_feature_set():

    assembler = PredictionFeatureAssembler()

    physicochemical = {
        "sequence_length": 100.0,
        "valid_sequence_length": 100.0,
        "molecular_weight": 11000.0,
        "isoelectric_point": 7.2,
        "gravy": 0.1,
        "instability_index": 35.0,
        "aromaticity": 0.08,
        "charged_residue_fraction": 0.20,
        "polar_residue_fraction": 0.25,
        "hydrophobic_residue_fraction": 0.40,
        "cysteine_fraction": 0.02,
        "glycine_fraction": 0.08,
        "proline_fraction": 0.05,
        "ambiguous_residue_count": 0.0,
        "ambiguous_residue_fraction": 0.0,
    }

    similarity = {
        "best_evalue": 1e-20,
        "best_bitscore": 200.0,
        "best_identity": 95.0,
        "best_coverage": 0.9,
        "best_alignment_length": 100.0,
        "sequence_strong_hit_count": 3.0,
        "structure_strong_hit_count": 2.0,
        "best_rmsd": 1.2,
        "best_query_tm_score": 0.92,
        "best_target_tm_score": 0.90,
        "domain_count": 2.0,
        "pfam_count": 3.0,
        "interpro_count": 4.0,
    }

    result = assembler.assemble(
        physicochemical=physicochemical,
        similarity=similarity,
    )

    expected = {
        "sequence_length",
        "valid_sequence_length",
        "molecular_weight",
        "isoelectric_point",
        "gravy",
        "instability_index",
        "aromaticity",
        "charged_residue_fraction",
        "polar_residue_fraction",
        "hydrophobic_residue_fraction",
        "cysteine_fraction",
        "glycine_fraction",
        "proline_fraction",
        "ambiguous_residue_count",
        "ambiguous_residue_fraction",
        "best_evalue",
        "best_bitscore",
        "best_identity",
        "best_coverage",
        "best_alignment_length",
        "sequence_strong_hit_count",
        "structure_strong_hit_count",
        "best_rmsd",
        "best_query_tm_score",
        "best_target_tm_score",
        "domain_count",
        "pfam_count",
        "interpro_count",
    }

    assert set(result.keys()) == expected


def test_missing_feature_is_rejected():

    assembler = PredictionFeatureAssembler()

    physicochemical = {
        "sequence_length": 100.0,
    }

    similarity = {}

    with pytest.raises(ValueError):
        assembler.assemble(
            physicochemical=physicochemical,
            similarity=similarity,
        )

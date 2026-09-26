import pytest

from operation_x.ml.real_prediction_pipeline import (
    RealPredictionPipeline,
)


class FakeInputFeatureBuilder:

    def build(self, sequence):
        return {
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


class FakeSimilarityPipeline:

    def analyze(
        self,
        pdb_id,
        sequence,
        entity,
        max_hits=10,
    ):
        return {
            "pdb_id": pdb_id,
            "evidence": {
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
            },
        }


class FakePredictor:

    def __init__(
        self,
        target,
        prediction,
    ):
        self.target = target
        self.prediction = prediction
        self.fitted = False
        self.received_features = None

    def fit(self):
        self.fitted = True

    def predict(self, features):
        self.received_features = features

        return {
            "target": self.target,
            "prediction": self.prediction,
        }


def test_real_prediction_pipeline():

    predictors = {
        "pH": FakePredictor("pH", 7.2),
        "temperature_kelvin": FakePredictor(
            "temperature_kelvin",
            298.15,
        ),
        "matthews_coefficient": FakePredictor(
            "matthews_coefficient",
            2.4,
        ),
        "solvent_percent": FakePredictor(
            "solvent_percent",
            48.0,
        ),
    }

    pipeline = RealPredictionPipeline(
        input_feature_builder=FakeInputFeatureBuilder(),
        similarity_pipeline=FakeSimilarityPipeline(),
        predictors=predictors,
    )

    result = pipeline.predict(
        sequence="M" * 100,
        pdb_id="TEST",
        entity={},
    )

    assert result["pH"] == 7.2
    assert result["temperature_kelvin"] == 298.15
    assert result["matthews_coefficient"] == 2.4
    assert result["solvent_percent"] == 48.0


def test_pipeline_passes_28_features_to_predictors():

    predictors = {
        target: FakePredictor(target, 1.0)
        for target in [
            "pH",
            "temperature_kelvin",
            "matthews_coefficient",
            "solvent_percent",
        ]
    }

    pipeline = RealPredictionPipeline(
        input_feature_builder=FakeInputFeatureBuilder(),
        similarity_pipeline=FakeSimilarityPipeline(),
        predictors=predictors,
    )

    pipeline.predict(
        sequence="M" * 100,
        pdb_id="TEST",
        entity={},
    )

    for predictor in predictors.values():
        assert predictor.fitted
        assert predictor.received_features is not None
        assert len(predictor.received_features) == 28


def test_pipeline_requires_sequence():

    pipeline = RealPredictionPipeline(
        input_feature_builder=FakeInputFeatureBuilder(),
        similarity_pipeline=FakeSimilarityPipeline(),
        predictors={},
    )

    with pytest.raises(ValueError):

        pipeline.predict(
            sequence="",
            pdb_id="TEST",
            entity={},
        )

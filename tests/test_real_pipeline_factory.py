from operation_x.ml.real_pipeline_factory import (
    RealPipelineFactory,
)


def test_factory_creates_pipeline():

    factory = RealPipelineFactory(
        blast_database="data/blast/pilot/pdb_sequences",
        structure_database="data/foldseek/pilot/structures",
    )

    pipeline = factory.create()

    assert pipeline is not None
    assert pipeline.similarity_pipeline is not None
    assert pipeline.input_feature_builder is not None
    assert pipeline.predictors is not None

    assert set(pipeline.predictors.keys()) == {
        "pH",
        "temperature_kelvin",
        "matthews_coefficient",
        "solvent_percent",
    }

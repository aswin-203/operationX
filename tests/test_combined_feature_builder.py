import pandas as pd

from operation_x.ml.combined_feature_builder import (
    CombinedFeatureBuilder,
)


def test_build_target(tmp_path):

    similarity_dir = tmp_path / "similarity"
    similarity_dir.mkdir()

    (
        similarity_dir / "ph_features.csv"
    ).write_text(
        """sequence_identity,sequence_coverage
99.0,1.0
80.0,0.8
"""
    )

    physicochemical_file = (
        tmp_path / "physicochemical.csv"
    )

    physicochemical_file.write_text(
        """pdb_id,sequence_length,molecular_weight
AAA1,150,17000
AAA2,200,22000
"""
    )

    targets_dir = tmp_path / "targets"
    metadata_dir = tmp_path / "metadata"

    targets_dir.mkdir()
    metadata_dir.mkdir()

    (
        targets_dir / "ph_targets.csv"
    ).write_text(
        """pH
7.0
8.0
"""
    )

    (
        metadata_dir / "ph_metadata.csv"
    ).write_text(
        """query_pdb,target
AAA1,pH
AAA2,pH
"""
    )

    builder = CombinedFeatureBuilder(
        similarity_features_directory=str(
            similarity_dir
        ),
        physicochemical_features_file=str(
            physicochemical_file
        ),
        targets_directory=str(
            targets_dir
        ),
        metadata_directory=str(
            metadata_dir
        ),
        output_directory=str(
            tmp_path / "output"
        ),
    )

    result = builder.build_target("pH")

    assert len(result) == 2
    assert "query_pdb" in result.columns
    assert "sequence_identity" in result.columns
    assert "sequence_coverage" in result.columns
    assert "sequence_length" in result.columns
    assert "molecular_weight" in result.columns
    assert "pH" in result.columns

    assert result["query_pdb"].tolist() == [
        "AAA1",
        "AAA2",
    ]

    assert result["pH"].tolist() == [
        7.0,
        8.0,
    ]


def test_save(tmp_path):

    similarity_dir = tmp_path / "similarity"
    similarity_dir.mkdir()

    for prefix in [
        "ph",
        "temperature_kelvin",
        "matthews_coefficient",
        "solvent_percent",
    ]:

        (
            similarity_dir
            / f"{prefix}_features.csv"
        ).write_text(
            """sequence_identity
99.0
80.0
"""
        )

    physicochemical_file = (
        tmp_path / "physicochemical.csv"
    )

    physicochemical_file.write_text(
        """pdb_id,sequence_length
AAA1,150
AAA2,200
"""
    )

    targets_dir = tmp_path / "targets"
    metadata_dir = tmp_path / "metadata"

    targets_dir.mkdir()
    metadata_dir.mkdir()

    target_names = {
        "ph": "pH",
        "temperature_kelvin": "temperature_kelvin",
        "matthews_coefficient": "matthews_coefficient",
        "solvent_percent": "solvent_percent",
    }

    for prefix, target in target_names.items():

        (
            targets_dir
            / f"{prefix}_targets.csv"
        ).write_text(
            f"{target}\n"
            "1.0\n"
            "2.0\n"
        )

        (
            metadata_dir
            / f"{prefix}_metadata.csv"
        ).write_text(
            "query_pdb,target\n"
            f"AAA1,{target}\n"
            f"AAA2,{target}\n"
        )

    builder = CombinedFeatureBuilder(
        similarity_features_directory=str(
            similarity_dir
        ),
        physicochemical_features_file=str(
            physicochemical_file
        ),
        targets_directory=str(
            targets_dir
        ),
        metadata_directory=str(
            metadata_dir
        ),
        output_directory=str(
            tmp_path / "output"
        ),
    )

    result = builder.save()

    assert len(result) == 4

    for target, info in result.items():

        assert info["rows"] == 2
        assert info["feature_count"] > 0

        output_file = (
            tmp_path
            / "output"
            / f"{target}_combined.csv"
        )

        assert output_file.exists()

        df = pd.read_csv(output_file)

        assert len(df) == 2
        assert "query_pdb" in df.columns
        assert target in df.columns
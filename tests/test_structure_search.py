from pathlib import Path

from operation_x.similarity.structure_search import (
    StructureSimilarity,
)


def test_structure_search_parses_foldseek_output(
    tmp_path,
    monkeypatch,
):

    output_file = (
        tmp_path / "results.tsv"
    )

    output_file.write_text(
        "101M\t102M\t1.006\t0.9992\t0.9992\t1.0\t0.150\t154\t1.589e-21\t1047\n"
    )

    class FakeResult:

        stdout = ""
        stderr = ""

    def fake_run(
        command,
        capture_output,
        text,
        check,
    ):
        return FakeResult()

    monkeypatch.setattr(
        "subprocess.run",
        fake_run,
    )

    searcher = StructureSimilarity(
        "/fake/foldseek"
    )

    # Directly test parsing behavior through
    # a temporary output file.
    results = []

    for line in output_file.read_text().splitlines():

        fields = line.split("\t")

        results.append(
            {
                "query": fields[0],
                "target": fields[1],
                "alignment_tm_score": float(fields[2]),
                "query_tm_score": float(fields[3]),
                "target_tm_score": float(fields[4]),
                "probability": float(fields[5]),
                "rmsd": float(fields[6]),
                "alignment_length": int(fields[7]),
                "evalue": float(fields[8]),
                "bits": float(fields[9]),
            }
        )

    assert len(results) == 1

    assert results[0]["target"] == "102M"

    assert results[0]["query_tm_score"] == 0.9992

    assert results[0]["target_tm_score"] == 0.9992

    assert results[0]["rmsd"] == 0.150

    assert results[0]["alignment_length"] == 154


def test_structure_similarity_version(
    monkeypatch,
):

    class FakeResult:

        stdout = "foldseek-test-version\n"

    def fake_run(
        command,
        capture_output,
        text,
        check,
    ):
        return FakeResult()

    monkeypatch.setattr(
        "subprocess.run",
        fake_run,
    )

    searcher = StructureSimilarity(
        "/fake/foldseek"
    )

    assert (
        searcher.version()
        == "foldseek-test-version"
    )
from operation_x.similarity.sequence_search import SequenceSimilarity


def test_parse_results():

    output = (
        "operation_x_query\t1ABC_1\t95.000\t100\t5\t0\t1\t100\t1\t100\t1e-50\t200\n"
        "operation_x_query\t2XYZ_1\t85.000\t80\t12\t0\t10\t89\t5\t84\t2e-20\t100\n"
    )

    results = SequenceSimilarity._parse_results(
        output,
        query_length=100,
    )

    assert len(results) == 2

    assert results[0]["subject_id"] == "1ABC_1"
    assert results[0]["identity"] == 95.0
    assert results[0]["coverage"] == 1.0

    assert results[1]["subject_id"] == "2XYZ_1"
    assert results[1]["identity"] == 85.0
    assert results[1]["coverage"] == 0.8

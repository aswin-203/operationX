from operation_x.dataset.collector import DatasetCollector


def test_collect_candidates():

    collector = DatasetCollector()

    result = collector.collect(
        [
            "101M",
            "102L",
            "102M",
        ]
    )

    assert isinstance(result, dict)

    assert "records" in result
    assert "rejected" in result
    assert "errors" in result

    assert len(result["records"]) > 0

    assert "102L" in result["rejected"]

    assert len(result["errors"]) == 0

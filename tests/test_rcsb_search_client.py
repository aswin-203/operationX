from operation_x.clients.rcsb_search_client import (
    RCSBSearchClient,
)


def test_search_xray_entries():

    client = RCSBSearchClient()

    query = {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": "exptl.method",
            "operator": "exact_match",
            "value": "X-RAY DIFFRACTION",
        },
    }

    results = client.search(
        query=query,
        rows=5,
    )

    assert isinstance(results, list)
    assert len(results) > 0
    assert all(
        isinstance(pdb_id, str)
        for pdb_id in results
    )


def test_search_protein_xray():

    client = RCSBSearchClient()

    results = client.search_protein_xray(
        rows=10
    )

    assert isinstance(results, list)
    assert len(results) > 0
    assert all(
        isinstance(pdb_id, str)
        for pdb_id in results
    )


def test_search_supports_pagination():

    client = RCSBSearchClient()

    query = {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": "exptl.method",
            "operator": "exact_match",
            "value": "X-RAY DIFFRACTION",
        },
    }

    first_page = client.search(
        query=query,
        rows=5,
        start=0,
    )

    second_page = client.search(
        query=query,
        rows=5,
        start=5,
    )

    assert isinstance(first_page, list)
    assert isinstance(second_page, list)

    assert len(first_page) == 5
    assert len(second_page) == 5

    assert first_page != second_page
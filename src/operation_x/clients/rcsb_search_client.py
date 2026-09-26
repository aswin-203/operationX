import requests


class RCSBSearchClient:
    SEARCH_URL = (
        "https://search.rcsb.org/rcsbsearch/v2/query"
    )

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def search(
        self,
        query: dict,
        return_type: str = "entry",
        rows: int = 100,
        start: int = 0,
    ) -> list[str]:

        if rows <= 0:
            raise ValueError(
                "rows must be greater than zero"
            )

        if start < 0:
            raise ValueError(
                "start must be zero or greater"
            )

        payload = {
            "query": query,
            "return_type": return_type,
            "request_options": {
                "paginate": {
                    "start": start,
                    "rows": rows,
                }
            },
        }

        response = requests.post(
            self.SEARCH_URL,
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        return [
            result["identifier"]
            for result in data.get(
                "result_set",
                []
            )
        ]

    def search_protein_xray(
        self,
        rows: int = 100,
        start: int = 0,
    ) -> list[str]:

        query = {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": (
                            "entity_poly."
                            "rcsb_entity_polymer_type"
                        ),
                        "operator": "exact_match",
                        "value": "Protein",
                    },
                },
            ],
        }

        return self.search(
            query=query,
            rows=rows,
            start=start,
        )
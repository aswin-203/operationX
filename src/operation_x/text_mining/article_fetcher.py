import requests


class ArticleFetcher:
    """Fetch article metadata and abstract from PubMed."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def fetch_pubmed(self, pubmed_id: int | str) -> dict:
        pubmed_id = str(pubmed_id).strip()

        if not pubmed_id:
            raise ValueError("PubMed ID cannot be empty")

        params = {
            "db": "pubmed",
            "id": pubmed_id,
            "retmode": "json",
        }

        response = requests.get(
            f"{self.BASE_URL}/esummary.fcgi",
            params=params,
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()

    def fetch_abstract(self, pubmed_id: int | str) -> str:
        pubmed_id = str(pubmed_id).strip()

        if not pubmed_id:
            raise ValueError("PubMed ID cannot be empty")

        params = {
            "db": "pubmed",
            "id": pubmed_id,
            "rettype": "abstract",
            "retmode": "text",
        }

        response = requests.get(
            f"{self.BASE_URL}/efetch.fcgi",
            params=params,
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.text

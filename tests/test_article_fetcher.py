from operation_x.text_mining.article_fetcher import ArticleFetcher


def test_fetch_pubmed():
    fetcher = ArticleFetcher()

    data = fetcher.fetch_pubmed(7704533)

    assert isinstance(data, dict)
    assert "result" in data


def test_fetch_abstract():
    fetcher = ArticleFetcher()

    text = fetcher.fetch_abstract(7704533)

    assert isinstance(text, str)
    assert len(text) > 0

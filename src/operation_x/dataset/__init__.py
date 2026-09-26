def __init__(self):
    self.client = PDBClient()
    self.mmcif_parser = MMCIFParser()
    self.article_fetcher = ArticleFetcher()
    self.expression_extractor = ExpressionExtractor()
    self.domain_similarity = DomainSimilarity()
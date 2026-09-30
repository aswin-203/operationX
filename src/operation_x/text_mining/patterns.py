EXPRESSION_HOST_PATTERNS = [
    r"\bEscherichia coli\b",
    r"\bE\.?\s*coli\b",
    r"\bBL21(?:\(DE3\))?\b",
    r"\bHEK293\b",
    r"\bSf9\b",
    r"\bPichia pastoris\b",
    r"\bSaccharomyces cerevisiae\b",
    r"\bCHO cells?\b",
    r"\binsect cells?\b",
]

EXPRESSION_KEYWORDS = [
    "expressed",
    "expression",
    "recombinant protein",
    "transformed",
    "induced",
    "cultured",
]

INDUCER_PATTERNS = [
    r"\bIPTG\b",
    r"\bgalactose\b",
    r"\barabinose\b",
]

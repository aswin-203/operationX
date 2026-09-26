
from operation_x.text_mining.expression_extractor import (
    ExpressionExtractor,
)


def test_extract_expression_host_and_strain():
    text = (
        "The recombinant protein was expressed "
        "in Escherichia coli BL21(DE3) cells."
    )

    extractor = ExpressionExtractor()

    result = extractor.extract(text)

    assert result["expression_host"] == "Escherichia coli"
    assert result["expression_strain"] == "BL21(DE3)"
    assert result["expression_system"] == "bacterial"


def test_extract_inducer():
    text = (
        "Cells were induced with IPTG "
        "for protein expression."
    )

    extractor = ExpressionExtractor()

    result = extractor.extract(text)

    assert result["inducer"] == "IPTG"


def test_extract_no_expression_information():
    text = (
        "The purified protein was analyzed "
        "using X-ray crystallography."
    )

    extractor = ExpressionExtractor()

    result = extractor.extract(text)

    assert result["expression_host"] is None
    assert result["expression_strain"] is None
    assert result["expression_system"] is None
    assert result["inducer"] is None


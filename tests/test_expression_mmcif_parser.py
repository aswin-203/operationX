from operation_x.parsers.expression_mmcif_parser import (
    ExpressionMMCIFParser,
)


def test_parse_ecoli_bl21_expression_metadata():
    cif_text = """
data_test

loop_
_entity_src_gen.entity_id
_entity_src_gen.pdbx_host_org_scientific_name
_entity_src_gen.pdbx_host_org_strain
_entity_src_gen.pdbx_host_org_variant
_entity_src_gen.pdbx_host_org_cell_line
_entity_src_gen.pdbx_host_org_vector_type
_entity_src_gen.pdbx_host_org_vector
_entity_src_gen.expression_system_id
_entity_src_gen.plasmid_name
_entity_src_gen.pdbx_description
1 'Escherichia coli BL21(DE3)' ? Lemo21 ? plasmid ? ? pQE80L ?
"""

    parser = ExpressionMMCIFParser()

    result = parser.parse(cif_text)

    assert result.host == "Escherichia coli"
    assert result.strain == "BL21(DE3)"
    assert result.system == "bacterial"

    assert "mmCIF host: Escherichia coli BL21(DE3)" in (
        result.evidence
    )

    assert "mmCIF variant: Lemo21" in result.evidence
    assert "mmCIF plasmid: pQE80L" in result.evidence


def test_parse_explicit_strain():
    cif_text = """
data_test

loop_
_entity_src_gen.entity_id
_entity_src_gen.pdbx_host_org_scientific_name
_entity_src_gen.pdbx_host_org_strain
_entity_src_gen.pdbx_host_org_variant
1 'Escherichia coli' 'PHAGE RESISTANT TB1' ?
"""

    parser = ExpressionMMCIFParser()

    result = parser.parse(cif_text)

    assert result.host == "Escherichia coli"
    assert result.strain == "PHAGE RESISTANT TB1"
    assert result.system == "bacterial"


def test_parse_cell_line():
    cif_text = """
data_test

loop_
_entity_src_gen.entity_id
_entity_src_gen.pdbx_host_org_scientific_name
_entity_src_gen.pdbx_host_org_strain
_entity_src_gen.pdbx_host_org_variant
_entity_src_gen.pdbx_host_org_cell_line
1 ? ? ? Sf9
"""

    parser = ExpressionMMCIFParser()

    result = parser.parse(cif_text)

    assert result.host == "Sf9"
    assert result.strain == "Sf9"
    assert result.system == "cell-based"


def test_parse_missing_expression_metadata():
    cif_text = """
data_test

loop_
_entity_src_gen.entity_id
_entity_src_gen.pdbx_host_org_scientific_name
_entity_src_gen.pdbx_host_org_strain
_entity_src_gen.pdbx_host_org_variant
1 ? ? ?
"""

    parser = ExpressionMMCIFParser()

    result = parser.parse(cif_text)

    assert result.host is None
    assert result.strain is None
    assert result.system is None
    assert result.inducer is None
    assert result.evidence == []


def test_parse_real_10qh_file():
    with open(
        "data/structures/large/10QH.cif",
        encoding="utf-8",
    ) as file:
        cif_text = file.read()

    parser = ExpressionMMCIFParser()

    result = parser.parse(cif_text)

    assert result.host == "Escherichia coli"
    assert result.strain == "BL21(DE3)"
    assert result.system == "bacterial"

    assert any(
        "Lemo21" in evidence
        for evidence in result.evidence
    )

    assert any(
        "pQE80L" in evidence
        for evidence in result.evidence
    )

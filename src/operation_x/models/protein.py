from pydantic import BaseModel, Field


class ProteinRecord(BaseModel):
    pdb_id: str
    entity_id: str

    title: str | None = None

    sequence: str
    sequence_length: int

    molecular_weight: float | None = None

    experimental_method: str | None = None

    chain_ids: list[str] = Field(default_factory=list)

    source_organism: str | None = None
    expression_host: str | None = None

    uniprot_ids: list[str] = Field(default_factory=list)

    pubmed_id: int | None = None
    doi: str | None = None


from pydantic import BaseModel, Field


class ExpressionRecord(BaseModel):
    host: str | None = None
    strain: str | None = None
    system: str | None = None
    inducer: str | None = None

    evidence: list[str] = Field(
        default_factory=list
    )

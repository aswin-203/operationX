from pydantic import BaseModel, Field


class CrystallizationRecord(BaseModel):
    available: bool = False

    method: str | None = None

    pH: float | None = None
    pH_range: str | None = None

    temperature_kelvin: float | None = None
    temperature_details: str | None = None

    pressure: float | None = None
    time: str | None = None

    apparatus: str | None = None
    atmosphere: str | None = None
    seeding: str | None = None

    details: str | None = None

    matthews_coefficient: float | None = None
    solvent_percent: float | None = None

    raw_conditions: dict = Field(default_factory=dict)

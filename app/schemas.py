from enum import Enum

from pydantic import BaseModel, Field


class Category(str, Enum):
    GPU = "gpu"
    CPU = "cpu"
    RAM = "ram"
    MOTHERBOARD = "motherboard"
    PSU = "psu"
    STORAGE = "storage"
    COOLER = "cooler"
    CASE = "case"
    LAPTOP = "laptop"
    FULL_SYSTEM = "full_system"
    MONITOR = "monitor"
    OTHER = "other"


class Condition(str, Enum):
    WORKING = "working"
    BROKEN = "broken"
    FOR_PARTS = "for_parts"
    UNKNOWN = "unknown"


class RawListing(BaseModel):
    title: str
    description: str = ""
    price: float | None = None
    source: str = "unknown"
    url: str | None = None


class ExtractedListing(BaseModel):
    category: Category
    brand: str | None = None
    model: str | None = None
    variant: str | None = None
    condition: Condition
    defect: str | None = Field(
        default=None, description="Short description of the fault, null if condition is working"
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str | None = Field(
        default=None, description="One short sentence on what drove the category/condition call"
    )


class ExtractionResult(BaseModel):
    listing: RawListing
    extracted: ExtractedListing
    latency_ms: float
    model_used: str

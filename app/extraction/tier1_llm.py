import json
import time

from ollama import Client

from app.schemas import ExtractedListing, ExtractionResult, RawListing

DEFAULT_MODEL = "qwen2.5:3b-instruct"

SYSTEM_PROMPT = """You are a component-identification assistant for a computer hardware resale tool.
Extract structured data from the listing title and description below.

- category: one of gpu, cpu, ram, motherboard, psu, storage, cooler, case, laptop, full_system, monitor, other.
  Use full_system for a whole prebuilt/custom PC even if individual parts are named.
- brand/model/variant: extract the literal words from the text when stated (brand = manufacturer, model =
  product line/number, variant = sub-designation like "Ti" or "FTW3 Ultra"). Use "" only when that field is
  truly not mentioned at all.
- condition: "broken" if the text names ANY specific fault, even if the item still partially works. "for_parts"
  if the seller says it is totally non-functional with no specific fault named. "working" if no fault at all is
  mentioned. "unknown" only if there is zero signal either way - this should be rare.
- defect: short phrase naming the fault, taken from the text. "" unless condition is "broken".
- confidence: 0.0-1.0, how confident you are in this extraction given how specific and unambiguous the text was.
- reasoning: one short sentence, specific to this listing, on what drove your condition call.

Base every field strictly on the actual listing text below - do not reuse wording from any other example.
Use "" for not-applicable fields, never the words "null" or "none". Respond with JSON only."""

# Ollama's structured-output grammar (llama.cpp) is biased toward the null branch of an
# `anyOf: [string, null]` schema regardless of prompt content - it will null out fields like
# brand/model even when they're stated plainly in the text. Pydantic's model_json_schema()
# produces exactly that shape for Optional[str] fields, so we hand-build a schema with plain
# (non-nullable) string types instead, using "" as the sentinel for "not applicable", and
# normalize "" back to None before handing the payload to the pydantic model.
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": [c.value for c in ExtractedListing.model_fields["category"].annotation],
        },
        "brand": {"type": "string", "description": "Manufacturer, or \"\" if not stated"},
        "model": {"type": "string", "description": "Product model/number, or \"\" if not stated"},
        "variant": {"type": "string", "description": "Sub-designation, or \"\" if not stated"},
        "condition": {
            "type": "string",
            "enum": [c.value for c in ExtractedListing.model_fields["condition"].annotation],
        },
        "defect": {"type": "string", "description": "Fault description, or \"\" if condition is working"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning": {"type": "string"},
    },
    "required": ["category", "brand", "model", "variant", "condition", "defect", "confidence", "reasoning"],
}

_EMPTY_SENTINELS = {"", "null", "none", "n/a", "unknown"}


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    if value.strip().lower() in _EMPTY_SENTINELS:
        return None
    return value


class Tier1Extractor:
    def __init__(self, model: str = DEFAULT_MODEL, host: str | None = None):
        self.model = model
        self.client = Client(host=host) if host else Client()

    def extract(self, listing: RawListing) -> ExtractionResult:
        price_str = f"${listing.price:.2f}" if listing.price is not None else "unknown"
        user_prompt = (
            f"Title: {listing.title}\n"
            f"Description: {listing.description or '(none)'}\n"
            f"Asking price: {price_str}"
        )

        start = time.perf_counter()
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format=EXTRACTION_SCHEMA,
            options={"temperature": 0.1},
        )
        latency_ms = (time.perf_counter() - start) * 1000

        content = response["message"]["content"]
        try:
            data = json.loads(content)
            for field in ("brand", "model", "variant", "defect", "reasoning"):
                data[field] = _blank_to_none(data.get(field))
            extracted = ExtractedListing.model_validate(data)
        except Exception as e:
            raise ValueError(f"Model returned invalid extraction JSON: {content!r}") from e

        return ExtractionResult(
            listing=listing,
            extracted=extracted,
            latency_ms=latency_ms,
            model_used=self.model,
        )

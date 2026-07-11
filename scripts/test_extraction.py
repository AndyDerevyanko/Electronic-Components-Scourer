import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.extraction.tier1_llm import Tier1Extractor
from app.schemas import RawListing

SAMPLE_LISTINGS = [
    RawListing(
        title="EVGA RTX 3080 FTW3 Ultra 10GB - works perfectly, upgrading to a 4090",
        description="Great card, no issues, ran it for a year with no crashes. Comes with original box.",
        price=380.0,
        source="test",
    ),
    RawListing(
        title="Broken GPU lot - MSI RTX 2070 no display, artifacts on boot, sold as-is for parts",
        description="Pulled from a dead machine, screen stays black or shows artifacts. Not tested beyond that.",
        price=60.0,
        source="test",
    ),
    RawListing(
        title="Gaming PC bundle, i7, 16gb ram, GTX 1660, powers on but no post - project machine",
        description="Fans spin and lights come on but never posts to BIOS. Selling as a project.",
        price=150.0,
        source="test",
    ),
    RawListing(
        title="Ryzen 5 5600x CPU only, pulled from working build",
        description="Upgraded to a 7800x3d, this one worked fine until removed.",
        price=90.0,
        source="test",
    ),
    RawListing(
        title="Corsair RM850x PSU, clicking noise under load, otherwise powers on",
        description="Noticed a clicking sound when gaming, everything still boots and runs though.",
        price=35.0,
        source="test",
    ),
    RawListing(
        title="Lot of DDR4 RAM sticks, assorted brands, untested",
        description="Pulled from various old builds, never tested, selling as a bundle.",
        price=20.0,
        source="test",
    ),
]


def main() -> None:
    extractor = Tier1Extractor()
    print(f"Using model: {extractor.model}\n")

    total_start = time.perf_counter()
    for listing in SAMPLE_LISTINGS:
        result = extractor.extract(listing)
        e = result.extracted
        print(f"TITLE: {listing.title}  (price=${listing.price:.2f})")
        print(
            f"  -> category={e.category.value} brand={e.brand} model={e.model} variant={e.variant}\n"
            f"     condition={e.condition.value} defect={e.defect!r} confidence={e.confidence:.2f}\n"
            f"     reasoning={e.reasoning!r}\n"
            f"     latency={result.latency_ms:.0f}ms"
        )
        print()

    total_elapsed = time.perf_counter() - total_start
    print(f"Total: {total_elapsed:.1f}s for {len(SAMPLE_LISTINGS)} listings "
          f"({total_elapsed / len(SAMPLE_LISTINGS):.1f}s/listing avg)")


if __name__ == "__main__":
    main()

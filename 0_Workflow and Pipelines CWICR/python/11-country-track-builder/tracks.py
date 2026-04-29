"""
Track registry — single source of truth for all DDC CWICR country tracks.

A "track" is one country/language/currency variant of the CWICR dataset.
Both existing tracks (shipped in the repo) and target tracks (to be built
by add_country_track.py) are declared here.

Replaces the local DEFAULT_PARQUET_FILES dict in
07-multi-language-comparison/compare_regions.py — all consumers should
import from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

TrackStatus = Literal["existing", "target"]


@dataclass(frozen=True)
class Track:
    """Single country/region track."""

    folder: str            # directory name, e.g. "DE___DDC_CWICR"
    region: str            # file-prefix region tag, e.g. "DE_BERLIN"
    iso_country: str       # ISO 3166-1 alpha-2, e.g. "DE"
    language: str          # ISO 639-1 lowercase, e.g. "de"
    currency: str          # ISO 4217 alpha-3, e.g. "EUR"
    region_label: str      # human-readable, e.g. "Germany / Berlin"
    status: TrackStatus    # "existing" (shipped) or "target" (to build)
    source_track: str | None = None   # for targets: which track to clone

    @property
    def parquet_path(self) -> Path:
        return (
            REPO_ROOT / self.folder
            / f"{self.region}_workitems_costs_resources_DDC_CWICR.parquet"
        )

    @property
    def catalog_csv_path(self) -> Path:
        return REPO_ROOT / self.folder / f"DDC_CWICR_{self.region}_Catalog.csv"

    @property
    def catalog_xlsx_path(self) -> Path:
        return REPO_ROOT / self.folder / f"DDC_CWICR_{self.region}_Catalog.xlsx"

    @property
    def simple_xlsx_path(self) -> Path:
        return (
            REPO_ROOT / self.folder
            / f"{self.region}_workitems_costs_resources_DDC_CWICR_SIMPLE.xlsx"
        )

    @property
    def formatted_xlsx_path(self) -> Path:
        return (
            REPO_ROOT / self.folder
            / f"{self.region}_workitems_costs_resources_DDC_CWICR_FORMATTED.xlsx"
        )

    @property
    def embeddings_snapshot_path(self) -> Path:
        return (
            REPO_ROOT / self.folder
            / f"{self.region}_workitems_costs_resources_EMBEDDINGS_3072_DDC_CWICR.snapshot"
        )

    @property
    def readme_path(self) -> Path:
        return REPO_ROOT / self.folder / "README.md"

    @property
    def qdrant_collection(self) -> str:
        # Convention: lowercase region prefix. Matches existing snapshots
        # (ddc_de_berlin, ddc_uk_gbp, ddc_zh_shanghai, etc.) and gives a
        # human-readable name aligned with the file_prefix on disk.
        return f"ddc_{self.region.lower()}"

    def all_paths(self) -> dict[str, Path]:
        """All 10 expected files for this track."""
        return {
            "parquet": self.parquet_path,
            "catalog_csv": self.catalog_csv_path,
            "catalog_xlsx": self.catalog_xlsx_path,
            "simple_xlsx": self.simple_xlsx_path,
            "formatted_xlsx": self.formatted_xlsx_path,
            "embeddings_snapshot": self.embeddings_snapshot_path,
            "readme_md": self.readme_path,
            "readme_pdf": REPO_ROOT / self.folder
            / f"README_DDC_CWICR_TABULAR_{self.region}.pdf",
            "readme_txt": REPO_ROOT / self.folder
            / f"README_DDC_CWICR_TABULAR_{self.region}.txt",
        }


# ---------------------------------------------------------------------------
# Existing tracks (shipped in the repo).
# Region prefixes match the actual filenames on disk.
# ---------------------------------------------------------------------------
EXISTING_TRACKS: dict[str, Track] = {
    "AR_DUBAI": Track(
        folder="AR___DDC_CWICR", region="AR_DUBAI",
        iso_country="AE", language="ar", currency="AED",
        region_label="UAE / Dubai", status="existing",
    ),
    "DE_BERLIN": Track(
        folder="DE___DDC_CWICR", region="DE_BERLIN",
        iso_country="DE", language="de", currency="EUR",
        region_label="Germany / Berlin", status="existing",
    ),
    "ENG_TORONTO": Track(
        folder="EN___DDC_CWICR", region="ENG_TORONTO",
        iso_country="CA", language="en", currency="CAD",
        region_label="Canada / Toronto", status="existing",
    ),
    "SP_BARCELONA": Track(
        folder="ES___DDC_CWICR", region="SP_BARCELONA",
        iso_country="ES", language="es", currency="EUR",
        region_label="Spain / Barcelona", status="existing",
    ),
    "FR_PARIS": Track(
        folder="FR___DDC_CWICR", region="FR_PARIS",
        iso_country="FR", language="fr", currency="EUR",
        region_label="France / Paris", status="existing",
    ),
    "HI_MUMBAI": Track(
        folder="HI___DDC_CWICR", region="HI_MUMBAI",
        iso_country="IN", language="hi", currency="INR",
        region_label="India / Mumbai", status="existing",
    ),
    "PT_SAOPAULO": Track(
        folder="PT___DDC_CWICR", region="PT_SAOPAULO",
        iso_country="BR", language="pt", currency="BRL",
        region_label="Brazil / São Paulo", status="existing",
    ),
    "RU_STPETERSBURG": Track(
        folder="RU___DDC_CWICR", region="RU_STPETERSBURG",
        iso_country="RU", language="ru", currency="RUB",
        region_label="Russia / St. Petersburg", status="existing",
    ),
    "UK_GBP": Track(
        folder="UK___DDC_CWICR", region="UK_GBP",
        iso_country="GB", language="en", currency="GBP",
        region_label="United Kingdom / London", status="existing",
    ),
    "USA_USD": Track(
        folder="US___DDC_CWICR", region="USA_USD",
        iso_country="US", language="en", currency="USD",
        region_label="United States", status="existing",
    ),
    "ZH_SHANGHAI": Track(
        folder="ZH___DDC_CWICR", region="ZH_SHANGHAI",
        iso_country="CN", language="zh", currency="CNY",
        region_label="China / Shanghai", status="existing",
    ),
}


# ---------------------------------------------------------------------------
# Target tracks (to be built by add_country_track.py).
# Source-track choice rule: same currency → same language → UK_GBP fallback.
# UK_GBP is the canonical English source (93 cols, clean schema). EN_TORONTO
# has legacy column names (_eur suffixes, prais_* transliterations) and is
# NOT used as a source. US not used either (anomalous extras: location_factor,
# has_obfuscated_values, cad_to_usd_rate).
# ---------------------------------------------------------------------------
TARGET_TRACKS: dict[str, Track] = {
    "AU_SYDNEY": Track(
        folder="AU___DDC_CWICR", region="AU_SYDNEY",
        iso_country="AU", language="en", currency="AUD",
        region_label="Australia / Sydney", status="target",
        source_track="UK_GBP",
    ),
    "HR_ZAGREB": Track(
        folder="HR___DDC_CWICR", region="HR_ZAGREB",
        iso_country="HR", language="hr", currency="EUR",
        region_label="Croatia / Zagreb", status="target",
        source_track="DE_BERLIN",
    ),
    "BG_SOFIA": Track(
        folder="BG___DDC_CWICR", region="BG_SOFIA",
        iso_country="BG", language="bg", currency="BGN",
        region_label="Bulgaria / Sofia", status="target",
        source_track="DE_BERLIN",
    ),
    "JA_TOKYO": Track(
        folder="JA___DDC_CWICR", region="JA_TOKYO",
        iso_country="JP", language="ja", currency="JPY",
        region_label="Japan / Tokyo", status="target",
        source_track="UK_GBP",
    ),
    "KO_SEOUL": Track(
        folder="KO___DDC_CWICR", region="KO_SEOUL",
        iso_country="KR", language="ko", currency="KRW",
        region_label="South Korea / Seoul", status="target",
        source_track="UK_GBP",
    ),
    "IT_ROME": Track(
        folder="IT___DDC_CWICR", region="IT_ROME",
        iso_country="IT", language="it", currency="EUR",
        region_label="Italy / Rome", status="target",
        source_track="DE_BERLIN",
    ),
    "NL_AMSTERDAM": Track(
        folder="NL___DDC_CWICR", region="NL_AMSTERDAM",
        iso_country="NL", language="nl", currency="EUR",
        region_label="Netherlands / Amsterdam", status="target",
        source_track="DE_BERLIN",
    ),
    "PL_WARSAW": Track(
        folder="PL___DDC_CWICR", region="PL_WARSAW",
        iso_country="PL", language="pl", currency="PLN",
        region_label="Poland / Warsaw", status="target",
        source_track="DE_BERLIN",
    ),
    "SV_STOCKHOLM": Track(
        folder="SV___DDC_CWICR", region="SV_STOCKHOLM",
        iso_country="SE", language="sv", currency="SEK",
        region_label="Sweden / Stockholm", status="target",
        source_track="DE_BERLIN",
    ),
    "CS_PRAGUE": Track(
        folder="CS___DDC_CWICR", region="CS_PRAGUE",
        iso_country="CZ", language="cs", currency="CZK",
        region_label="Czech Republic / Prague", status="target",
        source_track="DE_BERLIN",
    ),
    "TR_ISTANBUL": Track(
        folder="TR___DDC_CWICR", region="TR_ISTANBUL",
        iso_country="TR", language="tr", currency="TRY",
        region_label="Turkey / Istanbul", status="target",
        source_track="DE_BERLIN",
    ),
    "ID_JAKARTA": Track(
        folder="ID___DDC_CWICR", region="ID_JAKARTA",
        iso_country="ID", language="id", currency="IDR",
        region_label="Indonesia / Jakarta", status="target",
        source_track="UK_GBP",
    ),
    "VI_HANOI": Track(
        folder="VI___DDC_CWICR", region="VI_HANOI",
        iso_country="VN", language="vi", currency="VND",
        region_label="Vietnam / Hanoi", status="target",
        source_track="UK_GBP",
    ),
    "TH_BANGKOK": Track(
        folder="TH___DDC_CWICR", region="TH_BANGKOK",
        iso_country="TH", language="th", currency="THB",
        region_label="Thailand / Bangkok", status="target",
        source_track="UK_GBP",
    ),
    "RO_BUCHAREST": Track(
        folder="RO___DDC_CWICR", region="RO_BUCHAREST",
        iso_country="RO", language="ro", currency="RON",
        region_label="Romania / Bucharest", status="target",
        source_track="DE_BERLIN",
    ),
    "MX_MEXICOCITY": Track(
        folder="MX___DDC_CWICR", region="MX_MEXICOCITY",
        iso_country="MX", language="es", currency="MXN",
        region_label="Mexico / Mexico City", status="target",
        source_track="SP_BARCELONA",
    ),
    "NZ_AUCKLAND": Track(
        folder="NZ___DDC_CWICR", region="NZ_AUCKLAND",
        iso_country="NZ", language="en", currency="NZD",
        region_label="New Zealand / Auckland", status="target",
        source_track="UK_GBP",
    ),
    "NG_LAGOS": Track(
        folder="NG___DDC_CWICR", region="NG_LAGOS",
        iso_country="NG", language="en", currency="NGN",
        region_label="Nigeria / Lagos", status="target",
        source_track="UK_GBP",
    ),
    "ZA_JOHANNESBURG": Track(
        folder="ZA___DDC_CWICR", region="ZA_JOHANNESBURG",
        iso_country="ZA", language="en", currency="ZAR",
        region_label="South Africa / Johannesburg", status="target",
        source_track="UK_GBP",
    ),
}


ALL_TRACKS: dict[str, Track] = {**EXISTING_TRACKS, **TARGET_TRACKS}


# ---------------------------------------------------------------------------
# Reference track for schema parity checks.
# DE_BERLIN is canonical (93 cols, no US-only extras).
# ---------------------------------------------------------------------------
SCHEMA_REFERENCE = "DE_BERLIN"


def get_track(region: str) -> Track:
    if region not in ALL_TRACKS:
        raise KeyError(
            f"Unknown track: {region}. "
            f"Known: {sorted(ALL_TRACKS)}"
        )
    return ALL_TRACKS[region]


def existing_parquet_paths() -> dict[str, Path]:
    """Drop-in replacement for compare_regions.py DEFAULT_PARQUET_FILES."""
    return {region: t.parquet_path for region, t in EXISTING_TRACKS.items()}

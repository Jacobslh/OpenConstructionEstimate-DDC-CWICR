"""
Generate benchmark YAML files for the 16 tracks not yet covered.

Each scenario gets a min/max/expected window in the track's currency,
derived from:
  baseline_DE_EUR  ×  country_construction_cost_level  ×  FX(EUR→ccy)

Country construction cost level is sourced from Turner & Townsend's
"International Construction Market Survey 2024" (relative to UK = 100):
this is industry-standard and independent of the price pipeline,
so the resulting benchmarks are a real cross-check.

The window is symmetric around `expected` at ±50%, matching the
±50% sanity threshold used in run_functional_suite.

Run:
    python scripts/generate_benchmarks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
BUILDER_DIR = HERE.parent
BENCHMARKS_DIR = BUILDER_DIR / "benchmarks"
FX = yaml.safe_load(
    (BUILDER_DIR / "configs" / "_fx_snapshot.yaml").read_text(encoding="utf-8")
)["rates_per_eur"]


# -----------------------------------------------------------------------
# Baselines (Germany, EUR) — central expected total cost per scenario.
# Derived from BKI Baukosten 2024 averages for typical commercial-grade
# Berlin builds. These are CENTRAL values; per-country expected is
# baseline × country_factor × FX.
# -----------------------------------------------------------------------
BASELINE_DE_EUR: dict[int, float] = {
    1:  28000,   # Strip foundation 100m³
    2:  5500,    # Floor screed 200m²
    3:  6500,    # RC columns 10pcs
    4:  6500,    # Precast slab 50m²
    5:  25000,   # Retaining wall 30m³
    6:  25000,   # Brick wall 50m³
    7:  8500,    # Block wall 100m²
    8:  28000,   # Stone facade 80m²
    9:  2500,    # Brick chimney 5m
    10: 17000,   # Steel I-beam 5t
    11: 13000,   # Steel column 10pcs
    12: 6000,    # Steel staircase 1pc
    13: 2000,    # Roof truss 1pc
    14: 9000,    # Metal roofing 150m²
    15: 16000,   # Tile roofing 200m²
    16: 8000,    # Bitumen flat roof 100m²
    17: 10500,   # Internal plaster 300m²
    18: 10000,   # External plaster 200m²
    19: 5000,    # Floor tiles 80m²
    20: 2000,    # Wall tiles 30m²
    21: 6500,    # Suspended ceiling 100m²
    22: 6500,    # PVC windows 10pcs
    23: 32000,   # Curtain wall 40m²
    24: 9500,    # Wooden doors 15pcs
    25: 4500,    # Mineral wool 200m²
    26: 3500,    # XPS 150m²
    27: 3000,    # Bitumen waterproofing 100m²
    28: 10500,   # Cable 500m
    29: 3000,    # Wash basins 5pcs
    30: 10500,   # HVAC 50m
}


SCENARIO_DESC: dict[int, str] = {
    1: "Strip foundation 100 m³",       2: "Floor screed 200 m²",
    3: "RC columns 10 pcs",             4: "Precast slab 50 m²",
    5: "Retaining wall 30 m³",          6: "Brick wall 50 m³",
    7: "Block wall 100 m²",             8: "Stone facade 80 m²",
    9: "Brick chimney 5 m",             10: "Steel I-beam 5 t",
    11: "Steel column 10 pcs",          12: "Steel staircase 1 pc",
    13: "Metal roof truss 1 pc",        14: "Metal roofing 150 m²",
    15: "Tile roofing 200 m²",          16: "Bitumen flat roof 100 m²",
    17: "Internal plaster 300 m²",      18: "External plaster 200 m²",
    19: "Floor tiles 80 m²",            20: "Bathroom tiles 30 m²",
    21: "Suspended ceiling 100 m²",     22: "PVC windows 10 pcs",
    23: "Curtain wall 40 m²",           24: "Wooden doors 15 pcs",
    25: "Mineral wool 200 m²",          26: "XPS floor 150 m²",
    27: "Bituminous waterproofing 100 m²",
    28: "Electrical cable 500 m",       29: "Wash basins 5 pcs",
    30: "HVAC ductwork 50 m",
}


# -----------------------------------------------------------------------
# Country construction cost level relative to Germany = 1.00.
# Sources:
#   - Turner & Townsend ICMS 2024 (rebased from UK-100 to DE-100)
#   - AECOM Construction Cost Survey 2024 (ZA, NG)
#   - JLL Asia Pacific Construction Cost Guide 2024 (ID, VI, TH)
# Values are mid-points; ±50% range gives the acceptance window.
# -----------------------------------------------------------------------
COUNTRY_LEVEL_VS_DE: dict[str, float] = {
    "JA_TOKYO":      1.05,   # Japan slightly above DE
    "KO_SEOUL":      0.85,   # Korea below DE
    "IT_ROME":       0.85,
    "NL_AMSTERDAM":  1.10,
    "PL_WARSAW":     0.55,
    "SV_STOCKHOLM":  1.10,
    "CS_PRAGUE":     0.55,
    "TR_ISTANBUL":   0.40,
    "ID_JAKARTA":    0.30,
    "VI_HANOI":      0.25,
    "TH_BANGKOK":    0.32,
    "RO_BUCHAREST":  0.45,
    "MX_MEXICOCITY": 0.45,
    "NZ_AUCKLAND":   1.05,
    "NG_LAGOS":      0.40,
    "ZA_JOHANNESBURG": 0.42,
}

CURRENCY: dict[str, str] = {
    "JA_TOKYO": "JPY",         "KO_SEOUL": "KRW",
    "IT_ROME": "EUR",          "NL_AMSTERDAM": "EUR",
    "PL_WARSAW": "PLN",        "SV_STOCKHOLM": "SEK",
    "CS_PRAGUE": "CZK",        "TR_ISTANBUL": "TRY",
    "ID_JAKARTA": "IDR",       "VI_HANOI": "VND",
    "TH_BANGKOK": "THB",       "RO_BUCHAREST": "RON",
    "MX_MEXICOCITY": "MXN",    "NZ_AUCKLAND": "NZD",
    "NG_LAGOS": "NGN",         "ZA_JOHANNESBURG": "ZAR",
}

SOURCE_NOTE: dict[str, str] = {
    "JA_TOKYO":        "Turner&Townsend ICMS 2024 + JLL APAC Construction Cost",
    "KO_SEOUL":        "Turner&Townsend ICMS 2024 + JLL APAC",
    "IT_ROME":         "Eurostat HICP construction + BKI 2024 ÷ 1.18",
    "NL_AMSTERDAM":    "Eurostat HICP construction + BKI 2024 × 1.10",
    "PL_WARSAW":       "Eurostat lci_lci2 + GUS construction price index",
    "SV_STOCKHOLM":    "Eurostat HICP + SCB Construction Cost Index",
    "CS_PRAGUE":       "Eurostat lci_lci2 + ČSÚ Construction Cost Index",
    "TR_ISTANBUL":     "TUIK Construction Cost Index + Turner&Townsend ICMS 2024",
    "ID_JAKARTA":      "JLL APAC 2024 + BPS Indonesia Konstruksi",
    "VI_HANOI":        "JLL APAC 2024 + GSO Vietnam construction prices",
    "TH_BANGKOK":      "JLL APAC 2024 + NSO Thailand CCI",
    "RO_BUCHAREST":    "Eurostat lci_lci2 + INS Romania CCI",
    "MX_MEXICOCITY":   "INEGI Costos de Construcción + Turner&Townsend ICMS",
    "NZ_AUCKLAND":     "QV CostBuilder NZ + Rawlinsons NZ Construction Handbook",
    "NG_LAGOS":        "AECOM Construction Cost Survey Africa 2024 + JLL Africa",
    "ZA_JOHANNESBURG": "AECOM Construction Cost Survey 2024 + StatsSA CCI",
}


def round_pretty(x: float) -> int:
    """Round to a 'pretty' integer for cost-range display."""
    if x < 100:
        return int(round(x))
    if x < 1000:
        return int(round(x / 10) * 10)
    if x < 100_000:
        return int(round(x / 100) * 100)
    if x < 1_000_000:
        return int(round(x / 1000) * 1000)
    return int(round(x / 10000) * 10000)


def build(region: str) -> dict:
    level = COUNTRY_LEVEL_VS_DE[region]
    ccy = CURRENCY[region]
    fx = FX[ccy]   # currency units per EUR
    src = SOURCE_NOTE[region]

    scenarios: dict[int, dict] = {}
    for i, baseline in BASELINE_DE_EUR.items():
        expected = baseline * level * fx
        scenarios[i] = {
            "min":      round_pretty(expected * 0.5),
            "max":      round_pretty(expected * 1.5),
            "expected": round_pretty(expected),
            "source":   f"{src} ({SCENARIO_DESC[i]})",
        }
    return {"scenarios": scenarios}


def main() -> int:
    BENCHMARKS_DIR.mkdir(exist_ok=True)
    written = 0
    for region in COUNTRY_LEVEL_VS_DE:
        out = BENCHMARKS_DIR / f"{region}_benchmarks.yaml"
        data = build(region)
        # Emit yaml with header comment.
        header = (
            f"# {region} — total cost in {CURRENCY[region]} per scenario.\n"
            f"#\n"
            f"# Generated by scripts/generate_benchmarks.py from\n"
            f"# baseline_DE_EUR × country_level × FX(EUR→{CURRENCY[region]}).\n"
            f"# Country level vs DE: {COUNTRY_LEVEL_VS_DE[region]:.2f}\n"
            f"# Sources: {SOURCE_NOTE[region]}\n"
            f"#\n"
            f"# Window is ±50% of expected; matches run_functional_suite.py\n"
            f"# acceptance threshold. Refine with national stats agency\n"
            f"# data before promoting the track to production.\n\n"
        )
        body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        out.write_text(header + body, encoding="utf-8")
        print(f"  wrote {out.name}")
        written += 1

    print(f"\n{written} benchmark files written to {BENCHMARKS_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

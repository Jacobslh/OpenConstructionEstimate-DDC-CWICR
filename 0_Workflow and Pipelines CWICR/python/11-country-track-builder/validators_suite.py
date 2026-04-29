"""
Functional validation: 30 canonical construction tasks per track.

Heavyweight regression: for each new track, run all 30 scenarios through
the cost-estimation pipeline (uses the new track's Qdrant collection),
then validate against four sources of truth:

  1. Public benchmarks (RSMeans / Rawlinsons / Eurostat / NSI / DZS)
     stored as min/max/expected per scenario per country in
     benchmarks/<TRACK>_benchmarks.yaml.
  2. Cross-track sanity: the result is within ±50% of soft-comparable
     neighbours (HR ≈ IT/PL; AU ≈ NZ/EN_TORONTO; BG ≈ RO/PL).
  3. LLM-judge: GPT-4o reviews each estimate "as a country-X estimator"
     and returns pass/fail with rationale.
  4. Manual expert review: HTML report renders all 30 + flagged-by-LLM
     items for human sign-off.

A track is promoted only when ≥27/30 scenarios pass the combined check
and a human signs off the report.

This module ships only the test set. The runner is invoked from
scripts/run_functional_suite.py to keep the orchestrator free of GPT/
Qdrant deps.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    id: int
    description: str       # English, fed to estimator (auto-translated by it)
    category: str          # high-level discipline
    expected_unit: str     # m2 / m3 / pcs / t / m
    expected_qty: float


SCENARIOS: tuple[Scenario, ...] = (
    # --- Concrete works (5) ---
    Scenario(1,  "100 m³ reinforced concrete strip foundation, depth 1.5 m",
             "concrete", "m3", 100.0),
    Scenario(2,  "200 m² cement screed 50 mm thick on prepared subfloor",
             "concrete", "m2", 200.0),
    Scenario(3,  "10 reinforced concrete columns 400x400 mm, height 3 m",
             "concrete", "pcs", 10.0),
    Scenario(4,  "50 m² precast concrete floor slab installation",
             "concrete", "m2", 50.0),
    Scenario(5,  "30 m³ reinforced concrete retaining wall",
             "concrete", "m3", 30.0),

    # --- Masonry (4) ---
    Scenario(6,  "50 m³ brick wall 380 mm thick, mortar M5",
             "masonry", "m3", 50.0),
    Scenario(7,  "100 m² concrete block wall 200 mm thick",
             "masonry", "m2", 100.0),
    Scenario(8,  "80 m² natural stone facade veneer",
             "masonry", "m2", 80.0),
    Scenario(9,  "5 m brick chimney, height 8 m, single flue",
             "masonry", "m", 5.0),

    # --- Steel & metalwork (4) ---
    Scenario(10, "5 tonnes structural steel I-beams IPE 300, supplied and installed",
             "steel", "t", 5.0),
    Scenario(11, "10 steel column installation, fabricated, height 4 m",
             "steel", "pcs", 10.0),
    Scenario(12, "1 steel staircase, single flight, 18 risers, with railings",
             "steel", "pcs", 1.0),
    Scenario(13, "1 metal roof truss, span 12 m, light-gauge steel",
             "steel", "pcs", 1.0),

    # --- Roofing (3) ---
    Scenario(14, "150 m² metal sheet roofing on timber substructure",
             "roofing", "m2", 150.0),
    Scenario(15, "200 m² clay tile roofing including underlay and battens",
             "roofing", "m2", 200.0),
    Scenario(16, "100 m² bituminous flat roofing, two-layer with insulation",
             "roofing", "m2", 100.0),

    # --- Finishing (5) ---
    Scenario(17, "300 m² internal cement plaster 15 mm with primer",
             "finishing", "m2", 300.0),
    Scenario(18, "200 m² external cement plaster 25 mm with mesh",
             "finishing", "m2", 200.0),
    Scenario(19, "80 m² ceramic floor tiles 600x600 mm including adhesive",
             "finishing", "m2", 80.0),
    Scenario(20, "30 m² bathroom wall ceramic tiles 200x200 mm",
             "finishing", "m2", 30.0),
    Scenario(21, "100 m² suspended ceiling, mineral fibre tiles 600x600 mm",
             "finishing", "m2", 100.0),

    # --- Doors & windows (3) ---
    Scenario(22, "10 PVC windows 1500x1500 mm double-glazed, supplied and fitted",
             "windows", "pcs", 10.0),
    Scenario(23, "40 m² aluminium curtain wall, double-glazed",
             "windows", "m2", 40.0),
    Scenario(24, "15 wooden interior doors with frames, painted finish",
             "doors", "pcs", 15.0),

    # --- Insulation & waterproofing (3) ---
    Scenario(25, "200 m² mineral wool wall insulation 100 mm",
             "insulation", "m2", 200.0),
    Scenario(26, "150 m² extruded polystyrene floor insulation 80 mm",
             "insulation", "m2", 150.0),
    Scenario(27, "100 m² bituminous waterproofing membrane on flat roof",
             "waterproofing", "m2", 100.0),

    # --- MEP (3) ---
    Scenario(28, "500 m electrical cable 3x2.5 mm² in conduit, hidden",
             "electrical", "m", 500.0),
    Scenario(29, "5 wash basins with mixer tap and drain, complete installation",
             "plumbing", "pcs", 5.0),
    Scenario(30, "50 m HVAC galvanised steel ductwork, rectangular 400x250 mm",
             "hvac", "m", 50.0),
)

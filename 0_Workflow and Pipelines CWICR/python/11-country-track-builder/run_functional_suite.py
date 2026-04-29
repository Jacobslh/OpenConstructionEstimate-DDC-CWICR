"""
Functional 30-work regression suite runner.

For one target track:
  1. For each of the 30 scenarios in validators_suite.py, query Qdrant
     for the top-K matching work items, compute an estimate, render an
     HTML row.
  2. Apply the four-method validation:
       a. Public benchmark range — benchmarks/<TRACK>_benchmarks.yaml
       b. Cross-track sanity — neighbour tracks listed in NEIGHBOURS
       c. LLM-judge — GPT-4o reviews "as a country-X estimator"
       d. Manual review — HTML output flagged where a/b/c failed
  3. Promote only when ≥27/30 scenarios pass and a human signs the report.

Required env vars: OPENAI_API_KEY, QDRANT_URL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from tracks import ALL_TRACKS, get_track
from validators_suite import SCENARIOS


HERE = Path(__file__).resolve().parent
BENCHMARKS_DIR = HERE / "benchmarks"
REPORTS_DIR = HERE / "validation_reports"


# Cross-track neighbours used for ±50% sanity check. Picked for similar
# economic / methodological context.
NEIGHBOURS: dict[str, tuple[str, ...]] = {
    "AU_SYDNEY": ("NZ_AUCKLAND", "UK_GBP", "USA_USD"),
    "NZ_AUCKLAND": ("AU_SYDNEY", "UK_GBP"),
    "NG_LAGOS": ("ZA_JOHANNESBURG", "UK_GBP"),
    "ZA_JOHANNESBURG": ("NG_LAGOS", "UK_GBP"),
    "HR_ZAGREB": ("IT_ROME", "PL_WARSAW", "BG_SOFIA", "DE_BERLIN"),
    "BG_SOFIA": ("RO_BUCHAREST", "HR_ZAGREB", "PL_WARSAW"),
    "IT_ROME": ("ES_BARCELONA", "FR_PARIS", "DE_BERLIN"),
    "NL_AMSTERDAM": ("DE_BERLIN", "UK_GBP", "FR_PARIS"),
    "PL_WARSAW": ("CS_PRAGUE", "DE_BERLIN", "RO_BUCHAREST"),
    "SV_STOCKHOLM": ("NL_AMSTERDAM", "DE_BERLIN"),
    "CS_PRAGUE": ("PL_WARSAW", "DE_BERLIN"),
    "TR_ISTANBUL": ("RO_BUCHAREST", "BG_SOFIA"),
    "JA_TOKYO": ("KO_SEOUL", "USA_USD"),
    "KO_SEOUL": ("JA_TOKYO", "USA_USD"),
    "ID_JAKARTA": ("VI_HANOI", "TH_BANGKOK"),
    "VI_HANOI": ("ID_JAKARTA", "TH_BANGKOK"),
    "TH_BANGKOK": ("VI_HANOI", "ID_JAKARTA"),
    "RO_BUCHAREST": ("BG_SOFIA", "PL_WARSAW", "HR_ZAGREB"),
    "MX_MEXICOCITY": ("PT_SAOPAULO", "SP_BARCELONA"),
}


@dataclass
class ScenarioResult:
    scenario_id: int
    description: str
    matches: int                 # how many work items matched in Qdrant
    estimate_total: float        # total estimate in target currency
    currency: str
    benchmark_min: float | None
    benchmark_max: float | None
    benchmark_pass: bool | None
    cross_track_pass: bool | None
    cross_track_detail: str
    llm_judge_pass: bool | None
    llm_judge_reason: str
    overall_pass: bool


# ---------------------------------------------------------------------------
# Qdrant retrieval + estimate
# ---------------------------------------------------------------------------

def _embed(text: str, openai_client) -> list[float]:
    resp = openai_client.embeddings.create(
        model="text-embedding-3-large", input=text,
    )
    return resp.data[0].embedding


def _search(qdrant_client, collection: str, query_emb: list[float], top_k: int = 5):
    return qdrant_client.search(
        collection_name=collection,
        query_vector=query_emb,
        limit=top_k,
    )


def estimate_scenario(
    scenario, track, qdrant_client, openai_client,
) -> tuple[float, int]:
    """Return (estimate_total_in_target_ccy, n_matches)."""
    emb = _embed(scenario.description, openai_client)
    hits = _search(qdrant_client, track.qdrant_collection, emb, top_k=5)
    if not hits:
        return 0.0, 0

    # Use the median total_cost_per_position of the top-K hits as the
    # unit rate, multiply by scenario.expected_qty.
    unit_costs = [
        h.payload.get("total_cost_per_position") or 0.0
        for h in hits
        if h.payload.get("total_cost_per_position") is not None
    ]
    if not unit_costs:
        return 0.0, len(hits)
    unit_costs.sort()
    median = unit_costs[len(unit_costs) // 2]
    return median * scenario.expected_qty, len(hits)


# ---------------------------------------------------------------------------
# Validation methods
# ---------------------------------------------------------------------------

def _load_benchmarks(track_name: str) -> dict[int, tuple[float, float]]:
    """Load per-scenario (min, max) benchmark prices for this track."""
    path = BENCHMARKS_DIR / f"{track_name}_benchmarks.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[int, tuple[float, float]] = {}
    for sid_str, rng in (data.get("scenarios") or {}).items():
        sid = int(sid_str)
        out[sid] = (float(rng["min"]), float(rng["max"]))
    return out


def check_benchmark(
    estimate: float, scenario_id: int, benchmarks: dict,
) -> tuple[bool | None, float | None, float | None]:
    rng = benchmarks.get(scenario_id)
    if rng is None:
        return None, None, None
    lo, hi = rng
    return (lo <= estimate <= hi), lo, hi


def check_cross_track(
    estimate: float, track_name: str, scenario_id: int, neighbour_estimates: dict,
) -> tuple[bool | None, str]:
    """±50% of the median of neighbour estimates."""
    neighbours = NEIGHBOURS.get(track_name, ())
    nest = [
        neighbour_estimates[n][scenario_id]
        for n in neighbours
        if n in neighbour_estimates and scenario_id in neighbour_estimates[n]
    ]
    if not nest:
        return None, "no neighbour data"
    median = sorted(nest)[len(nest) // 2]
    if median <= 0:
        return None, f"neighbour median is 0 (n={len(nest)})"
    ratio = estimate / median
    ok = 0.5 <= ratio <= 1.5
    return ok, f"ratio_to_neighbour_median={ratio:.2f} (n={len(nest)})"


def check_llm_judge(
    scenario, estimate: float, track, openai_client,
) -> tuple[bool, str]:
    sys_prompt = (
        f"You are a senior construction estimator working in {track.region_label}. "
        f"You will be given a building scope and an estimate in {track.currency}. "
        "Decide whether the estimate is realistic for the year 2026 in that region. "
        "Reply with the literal word PASS or FAIL on the first line, then one "
        "short sentence on the second line explaining your reasoning."
    )
    user_prompt = (
        f"Scope: {scenario.description}\n"
        f"Estimate: {estimate:,.2f} {track.currency}\n"
        f"Quantity: {scenario.expected_qty} {scenario.expected_unit}"
    )
    resp = openai_client.chat.completions.create(
        model=os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=120,
        temperature=0.0,
    )
    body = (resp.choices[0].message.content or "").strip()
    first_line = body.splitlines()[0] if body else "FAIL"
    passed = first_line.upper().startswith("PASS")
    reason = body[:200].replace("\n", " ")
    return passed, reason


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def render_report(track, results: list[ScenarioResult], out_path: Path) -> None:
    rows = []
    for r in results:
        flag = "PASS" if r.overall_pass else "FAIL"
        rows.append(
            f"<tr class={'ok' if r.overall_pass else 'fail'}>"
            f"<td>{r.scenario_id}</td>"
            f"<td>{r.description}</td>"
            f"<td>{r.matches}</td>"
            f"<td>{r.estimate_total:,.2f} {r.currency}</td>"
            f"<td>{r.benchmark_min} – {r.benchmark_max}</td>"
            f"<td>{r.benchmark_pass}</td>"
            f"<td>{r.cross_track_pass} ({r.cross_track_detail})</td>"
            f"<td>{r.llm_judge_pass}: {r.llm_judge_reason}</td>"
            f"<td>{flag}</td>"
            f"</tr>"
        )
    n_pass = sum(1 for r in results if r.overall_pass)
    body = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{track.region}</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui; padding: 20px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ border: 1px solid #ccc; padding: 6px; text-align: left; vertical-align: top; }}
  tr.ok td:last-child {{ background: #d4f7d4; }}
  tr.fail td:last-child {{ background: #f7d4d4; }}
  th {{ background: #eef; }}
</style></head><body>
<h1>Functional validation: {track.region_label}</h1>
<p>Track: <code>{track.region}</code> · Currency: {track.currency} · Score: <b>{n_pass}/{len(results)}</b></p>
<p>Promotion threshold: ≥27/30 + manual sign-off below.</p>
<table>
  <thead><tr>
    <th>#</th><th>Scenario</th><th>Matches</th><th>Estimate</th>
    <th>Benchmark range</th><th>Bench OK</th><th>Cross-track OK</th>
    <th>LLM judge</th><th>Overall</th>
  </tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
<h2>Sign-off</h2>
<p>Reviewed by: ___________________ &nbsp; Date: ___________________</p>
</body></html>"""
    out_path.write_text(body, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_for_track(track_name: str) -> int:
    if track_name not in ALL_TRACKS:
        print(f"unknown track {track_name}")
        return 1
    track = get_track(track_name)

    try:
        from openai import OpenAI
        from qdrant_client import QdrantClient
    except ImportError as e:
        print(f"missing dep: {e}")
        return 1

    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    benchmarks = _load_benchmarks(track_name)

    # Compute estimates for THIS track first.
    print(f"\nEstimating {len(SCENARIOS)} scenarios for {track_name} ...")
    estimates: dict[int, float] = {}
    matches: dict[int, int] = {}
    for s in SCENARIOS:
        try:
            est, n = estimate_scenario(s, track, qdrant_client, openai_client)
            estimates[s.id], matches[s.id] = est, n
            print(f"  #{s.id}: {est:,.2f} {track.currency} ({n} matches)")
        except Exception as e:
            estimates[s.id], matches[s.id] = 0.0, 0
            print(f"  #{s.id}: FAILED ({e})")

    # Compute neighbour estimates (cached if available).
    neighbour_estimates: dict[str, dict[int, float]] = {}
    for n_name in NEIGHBOURS.get(track_name, ()):
        cache_path = REPORTS_DIR / f"{n_name}_estimates_cache.json"
        if cache_path.exists():
            neighbour_estimates[n_name] = {
                int(k): v for k, v in json.loads(cache_path.read_text()).items()
            }

    # Validate.
    print("\nValidating ...")
    results: list[ScenarioResult] = []
    for s in SCENARIOS:
        est = estimates[s.id]
        bp, b_lo, b_hi = check_benchmark(est, s.id, benchmarks)
        cp, cd = check_cross_track(est, track_name, s.id, neighbour_estimates)
        try:
            jp, jr = check_llm_judge(s, est, track, openai_client)
        except Exception as e:
            jp, jr = False, f"judge error: {e}"

        # Overall: pass if all of the available checks pass. Missing checks
        # don't block (they aren't FAIL).
        signals = [v for v in (bp, cp, jp) if v is not None]
        overall = bool(signals) and all(signals) and matches[s.id] >= 1

        results.append(ScenarioResult(
            scenario_id=s.id, description=s.description,
            matches=matches[s.id], estimate_total=est, currency=track.currency,
            benchmark_min=b_lo, benchmark_max=b_hi, benchmark_pass=bp,
            cross_track_pass=cp, cross_track_detail=cd,
            llm_judge_pass=jp, llm_judge_reason=jr,
            overall_pass=overall,
        ))

    # Persist for neighbour-cache use by other tracks.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"{track_name}_estimates_cache.json").write_text(
        json.dumps({s.id: estimates[s.id] for s in SCENARIOS}, indent=2),
        encoding="utf-8",
    )

    report_path = REPORTS_DIR / f"{track_name}_estimates.html"
    render_report(track, results, report_path)
    print(f"\nReport: {report_path}")

    n_pass = sum(1 for r in results if r.overall_pass)
    print(f"Score: {n_pass}/{len(results)}")
    if n_pass < 27:
        print("Below promotion threshold (27/30).")
        return 1
    print("Threshold met. Manual sign-off still required in the HTML report.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--track", required=True)
    args = p.parse_args()
    return run_for_track(args.track)


if __name__ == "__main__":
    sys.exit(main())

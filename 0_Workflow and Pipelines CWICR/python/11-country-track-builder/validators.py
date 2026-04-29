"""
Acceptance gate for new country tracks.

Eight schema-level checks that must all pass before a target track is
promoted. Functional 30-work regression suite lives in validators_suite.py
and is run separately because it is heavy (~15 min, ~$3-5 GPT-judge per
track).

Usage:
    python validators.py --track AU_SYDNEY
    python validators.py --track AU_SYDNEY --skip embedding_count
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from tracks import ALL_TRACKS, EXISTING_TRACKS, SCHEMA_REFERENCE, Track, get_track


# ---------------------------------------------------------------------------
# Column buckets — derived from DE_BERLIN canonical schema (93 cols).
# Norms are bytewise-immutable. Prices are recomputed per track.
# Language columns are translated. Codes are stable across tracks.
# ---------------------------------------------------------------------------

NORMS_COLS: tuple[str, ...] = (
    "resource_quantity",
    "parameter_resource_quantity",
    "electricity_consumption_kwh_per_machine_hour",
    "count_workers_per_unit",
    "count_engineers_per_unit",
    "count_operators_per_unit",
    "count_total_people_per_unit",
    "count_people_per_day",
    "labor_hours_construction_workers",
    "labor_hours_operators",
    "labor_hours_engineers",
    "total_labor_hours_workers_operators",
    "total_labor_hours_all_personnel",
    "mass_value",
    "price_abstract_resource_position_count",
)

PRICE_COLS: tuple[str, ...] = (
    "resource_price_per_unit_current",
    "resource_cost",
    "price_abstract_resource_est_price_median",
    "price_abstract_resource_est_price_min",
    "price_abstract_resource_est_price_max",
    "price_abstract_resource_est_price_mean",
    "materials_resource_cost",
    "total_resource_cost_per_position",
    "total_cost_per_position",
    "total_material_cost_per_position",
    "total_value_machinery_equipment",
    "total_value_abstract_resources",
    "price_operator_wages",
    "electricity_cost_per_unit",
    "electricity_cost_total_sum",
    "cost_operator_sum",
    "cost_of_working_hours",
    "service_cost_sum",
    "price_cost_without_wages",
    "labor_rate_per_hr",
)

CODE_COLS: tuple[str, ...] = ("rate_code", "resource_code")
CURRENCY_COL = "currency"

# Sanity bounds for price ratio target/source (after FX conversion).
# Catches obvious config typos (wrong FX rate, forgotten conversion).
PRICE_RATIO_MIN = 0.01
PRICE_RATIO_MAX = 100.0


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}: {self.detail}"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_schema_parity(
    target_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    source_df: pd.DataFrame,
) -> CheckResult:
    """
    Column NAMES match the canonical reference (DE_BERLIN), with
    well-known taxonomy alternates allowed (UK NRM2 instead of CSI
    MasterFormat) and US-style metadata extras tolerated.

    Column DTYPES are checked against the *source* track, not the
    reference — because the pipeline never changes dtypes from source,
    only values. Comparing dtypes against a different track's parquet
    can falsely flag e.g. UK's `bool` columns vs DE's `bool[pyarrow]`.
    """
    target_cols = set(target_df.columns)
    ref_cols = set(ref_df.columns)
    src_cols = set(source_df.columns)
    missing = ref_cols - target_cols
    extra = target_cols - ref_cols

    us_extras = {
        "cad_to_usd_rate", "conversion_date", "has_obfuscated_values",
        "location_factor", "regulatory_note", "source_database",
        "us_labor_class", "us_labor_rate_usd_hr", "us_labor_title",
        "us_operator_class",
    }
    extra = extra - us_extras

    # UK uses NRM2 (RICS) instead of CSI MasterFormat. Either is accepted.
    taxonomy_alternates = {
        ("masterformat_division", "masterformat_section_title"):
            ("nrm2_section_code", "nrm2_section_title"),
    }
    for ref_pair, target_pair in taxonomy_alternates.items():
        if all(c in missing for c in ref_pair) and all(c in extra for c in target_pair):
            for c in ref_pair: missing.discard(c)
            for c in target_pair: extra.discard(c)

    if missing or extra:
        return CheckResult(
            "schema_parity", False,
            f"missing={sorted(missing)} extra={sorted(extra)}",
        )

    # Dtype parity vs source.
    common_with_source = src_cols & target_cols
    type_mismatches = [
        c for c in common_with_source
        if str(target_df[c].dtype) != str(source_df[c].dtype)
    ]
    if type_mismatches:
        return CheckResult(
            "schema_parity", False,
            f"dtype mismatches vs source: {type_mismatches[:5]} (showing first 5)",
        )
    return CheckResult(
        "schema_parity", True,
        f"{len(target_cols)} cols match (against ref names + source dtypes)",
    )


def check_code_stability(target_df: pd.DataFrame, source_df: pd.DataFrame) -> CheckResult:
    """rate_code and resource_code sets must be identical to source track."""
    issues = []
    for col in CODE_COLS:
        t = set(target_df[col].dropna().unique())
        s = set(source_df[col].dropna().unique())
        if t != s:
            missing_in_target = len(s - t)
            extra_in_target = len(t - s)
            issues.append(
                f"{col}: target_missing={missing_in_target} target_extra={extra_in_target}"
            )
    if issues:
        return CheckResult("code_stability", False, "; ".join(issues))
    return CheckResult("code_stability", True, "rate_code + resource_code sets identical")


def check_norms_immutability(
    target_df: pd.DataFrame, source_df: pd.DataFrame,
) -> CheckResult:
    """Norms columns must match source bytewise (after sort by stable codes)."""
    sort_keys = ["rate_code", "resource_code"]
    t = target_df.sort_values(sort_keys).reset_index(drop=True)
    s = source_df.sort_values(sort_keys).reset_index(drop=True)

    if len(t) != len(s):
        return CheckResult(
            "norms_immutability", False,
            f"row count differs: target={len(t)} source={len(s)}",
        )

    diffs = []
    for col in NORMS_COLS:
        if col not in t.columns or col not in s.columns:
            continue
        # Use equals() with tolerance for float NaN treatment.
        if not t[col].equals(s[col]):
            n_diff = (t[col].fillna(-1) != s[col].fillna(-1)).sum()
            diffs.append(f"{col}({n_diff} rows)")
    if diffs:
        return CheckResult(
            "norms_immutability", False,
            f"diverged norms: {diffs[:5]}",
        )
    return CheckResult(
        "norms_immutability", True,
        f"{len(NORMS_COLS)} norm columns bytewise-equal to source",
    )


def check_no_nan_in_prices(
    target_df: pd.DataFrame, source_df: pd.DataFrame,
) -> CheckResult:
    """No new NaN in price columns where source had a value."""
    sort_keys = ["rate_code", "resource_code"]
    t = target_df.sort_values(sort_keys).reset_index(drop=True)
    s = source_df.sort_values(sort_keys).reset_index(drop=True)

    introduced_nans = {}
    for col in PRICE_COLS:
        if col not in t.columns or col not in s.columns:
            continue
        new_nan = t[col].isna() & ~s[col].isna()
        if new_nan.any():
            introduced_nans[col] = int(new_nan.sum())
    if introduced_nans:
        return CheckResult(
            "no_nan_in_prices", False,
            f"introduced NaN: {introduced_nans}",
        )
    return CheckResult("no_nan_in_prices", True, "no NaN regressions in prices")


def check_currency_consistency(target_df: pd.DataFrame, expected_ccy: str) -> CheckResult:
    """All rows declare the target currency (and only the target currency)."""
    if CURRENCY_COL not in target_df.columns:
        return CheckResult(
            "currency_consistency", False,
            f"missing column: {CURRENCY_COL}",
        )
    found = set(target_df[CURRENCY_COL].dropna().unique())
    if found != {expected_ccy}:
        return CheckResult(
            "currency_consistency", False,
            f"expected {{{expected_ccy}}}, found {found}",
        )
    return CheckResult(
        "currency_consistency", True,
        f"all rows in {expected_ccy}",
    )


def check_sanity_ranges(
    target_df: pd.DataFrame, source_df: pd.DataFrame, fx_rate: float,
) -> CheckResult:
    """
    Per-row price ratio target/(source*fx) within [PRICE_RATIO_MIN, PRICE_RATIO_MAX].

    fx_rate is target_ccy per source_ccy (e.g. AUD per CAD).
    Catches forgotten or doubled conversions, decimal-point errors.
    """
    sort_keys = ["rate_code", "resource_code"]
    t = target_df.sort_values(sort_keys).reset_index(drop=True)
    s = source_df.sort_values(sort_keys).reset_index(drop=True)

    out_of_range = {}
    for col in ("total_cost_per_position", "resource_cost"):
        if col not in t.columns or col not in s.columns:
            continue
        mask = (s[col] > 0) & t[col].notna()
        if not mask.any():
            continue
        ratio = t.loc[mask, col] / (s.loc[mask, col] * fx_rate)
        bad = ((ratio < PRICE_RATIO_MIN) | (ratio > PRICE_RATIO_MAX)).sum()
        if bad:
            out_of_range[col] = (
                int(bad), float(ratio.min()), float(ratio.max()),
            )
    if out_of_range:
        return CheckResult(
            "sanity_ranges", False,
            f"out-of-range ratios: {out_of_range} "
            f"(allowed [{PRICE_RATIO_MIN}, {PRICE_RATIO_MAX}])",
        )
    return CheckResult(
        "sanity_ranges", True,
        f"price ratios within [{PRICE_RATIO_MIN}, {PRICE_RATIO_MAX}] of source*FX",
    )


def check_file_set(track: Track) -> CheckResult:
    """All 10 expected files exist and are non-empty."""
    missing, empty = [], []
    for label, path in track.all_paths().items():
        if not path.exists():
            missing.append(label)
        elif path.stat().st_size == 0:
            empty.append(label)
    if missing or empty:
        return CheckResult(
            "file_set", False,
            f"missing={missing} empty={empty}",
        )
    return CheckResult("file_set", True, "all 10 files present and non-empty")


# Strings that are legitimately language-independent (numeric / unit /
# code patterns) — counting them as "untranslated" would be a false
# flag. Reuses the comprehensive list from the QA module so we treat
# specs like "kg", "kVA", "DN-200", "m³/h" identically.
import re as _re
_PASSTHROUGH_NUMERIC_RE = _re.compile(r"^[\d\s.,/x*×²³%·\-:_+()]+$")
_PASSTHROUGH_UNIT_TOKENS = (
    "kVA", "kW", "MW", "GW", "Wh", "kWh", "MWh",
    "V", "A", "mA", "Hz", "kHz", "MHz",
    "kg", "g", "mg", "t", "kt", "Mt",
    "m", "cm", "mm", "km", "nm", "pm",
    "m²", "m³", "cm²", "cm³", "mm²", "mm³",
    "m2", "m3", "cm2", "cm3", "mm2", "mm3",
    "m3/h", "m2/h", "m3/s",
    "l", "L", "ml", "cl", "dl", "hl",
    "h", "min", "s", "ms",
    "kN", "MN", "Pa", "kPa", "MPa", "GPa", "bar",
    "°C", "°F", "K", "%", "ppm", "Bq", "Sv",
    "rpm", "Nm", "Wb", "T",
    "DN", "PN", "PE", "PVC", "ABS", "PP", "HDPE", "PEX",
    "kgf", "tonf",
)
_PASSTHROUGH_UNIT_BODY = "|".join(
    _re.escape(u) for u in sorted(_PASSTHROUGH_UNIT_TOKENS, key=len, reverse=True)
)
_PASSTHROUGH_TECH_RE = _re.compile(
    rf"^\s*\d+(?:[.,]\d+)?"  # leading number with optional decimal
    rf"(?:\s*[\-–]\s*\d+(?:[.,]\d+)?)?"  # optional range (e.g. 2,5-2,9)
    rf"(?:\s*[xх×*]\s*\d+(?:[.,]\d+)?)*"  # optional dim multipliers
    rf"\s*(?:{_PASSTHROUGH_UNIT_BODY})"  # required unit
    rf"(?:\s*[/×x]\s*(?:{_PASSTHROUGH_UNIT_BODY}))*\s*$",  # optional /unit chain
    _re.UNICODE,
)
# Number + cognate noun (e.g. "100 Komplett", "10 Knoten").
_PASSTHROUGH_NUMBERED_LOANWORD_RE = None  # filled after _PASSTHROUGH_LOANWORDS defined
_PASSTHROUGH_PURE_UNIT_RE = _re.compile(
    rf"^\s*(?:{_PASSTHROUGH_UNIT_BODY})\s*$", _re.UNICODE,
)
_PASSTHROUGH_CODE_RE = _re.compile(
    r"^[A-Z0-9][A-Z0-9\-/×x*.,\s]*$", _re.UNICODE,
)
# Pipe-size / nominal-diameter / pressure-class specs (DN-200, Du 1000 mm, PN16)
_PASSTHROUGH_PIPE_SPEC_RE = _re.compile(
    r"^\s*(?:DN|Du|PN|PE|HD|LD)\s*\-?\s*\d+(?:\s*[xх×*/]\s*\d+)?(?:\s*mm)?\s*$",
    _re.UNICODE | _re.IGNORECASE,
)
# Voltage / power specs ("220 kV", "100 kVA", "6000 t/day")
_PASSTHROUGH_RATING_RE = _re.compile(
    r"^\s*\d+(?:[.,]\d+)?(?:\s*[xх×*/\-]\s*\d+(?:[.,]\d+)?)?\s*"
    r"(?:kV|MV|kVA|MVA|kW|MW|HP|PS|t/day|t/h|m3/day|km/h|rpm)\s*$",
    _re.UNICODE | _re.IGNORECASE,
)
# Common engineering loanwords / cognates (same word in source and target).
# Includes EN/ID/VI loanwords AND DE-SV/DE-NL Germanic cognates.
_PASSTHROUGH_LOANWORDS = frozenset({
    # English engineering vocabulary
    "set", "unit", "ha", "starter", "tripod", "streif", "channel.km",
    "lap", "lapis", "m farm", "100 ecm", "half-set", "100 knots",
    "camera", "cabin", "press", "stand", "horizontal", "vertical",
    "manipulator", "brander", "patchuk",
    # Germanic cognates (DE→SV/NL/DA — valid in all)
    "komplett", "foto", "installation", "system", "punkt", "nummer",
    "sektion", "ring", "kanal", "stativ", "knoten", "rez", "etui",
    "gips", "sand", "mullit",
})

# "100 Komplett", "10 Knoten" — numeric prefix + cognate.
_PASSTHROUGH_NUMBERED_LOANWORD_RE = _re.compile(
    rf"^\s*\d+\s+(?:{'|'.join(_re.escape(w) for w in sorted(_PASSTHROUGH_LOANWORDS, key=len, reverse=True))})\s*$",
    _re.UNICODE | _re.IGNORECASE,
)


def _is_passthrough_value(s: str) -> bool:
    """True if the string is a code/unit/numeric pattern that doesn't
    need translation (and should not count as untranslated)."""
    if not s:
        return True
    if _PASSTHROUGH_NUMERIC_RE.match(s):
        return True
    if _PASSTHROUGH_PURE_UNIT_RE.match(s):
        return True
    if _PASSTHROUGH_TECH_RE.match(s):
        return True
    if _PASSTHROUGH_PIPE_SPEC_RE.match(s):
        return True
    if _PASSTHROUGH_RATING_RE.match(s):
        return True
    if s.strip().lower() in _PASSTHROUGH_LOANWORDS:
        return True
    if _PASSTHROUGH_NUMBERED_LOANWORD_RE.match(s):
        return True
    if len(s) <= 30 and _PASSTHROUGH_CODE_RE.match(s):
        return True
    return False


def check_translation_completeness(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    target_lang: str,
    source_lang: str,
) -> CheckResult:
    """For each TEXT_COLS column, count rows where target value still
    equals source value verbatim (not translated). Numeric/unit-only
    strings are allowed to pass through. Threshold: <0.5% untranslated.
    """
    if target_lang == source_lang:
        return CheckResult(
            "translation_completeness", True,
            f"skipped (same language: {target_lang})",
        )

    try:
        from text_pipeline import TEXT_COLS
    except ImportError:
        return CheckResult(
            "translation_completeness", False,
            "could not import TEXT_COLS from text_pipeline",
        )

    sort_keys = ["rate_code", "resource_code"]
    t = target_df.sort_values(sort_keys).reset_index(drop=True)
    s = source_df.sort_values(sort_keys).reset_index(drop=True)

    total_text = 0
    untranslated = 0
    per_col: dict[str, tuple[int, int]] = {}
    for col in TEXT_COLS:
        if col not in t.columns or col not in s.columns:
            continue
        ts = t[col].astype("string").fillna("")
        ss = s[col].astype("string").fillna("")
        # Both non-empty and identical => same string survived translation.
        same = (ts == ss) & ts.str.len().gt(0)
        col_total = int(ts.str.len().gt(0).sum())
        if not same.any():
            total_text += col_total
            continue
        # Score each "same" row: only count as untranslated if value is
        # not a code/unit/spec passthrough. Run the predicate on unique
        # values to avoid 900K Python-level calls per column.
        same_values = ts[same].unique()
        flag_set = {v for v in same_values if not _is_passthrough_value(v)}
        col_untrans = int(ts[same].isin(flag_set).sum())
        per_col[col] = (col_untrans, col_total)
        untranslated += col_untrans
        total_text += col_total

    if total_text == 0:
        return CheckResult(
            "translation_completeness", False,
            "no text values found in any TEXT_COLS column",
        )

    pct = untranslated / total_text * 100
    worst = sorted(per_col.items(), key=lambda kv: -kv[1][0])[:5]
    worst_str = ", ".join(
        f"{c}={u}/{n}" for c, (u, n) in worst if u > 0
    ) or "none"

    if pct >= 0.5:
        return CheckResult(
            "translation_completeness", False,
            f"{untranslated:,}/{total_text:,} ({pct:.2f}%) untranslated; "
            f"worst: {worst_str}",
        )
    return CheckResult(
        "translation_completeness", True,
        f"{untranslated:,}/{total_text:,} ({pct:.3f}%) untranslated "
        f"(threshold 0.5%)",
    )


def check_embedding_count(track: Track, target_df: pd.DataFrame) -> CheckResult:
    """Qdrant collection point count equals dataset row count."""
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        return CheckResult(
            "embedding_count", False,
            "qdrant-client not installed",
        )

    import os
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")
    client = QdrantClient(url=url, api_key=api_key)
    collection = track.qdrant_collection
    try:
        info = client.get_collection(collection)
        n = info.points_count
    except Exception as e:
        return CheckResult(
            "embedding_count", False,
            f"could not query collection {collection!r}: {e}",
        )
    if n != len(target_df):
        return CheckResult(
            "embedding_count", False,
            f"qdrant points={n} != dataset rows={len(target_df)}",
        )
    return CheckResult(
        "embedding_count", True,
        f"qdrant {collection!r}: {n:,} points == {len(target_df):,} rows",
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all(
    track_name: str,
    fx_rate: float | None = None,
    skip: tuple[str, ...] = (),
) -> list[CheckResult]:
    track = get_track(track_name)
    if track.status != "target":
        raise ValueError(
            f"validators.py is for target tracks. "
            f"{track_name} is {track.status}."
        )
    if track.source_track is None:
        raise ValueError(f"target track {track_name} has no source_track set")

    source = get_track(track.source_track)
    print(f"\nValidating target {track_name} against source {source.region}\n")

    if not track.parquet_path.exists():
        return [CheckResult(
            "load", False,
            f"target parquet missing: {track.parquet_path}",
        )]

    target_df = pd.read_parquet(track.parquet_path)
    source_df = pd.read_parquet(source.parquet_path)
    ref_df = pd.read_parquet(EXISTING_TRACKS[SCHEMA_REFERENCE].parquet_path)

    results: list[CheckResult] = []

    def maybe(name: str, fn):
        if name in skip:
            results.append(CheckResult(name, True, "skipped"))
            return
        results.append(fn())

    maybe("schema_parity", lambda: check_schema_parity(target_df, ref_df, source_df))
    maybe("code_stability", lambda: check_code_stability(target_df, source_df))
    maybe("norms_immutability", lambda: check_norms_immutability(target_df, source_df))
    maybe("no_nan_in_prices", lambda: check_no_nan_in_prices(target_df, source_df))
    maybe("currency_consistency",
          lambda: check_currency_consistency(target_df, track.currency))

    if fx_rate is not None and "sanity_ranges" not in skip:
        results.append(check_sanity_ranges(target_df, source_df, fx_rate))
    elif fx_rate is None:
        results.append(CheckResult(
            "sanity_ranges", True,
            "skipped (no --fx-rate provided)",
        ))

    maybe("file_set", lambda: check_file_set(track))
    maybe(
        "translation_completeness",
        lambda: check_translation_completeness(
            target_df, source_df,
            target_lang=track.language,
            source_lang=source.language,
        ),
    )
    maybe("embedding_count", lambda: check_embedding_count(track, target_df))

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a CWICR target track.")
    parser.add_argument("--track", required=True,
                        help="target track region, e.g. AU_SYDNEY")
    parser.add_argument("--fx-rate", type=float, default=None,
                        help="target_ccy per source_ccy (enables sanity_ranges)")
    parser.add_argument("--skip", nargs="*", default=(),
                        help="check names to skip (e.g. embedding_count)")
    args = parser.parse_args()

    results = run_all(args.track, fx_rate=args.fx_rate, skip=tuple(args.skip))
    for r in results:
        print(r)
    failed = [r for r in results if not r.passed]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

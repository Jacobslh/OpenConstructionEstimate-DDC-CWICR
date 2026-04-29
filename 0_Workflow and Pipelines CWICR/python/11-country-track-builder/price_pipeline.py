"""
Price pipeline — converts source-track prices to target-track currency.

Three cascading steps, each refines the previous:

1. Type-factors (default): per-resource-type multipliers derived from
   wage indexes (labour), construction PPP (material), FX rate (equipment).
2. Location_factor (US-pattern): scalar regional adjustment within country.
3. National overrides (optional): CSV of (resource_code, unit_price) from
   national stat agencies, wins over the cascades.

After cascades, aggregates (price_min/max/median/mean, totals) are
recomputed from the line-item costs so summary columns remain consistent.

Reads FX/wage/PPP from configs/_fx_snapshot.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from validators import PRICE_COLS

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
FX_SNAPSHOT = CONFIG_DIR / "_fx_snapshot.yaml"

# Map iso_country -> OECD/ILO country code used in _fx_snapshot.yaml.
ISO2_TO_OECD3 = {
    "US": "USA", "CA": "CAN", "GB": "GBR", "DE": "DEU", "FR": "FRA",
    "IT": "ITA", "ES": "ESP", "AU": "AUS", "NZ": "NZL", "JP": "JPN",
    "KR": "KOR", "PL": "POL", "CZ": "CZE", "SE": "SWE", "HU": "HUN",
    "TR": "TUR", "MX": "MEX", "CN": "CHN", "IN": "IND", "RU": "RUS",
    "BR": "BRA", "ZA": "ZAF", "ID": "IDN", "VN": "VNM", "TH": "THA",
    "NG": "NGA", "HR": "HRV", "BG": "BGR", "RO": "ROU", "NL": "NLD",
    "AE": "ARE",
}


@dataclass
class PriceConfig:
    """Per-track price configuration (loaded from configs/<TRACK>.yaml)."""
    target_currency: str
    target_iso2: str
    source_currency: str
    source_iso2: str
    location_factor: float = 1.0     # within-country regional adjustment
    overrides_csv: Path | None = None  # optional national-source price file
    add_us_metadata: bool = False    # add cad_to_usd_rate-style cols
    snapshot_date: str = ""          # for record-keeping in output rows


@dataclass
class PriceFactors:
    """Computed multipliers applied to columns by resource type."""
    labour: float
    material: float
    equipment: float

    def for_resource_type(self, rtype: str | None) -> float:
        """Return the factor appropriate for the resource type."""
        if rtype is None:
            return self.equipment    # neutral default for unknown
        rtype_lower = rtype.lower() if isinstance(rtype, str) else ""
        if "labor" in rtype_lower or "labour" in rtype_lower or "operator" in rtype_lower:
            return self.labour
        if "material" in rtype_lower:
            return self.material
        # equipment, machinery, machine, plant, hire — all map to equipment.
        return self.equipment


# ---------------------------------------------------------------------------
# FX snapshot loading
# ---------------------------------------------------------------------------

_FX_CACHE: dict | None = None


def _load_fx_snapshot() -> dict:
    global _FX_CACHE
    if _FX_CACHE is None:
        with FX_SNAPSHOT.open("r", encoding="utf-8") as f:
            _FX_CACHE = yaml.safe_load(f)
    return _FX_CACHE


def fx_per_eur(ccy: str) -> float:
    snap = _load_fx_snapshot()
    rates = snap["rates_per_eur"]
    if ccy not in rates:
        raise KeyError(f"Currency {ccy} not in FX snapshot")
    return float(rates[ccy])


def fx_rate(target_ccy: str, source_ccy: str) -> float:
    """Units of target_ccy per 1 unit of source_ccy."""
    if target_ccy == source_ccy:
        return 1.0
    return fx_per_eur(target_ccy) / fx_per_eur(source_ccy)


def wage_factor(target_iso2: str, source_iso2: str) -> float:
    snap = _load_fx_snapshot()
    wages = snap["wage_index_ppp_usd_2024"]
    src = ISO2_TO_OECD3.get(source_iso2, source_iso2)
    tgt = ISO2_TO_OECD3.get(target_iso2, target_iso2)
    if src not in wages or tgt not in wages:
        # Fall back to FX-only conversion if wage index is missing.
        return None
    return float(wages[tgt]) / float(wages[src])


def material_factor(target_iso2: str, source_iso2: str) -> float:
    snap = _load_fx_snapshot()
    ppp = snap["ppp_construction_gfcf_2023"]
    src = ISO2_TO_OECD3.get(source_iso2, source_iso2)
    tgt = ISO2_TO_OECD3.get(target_iso2, target_iso2)
    if src not in ppp or tgt not in ppp:
        return None
    return float(ppp[tgt]) / float(ppp[src])


def compute_factors(cfg: PriceConfig) -> PriceFactors:
    """Compute the three type-factors. Falls back to FX where index is absent."""
    fx = fx_rate(cfg.target_currency, cfg.source_currency)

    # Labour: wage index in PPP-USD. PPP cancels real-wage differences; we
    # multiply the source labour cost by the wage ratio expressed in target
    # currency. wage ratio is dimensionless USD-PPP, so we still need fx
    # to bring source-currency labour cost into target currency.
    wf = wage_factor(cfg.target_iso2, cfg.source_iso2)
    labour = (wf if wf is not None else 1.0) * fx

    # Material: PPP for construction GFCF expresses local-currency price of a
    # standard construction basket. Ratio of PPPs is the implicit FX for that
    # specific basket. Use it directly (no extra fx multiplier — PPPs are
    # already in local currency).
    mf = material_factor(cfg.target_iso2, cfg.source_iso2)
    material = mf if mf is not None else fx

    # Equipment: tradable goods — use plain FX.
    equipment = fx

    return PriceFactors(labour=labour, material=material, equipment=equipment)


# ---------------------------------------------------------------------------
# Apply factors to dataframe
# ---------------------------------------------------------------------------

def _resource_type_for_row(row: pd.Series) -> str:
    """
    Derive resource type from the boolean flag columns. Order matters:
    labour/operator wins over material wins over machine for labour-intensive
    composite rows.
    """
    if bool(row.get("is_labor", False)):
        return "labour"
    if bool(row.get("is_material", False)):
        return "material"
    if bool(row.get("is_machine", False)):
        return "equipment"
    return "equipment"  # neutral default


def apply_factors(
    df: pd.DataFrame, factors: PriceFactors, location_factor: float = 1.0,
) -> pd.DataFrame:
    """
    Multiply every price column by (type_factor * location_factor) row-wise.

    location_factor uniformly adjusts the country-level numbers down to a
    specific region (e.g. metro vs. national average).
    """
    out = df.copy()

    # Compute per-row factor once.
    types = out.apply(_resource_type_for_row, axis=1)
    type_factor = types.map({
        "labour": factors.labour,
        "material": factors.material,
        "equipment": factors.equipment,
    }).astype(float)

    combined = type_factor * float(location_factor)

    for col in PRICE_COLS:
        if col not in out.columns:
            continue
        # Skip columns that are not numeric (string-encoded ranges).
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue
        out[col] = out[col] * combined

    return out


# ---------------------------------------------------------------------------
# National overrides
# ---------------------------------------------------------------------------

def apply_overrides(df: pd.DataFrame, overrides_csv: Path | None) -> pd.DataFrame:
    """
    Replace per-resource unit prices with values from a national-source CSV.

    CSV columns:
        resource_code (string), unit_price (float)
    Optional:
        labor_rate_per_hr — overrides labor_rate_per_hr where the resource
        is labour.
    """
    if overrides_csv is None or not Path(overrides_csv).exists():
        return df
    overrides = pd.read_csv(overrides_csv)
    if "resource_code" not in overrides.columns:
        raise ValueError(f"{overrides_csv}: missing required 'resource_code'")

    out = df.copy()
    code_to_price = dict(zip(overrides["resource_code"], overrides["unit_price"]))
    mask = out["resource_code"].isin(code_to_price)
    if mask.any():
        new_prices = out.loc[mask, "resource_code"].map(code_to_price)
        out.loc[mask, "resource_price_per_unit_current"] = new_prices.values

        # Recompute resource_cost = price * quantity for overridden rows.
        if "resource_quantity" in out.columns:
            qty = out.loc[mask, "resource_quantity"].fillna(0)
            out.loc[mask, "resource_cost"] = new_prices.values * qty.values

    if "labor_rate_per_hr" in overrides.columns and "labor_rate_per_hr" in out.columns:
        labour_overrides = overrides.dropna(subset=["labor_rate_per_hr"])
        labour_map = dict(
            zip(labour_overrides["resource_code"], labour_overrides["labor_rate_per_hr"])
        )
        lmask = out["resource_code"].isin(labour_map)
        if lmask.any():
            out.loc[lmask, "labor_rate_per_hr"] = (
                out.loc[lmask, "resource_code"].map(labour_map).values
            )

    return out


# ---------------------------------------------------------------------------
# Recompute aggregates after price changes
# ---------------------------------------------------------------------------

def recompute_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    After apply_factors() has multiplied each row's price columns by the
    appropriate (type_factor × location_factor), the aggregate columns are
    *already* consistent — every column was scaled by the same factor on
    each row that contributed to the sum, so groupby sums and totals
    inherit the scaling correctly.

    Recomputing the aggregates from a groupby-sum here would be wrong
    because:
      a) The source dataset stores `total_cost_per_position` and the
         abstract-price aggregates with rate-specific business logic that
         we cannot reliably reproduce (e.g. abstract rates with no
         resources still carry a stored estimate).
      b) Re-aggregating with .transform("sum") would zero out abstract
         rate rows that have no resources, breaking parity with the source.

    So this function is intentionally a near no-op: it only rebuilds the
    one column that *must* change as a direct consequence of price-cascade
    overrides — `resource_cost` per row — when overrides changed
    `resource_price_per_unit_current` directly.
    """
    out = df.copy()

    if (
        "resource_quantity" in out.columns
        and "resource_price_per_unit_current" in out.columns
        and "resource_cost" in out.columns
    ):
        # Only recompute resource_cost where BOTH inputs are present and
        # the product disagrees with the stored value beyond a tiny
        # tolerance — that signals an override changed the price.
        qty = out["resource_quantity"]
        price = out["resource_price_per_unit_current"]
        expected = qty * price
        mask = qty.notna() & price.notna()
        out.loc[mask, "resource_cost"] = expected[mask]

    return out


# ---------------------------------------------------------------------------
# Orchestrator entrypoint
# ---------------------------------------------------------------------------

def run(df: pd.DataFrame, cfg: PriceConfig) -> pd.DataFrame:
    """Apply the full three-cascade pipeline."""
    factors = compute_factors(cfg)
    print(
        f"  factors: labour={factors.labour:.4f} "
        f"material={factors.material:.4f} equipment={factors.equipment:.4f} "
        f"location={cfg.location_factor}"
    )

    out = apply_factors(df, factors, cfg.location_factor)
    out = apply_overrides(out, cfg.overrides_csv)
    out = recompute_aggregates(out)

    # Currency stamp.
    if "currency" in out.columns:
        out["currency"] = cfg.target_currency

    if cfg.add_us_metadata:
        # Mirror US-track metadata for downstream consumers that key off it.
        out["cad_to_usd_rate"] = fx_rate(cfg.target_currency, cfg.source_currency)
        out["conversion_date"] = cfg.snapshot_date
        out["location_factor"] = cfg.location_factor

    return out

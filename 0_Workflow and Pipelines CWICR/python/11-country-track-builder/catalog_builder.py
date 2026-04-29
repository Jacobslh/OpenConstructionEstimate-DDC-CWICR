"""
Catalog builder — produces the per-resource summary used in
`DDC_CWICR_<REGION>_Catalog.csv` and `.xlsx`.

One row per resource (resource_code), with price aggregates and parent
taxonomy fields. Mirrors the schema in DATA_DICTIONARY.md.
"""

from __future__ import annotations

import pandas as pd


CATALOG_COLUMNS = [
    "resource_code",
    "name",
    "type",
    "category",
    "unit",
    "price_avg",
    "price_min",
    "price_max",
    "price_median",
    "price_variants",
    "currency",
    "avg_cost_per_use",
    "avg_qty_per_use",
    "usage_count",
    "used_in_work_items",
    "parent_category",
    "parent_collection",
    "parent_department",
    "parent_section",
]


def _resource_type(group: pd.DataFrame) -> str:
    """Pick the most descriptive type label for a resource group."""
    if group.get("is_labor", pd.Series([False])).any():
        return "Labour"
    if group.get("is_material", pd.Series([False])).any():
        return "Material"
    if group.get("is_machine", pd.Series([False])).any():
        return "Equipment"
    return "Other"


def build(df: pd.DataFrame, currency: str) -> pd.DataFrame:
    """
    Aggregate the work-items dataframe into a one-row-per-resource catalog.
    """
    if "resource_code" not in df.columns:
        raise ValueError("workitems df is missing 'resource_code' column")

    # Drop abstract / synthetic rows that have no resource attached.
    have_resource = df.dropna(subset=["resource_code"]).copy()
    if have_resource.empty:
        return pd.DataFrame(columns=CATALOG_COLUMNS)

    rows = []
    for code, grp in have_resource.groupby("resource_code", sort=True):
        prices = grp["resource_price_per_unit_current"].dropna()
        costs = grp["resource_cost"].dropna()
        qtys = grp["resource_quantity"].dropna()

        rows.append({
            "resource_code": code,
            "name": grp["resource_name"].dropna().iloc[0] if grp["resource_name"].notna().any() else "",
            "type": _resource_type(grp),
            "category": grp.get("collection_name", pd.Series(dtype=str)).dropna().iloc[0]
            if "collection_name" in grp and grp["collection_name"].notna().any() else "",
            "unit": grp["resource_unit"].dropna().iloc[0] if grp["resource_unit"].notna().any() else "",
            "price_avg": float(prices.mean()) if len(prices) else None,
            "price_min": float(prices.min()) if len(prices) else None,
            "price_max": float(prices.max()) if len(prices) else None,
            "price_median": float(prices.median()) if len(prices) else None,
            "price_variants": int(len(prices)),
            "currency": currency,
            "avg_cost_per_use": float(costs.mean()) if len(costs) else None,
            "avg_qty_per_use": float(qtys.mean()) if len(qtys) else None,
            "usage_count": int(len(grp)),
            "used_in_work_items": int(grp["rate_code"].nunique()) if "rate_code" in grp else 0,
            "parent_category": grp.get("collection_name", pd.Series(dtype=str)).dropna().iloc[0]
            if "collection_name" in grp and grp["collection_name"].notna().any() else "",
            "parent_collection": grp.get("collection_code", pd.Series(dtype=str)).dropna().iloc[0]
            if "collection_code" in grp and grp["collection_code"].notna().any() else "",
            "parent_department": grp.get("department_name", pd.Series(dtype=str)).dropna().iloc[0]
            if "department_name" in grp and grp["department_name"].notna().any() else "",
            "parent_section": grp.get("section_name", pd.Series(dtype=str)).dropna().iloc[0]
            if "section_name" in grp and grp["section_name"].notna().any() else "",
        })

    return pd.DataFrame(rows, columns=CATALOG_COLUMNS)

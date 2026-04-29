# Country Track Builder

Generate a complete CWICR country track (Catalog + parquet + xlsx +
README + Qdrant snapshot) from a YAML config and a source track.

## Quick start

```bash
# Build Australia/Sydney from UK_GBP, no embeddings.
python add_country_track.py --config configs/AU_SYDNEY.yaml --skip-embeddings

# Build Croatia/Zagreb (translation pass: de -> hr).
OPENAI_API_KEY=... python add_country_track.py --config configs/HR_ZAGREB.yaml

# Build all 19 target tracks in serial.
python add_country_track.py --all --skip-embeddings
```

Pass `--full-xlsx` to generate real (~150 MB) xlsx files; default writes
LFS-pointer placeholders (134 bytes) matching the existing repo layout.

## What changes per track

The Resource-Based Costing methodology splits the 93 columns into three
buckets:

- **Norms (~15 cols)** — `resource_quantity`, `labor_hours_*`,
  `count_*_per_unit`, `electricity_consumption_kwh_per_machine_hour`,
  `mass_value`. **Never modified.** Validated bytewise vs source.
- **Prices (~20 cols)** — every `*_cost`, `*_rate`, `price_*`, plus
  `currency`. Recomputed via the price pipeline.
- **Language (~10 cols)** — `rate_original_name`, `work_composition_text`,
  `department_name`, `resource_name`, etc. Translated once per language
  pair and cached in `glossary/translations/<src>_<tgt>.json`.

Stable identifiers `rate_code` and `resource_code` carry across every
track — they are the alignment keys for cross-track comparison.

## Modules

| File | Purpose |
|---|---|
| `tracks.py` | Registry of all 30 tracks (existing + target). Single source of truth. |
| `validators.py` | 8 schema-level acceptance gates. |
| `validators_suite.py` | 30 canonical construction scenarios for functional regression. |
| `run_functional_suite.py` | Runs the 30-work suite; combines public benchmarks + cross-track sanity + LLM-judge + manual review. |
| `price_pipeline.py` | Three-cascade pricing: type-factors → location_factor → national overrides. |
| `text_pipeline.py` | SHA1-deduped translation via Claude/OpenAI/Google fallback. |
| `catalog_builder.py` | Builds the per-resource Catalog CSV/XLSX. |
| `writers.py` | Produces all 10 track-local files. LFS-pointer mode by default. |
| `format_template.py` | (Reserved.) Style extraction from existing FORMATTED.xlsx. |
| `add_country_track.py` | Orchestrator. Calls the pipelines in order, then validates. |
| `build_glossary_seed.py` | One-time builder for `glossary/construction_glossary.json` from existing tracks. |

## Configs

`configs/<TRACK>.yaml` declares `region`, `location_factor` (default 1.0),
optional `overrides_csv`, `add_us_metadata` flag, `translate` flag.

`configs/_fx_snapshot.yaml` holds the FX rates (ECB live + pegged BGN/AED
+ manual RUB/VND/NGN), OECD wage indexes, and OECD PPP for construction
GFCF. Refresh with `scripts/refresh_fx_snapshot.py` (TBD) and update
`snapshot_date`. The snapshot date is recorded in every track's README.

## Adding a new country

1. Add a `Track` entry to `TARGET_TRACKS` in `tracks.py`.
2. Pick a source track. Rule: same currency → same language →
   `UK_GBP` as universal English fallback. Never use `EN_TORONTO`
   (legacy schema) or `USA_USD` (anomalous extras) as source.
3. Create `configs/<TRACK>.yaml`. Set `location_factor` to the
   capital city premium per the national stats agency. Set
   `translate: true` if `target_language` differs from source.
4. (Optional) Add a national-source `price_overrides` CSV.
5. Run `python add_country_track.py --config configs/<TRACK>.yaml`.
6. Inspect `validation_reports/<TRACK>_estimates.html` after the
   functional suite runs.

## Validation gates

Schema gate (8 checks, run on every PR):

1. `schema_parity` — column names match canonical reference (DE_BERLIN);
   dtypes match the source track (UK NRM2 alternate accepted).
2. `code_stability` — `rate_code` and `resource_code` sets identical to source.
3. `norms_immutability` — all 15 norm columns bytewise-equal to source.
4. `no_nan_in_prices` — no NaN regressions in 20 price columns.
5. `currency_consistency` — every row carries the target currency.
6. `sanity_ranges` — per-row price ratio target/(source × FX) ∈ [0.01, 100].
7. `file_set` — all 10 expected files present and non-empty.
8. `embedding_count` — Qdrant collection point count == row count.

Functional gate (30 scenarios × 4 methods, manual trigger):

a. Public benchmarks (RSMeans / Rawlinsons / Eurostat / NSI / DZS).
b. Cross-track sanity (±50% to neighbour tracks).
c. LLM-judge (`gpt-4o-mini` reviews "as a country-X estimator").
d. Manual expert review of generated `_estimates.html`.

Promotion threshold: 8/8 schema gates + ≥27/30 functional + manual sign-off.

## Costs (rough)

- LLM translation per language: ~$2-3 (gpt-4o-mini, ~80K unique strings).
- LLM-judge per track: ~$1-2 (gpt-4o-mini, 30 calls).
- Embeddings per track: ~$8 (`text-embedding-3-large`, 900K rows × ~150 tokens).

Cumulative for all 19 tracks: roughly $50-80 in LLM/embedding spend.

"""
Orchestrator — build a complete CWICR track for one target country.

Reads YAML config, loads source parquet, runs the price + (optional)
text pipelines, generates all output files, and validates.

Usage:
    python add_country_track.py --config configs/AU_SYDNEY.yaml
    python add_country_track.py --config configs/AU_SYDNEY.yaml --skip-embeddings
    python add_country_track.py --all                 # build every target track
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

import env_loader  # noqa: F401  — auto-loads .env into os.environ
import catalog_builder
import price_pipeline
import validators
import writers
from tracks import TARGET_TRACKS, get_track


HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "configs"
EMBEDDING_PIPELINE = (
    HERE.parent / "10-embedding-pipeline" / "generate_embeddings.py"
)


@dataclass
class TrackConfig:
    region: str
    target_iso2: str
    target_currency: str
    target_language: str
    source_track: str
    location_factor: float = 1.0
    overrides_csv: str | None = None
    add_us_metadata: bool = False
    translate: bool = False
    raw: dict = None  # underlying yaml for forward-compat


def load_config(path: Path) -> TrackConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    region = raw["region"]
    track = TARGET_TRACKS.get(region)
    if track is None:
        raise ValueError(f"Unknown target region {region!r} in {path}")
    return TrackConfig(
        region=region,
        target_iso2=track.iso_country,
        target_currency=track.currency,
        target_language=track.language,
        source_track=track.source_track,
        location_factor=float(raw.get("location_factor", 1.0)),
        overrides_csv=raw.get("overrides_csv"),
        add_us_metadata=bool(raw.get("add_us_metadata", False)),
        translate=bool(raw.get("translate", True)),
        raw=raw,
    )


def build_one(cfg: TrackConfig, skip_embeddings: bool = False,
              full_xlsx: bool = False) -> int:
    """Returns 0 on success, non-zero on validation failure."""
    track = get_track(cfg.region)
    source = get_track(cfg.source_track)

    print(f"\n{'='*70}")
    print(f"Build target {cfg.region}  <-  source {source.region}")
    print(f"  language: {source.language!r} -> {cfg.target_language!r}")
    print(f"  currency: {source.currency!r} -> {cfg.target_currency!r}")
    print(f"{'='*70}")

    t0 = time.time()
    print(f"\n[1/6] Load source parquet ({source.parquet_path.name}) ...")
    df = pd.read_parquet(source.parquet_path)
    print(f"      {len(df):,} rows × {df.shape[1]} cols")

    # Price pipeline.
    print(f"\n[2/6] Apply price pipeline ...")
    pcfg = price_pipeline.PriceConfig(
        target_currency=cfg.target_currency,
        target_iso2=cfg.target_iso2,
        source_currency=source.currency,
        source_iso2=source.iso_country,
        location_factor=cfg.location_factor,
        overrides_csv=Path(cfg.overrides_csv) if cfg.overrides_csv else None,
        add_us_metadata=cfg.add_us_metadata,
        snapshot_date=str(price_pipeline._load_fx_snapshot()["snapshot_date"]),
    )
    df = price_pipeline.run(df, pcfg)

    # Translation pipeline (if applicable). Lazy-imported because the heavy
    # LLM deps shouldn't be required for English-target tracks.
    if cfg.translate and cfg.target_language != source.language:
        print(f"\n[3/6] Apply text pipeline ({source.language} -> {cfg.target_language}) ...")
        try:
            import text_pipeline
            df = text_pipeline.run(
                df, source_lang=source.language,
                target_lang=cfg.target_language,
            )
        except ImportError as e:
            print(f"      WARNING: text_pipeline not available ({e}); skipping translation")
    else:
        print(f"\n[3/6] Translation skipped (same language: {cfg.target_language})")

    # rate_unit_copy is a verbatim duplicate of rate_unit in every source.
    # The text pipeline can drift the two on retranslation (one batch
    # routes the value through cache hash A, the other through hash B
    # via different surrounding context). Reconcile here so the two
    # columns stay byte-equal across rebuilds.
    if "rate_unit_copy" in df.columns and "rate_unit" in df.columns:
        df["rate_unit_copy"] = df["rate_unit"]

    # Build catalog.
    print(f"\n[4/6] Build resource catalog ...")
    catalog_df = catalog_builder.build(df, currency=cfg.target_currency)
    print(f"      {len(catalog_df):,} unique resources")

    # Write all output files.
    print(f"\n[5/6] Write output files to {track.folder}/ ...")
    writers.write_all(
        df=df,
        catalog_df=catalog_df,
        track=track,
        source=source,
        n_resources=len(catalog_df),
        full_xlsx=full_xlsx,
    )

    # Embeddings.
    if skip_embeddings:
        print(f"\n[6/6] Embeddings: skipped (--skip-embeddings)")
    else:
        print(f"\n[6/6] Generate embeddings via 10-embedding-pipeline ...")
        if not EMBEDDING_PIPELINE.exists():
            print(f"      WARNING: embedding pipeline not at {EMBEDDING_PIPELINE}, skipping")
        else:
            cmd = [
                sys.executable, str(EMBEDDING_PIPELINE),
                "--input", str(track.parquet_path),
                "--collection", track.qdrant_collection,
            ]
            print(f"      $ {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"      WARNING: embedding pipeline failed: {e}")

    elapsed = time.time() - t0
    print(f"\nBuild complete in {elapsed:.1f}s\n")

    # Release the in-memory dataframes before validation. The validator
    # reloads source/target/reference parquets from disk, so holding the
    # build's copies in memory pushes peak RAM to ~12 GB per process and
    # OOMs when two builds run in parallel.
    import gc
    del df, catalog_df
    gc.collect()

    # Validation.
    print("=" * 70)
    print("Acceptance gate")
    print("=" * 70)
    fx = price_pipeline.fx_rate(cfg.target_currency, source.currency)
    skip = ("embedding_count",) if skip_embeddings else ()
    results = validators.run_all(cfg.region, fx_rate=fx, skip=skip)
    for r in results:
        print(r)
    failed = [r for r in results if not r.passed]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a CWICR target country track."
    )
    parser.add_argument("--config", type=str, default=None,
                        help="path to track YAML config (configs/<TRACK>.yaml)")
    parser.add_argument("--all", action="store_true",
                        help="build every target track in serial")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="don't call 10-embedding-pipeline")
    parser.add_argument("--full-xlsx", action="store_true",
                        help="generate real xlsx files (~30 min/track, ~150MB each); "
                             "default writes LFS pointer placeholders matching the "
                             "existing repo layout")
    args = parser.parse_args()

    if args.all:
        configs = sorted(CONFIG_DIR.glob("*.yaml"))
        configs = [c for c in configs if not c.name.startswith("_")]
        if not configs:
            print(f"No YAML configs in {CONFIG_DIR}")
            return 1
        rcs = []
        for cfg_path in configs:
            cfg = load_config(cfg_path)
            rc = build_one(cfg, skip_embeddings=args.skip_embeddings, full_xlsx=args.full_xlsx)
            rcs.append((cfg.region, rc))
        print("\n=== Summary ===")
        for region, rc in rcs:
            print(f"  {region}: {'OK' if rc == 0 else 'FAIL'}")
        return 0 if all(rc == 0 for _, rc in rcs) else 1

    if not args.config:
        print("Need --config <yaml> or --all")
        return 1
    cfg = load_config(Path(args.config))
    return build_one(cfg, skip_embeddings=args.skip_embeddings, full_xlsx=args.full_xlsx)


if __name__ == "__main__":
    sys.exit(main())

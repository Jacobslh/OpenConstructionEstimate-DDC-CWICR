"""
Batch builder — produce all (or a subset of) target tracks in serial.

Why serial, not parallel: each translate-needing track issues 8 concurrent
OpenAI calls already, and parallelising tracks on top of that risks
rate-limit thrashing on a single API key. Serial keeps the per-track time
reasonable (~5-10 min for English-target tracks, ~50-60 min for tracks
that translate) and the overall finish predictable.

Usage:
    python batch_build.py                       # all 19 target tracks
    python batch_build.py --only AU_SYDNEY HR_ZAGREB
    python batch_build.py --skip-translate      # only English-target tracks
    python batch_build.py --skip-embeddings     # don't call 10-embedding-pipeline
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

import add_country_track
from tracks import TARGET_TRACKS, get_track


HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "configs"


def collect_configs(
    only: list[str] | None,
    skip_translate: bool,
) -> list[Path]:
    if only:
        configs = []
        for region in only:
            p = CONFIG_DIR / f"{region}.yaml"
            if not p.exists():
                print(f"WARN: config {p.name} not found, skipping")
                continue
            configs.append(p)
        return configs

    paths = sorted(CONFIG_DIR.glob("*.yaml"))
    paths = [p for p in paths if not p.name.startswith("_")]

    if skip_translate:
        keep = []
        for p in paths:
            cfg_raw = yaml.safe_load(p.read_text(encoding="utf-8"))
            translate = bool(cfg_raw.get("translate", True))
            track = get_track(cfg_raw["region"])
            source = get_track(track.source_track)
            actually_translates = translate and source.language != track.language
            if not actually_translates:
                keep.append(p)
        return keep

    return paths


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", default=None,
                   help="region(s) to build, e.g. AU_SYDNEY HR_ZAGREB")
    p.add_argument("--skip-translate", action="store_true",
                   help="only build tracks where source language matches target")
    p.add_argument("--skip-embeddings", action="store_true",
                   help="don't call 10-embedding-pipeline")
    p.add_argument("--full-xlsx", action="store_true")
    args = p.parse_args()

    configs = collect_configs(args.only, args.skip_translate)
    if not configs:
        print("nothing to build")
        return 1

    print(f"\nBuilding {len(configs)} tracks in serial:")
    for c in configs:
        print(f"  - {c.stem}")

    results = []
    for cfg_path in configs:
        cfg = add_country_track.load_config(cfg_path)
        t0 = time.time()
        rc = add_country_track.build_one(
            cfg,
            skip_embeddings=args.skip_embeddings,
            full_xlsx=args.full_xlsx,
        )
        elapsed = time.time() - t0
        results.append((cfg.region, rc, elapsed))

    print("\n" + "=" * 70)
    print("Batch summary")
    print("=" * 70)
    n_pass = 0
    for region, rc, elapsed in results:
        status = "OK" if rc == 0 else "FAIL"
        if rc == 0:
            n_pass += 1
        print(f"  {region:<22s}  {status:<5s}  {elapsed/60:.1f} min")
    print(f"\n{n_pass}/{len(results)} tracks passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

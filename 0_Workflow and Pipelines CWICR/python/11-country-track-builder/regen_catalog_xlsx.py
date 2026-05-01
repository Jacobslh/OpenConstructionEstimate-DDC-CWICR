"""Regenerate Catalog.xlsx for 14 translate tracks from current parquets.

Wave 1 pushed fresh parquets with new TEXT_COLS translations. The
small per-track Catalog.xlsx (~800 KB, regular git not LFS) needs to
be refreshed to match. SIMPLE/FORMATTED xlsx remain stale LFS pointers
referencing old parquet hashes — those stay untouched here.
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from pathlib import Path

import tracks
import catalog_builder
import writers

TRACK_NAMES = [
    "IT_ROME", "NL_AMSTERDAM", "SV_STOCKHOLM", "HR_ZAGREB",
    "CS_PRAGUE", "PL_WARSAW", "RO_BUCHAREST", "TR_ISTANBUL", "BG_SOFIA",
    "ID_JAKARTA", "VI_HANOI", "JA_TOKYO", "KO_SEOUL", "TH_BANGKOK",
]


def regen_one(name: str) -> int:
    t0 = time.time()
    track = tracks.get_track(name)
    df = pd.read_parquet(track.parquet_path)
    cat_df = catalog_builder.build(df, track.currency)
    writers.write_catalog(
        cat_df, track.catalog_csv_path, track.catalog_xlsx_path,
        track.parquet_path, full_xlsx=True,
    )
    size = track.catalog_xlsx_path.stat().st_size
    print(f"  {name}: catalog xlsx {size:,} bytes ({time.time()-t0:.1f}s)",
          flush=True)
    return 0


def main() -> int:
    if len(sys.argv) >= 2:
        return regen_one(sys.argv[1])
    grand0 = time.time()
    for name in TRACK_NAMES:
        regen_one(name)
    print(f"\nDONE: 14 catalogs in {time.time()-grand0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

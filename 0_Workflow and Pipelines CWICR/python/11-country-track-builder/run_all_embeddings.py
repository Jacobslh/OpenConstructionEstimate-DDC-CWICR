"""
Generate Qdrant embeddings for every target track.

Drives ../10-embedding-pipeline/generate_embeddings.py once per track,
in serial. ~30 min/track × 19 tracks ≈ 9.5 h wall time. Cost
≈ $7-10/track at OpenAI text-embedding-3-large rates ≈ $130-180 total.

Usage:
    python run_all_embeddings.py                 # all 19 target tracks
    python run_all_embeddings.py --only AU_SYDNEY HR_ZAGREB
    python run_all_embeddings.py --skip-done     # skip tracks whose snapshot file already exists
    python run_all_embeddings.py --dry-run       # subprocess-level dry-run

Failure of one track does not stop the rest; a failure summary is
printed at the end.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from tracks import TARGET_TRACKS, get_track


HERE = Path(__file__).resolve().parent
EMBEDDING_PIPELINE = (
    HERE.parent / "10-embedding-pipeline" / "generate_embeddings.py"
)


def run_one(region: str, *, dry_run: bool, skip_done: bool) -> tuple[str, str, float]:
    track = get_track(region)
    parquet = track.parquet_path
    snapshot = track.embeddings_snapshot_path

    if not parquet.exists():
        return (region, "SKIP_NO_PARQUET", 0.0)
    if skip_done and snapshot.exists() and snapshot.stat().st_size > 1024:
        return (region, "SKIP_EXISTS", 0.0)

    cmd = [
        sys.executable, str(EMBEDDING_PIPELINE),
        "--input", str(parquet),
        "--collection", track.qdrant_collection,
    ]
    if dry_run:
        cmd.append("--dry-run")

    print("\n" + "=" * 70)
    print(f"Embedding {region} -> {track.qdrant_collection}")
    print(f"  $ {' '.join(cmd)}")
    print("=" * 70)

    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
        return (region, "OK", time.time() - t0)
    except subprocess.CalledProcessError as e:
        return (region, f"FAIL_{e.returncode}", time.time() - t0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--skip-done", action="store_true",
                   help="skip tracks whose snapshot file already exists")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    regions = args.only or list(TARGET_TRACKS)
    print(f"\nEmbedding {len(regions)} tracks in serial:")
    for r in regions:
        print(f"  - {r}")

    results: list[tuple[str, str, float]] = []
    for region in regions:
        results.append(
            run_one(region, dry_run=args.dry_run, skip_done=args.skip_done)
        )

    print("\n" + "=" * 70)
    print("Embedding summary")
    print("=" * 70)
    n_ok = 0
    for region, status, elapsed in results:
        flag = "OK" if status == "OK" else "FAIL"
        if status == "OK":
            n_ok += 1
        print(f"  {region:<22s}  {status:<16s}  {elapsed/60:.1f} min")
    print(f"\n{n_ok}/{len(results)} tracks embedded successfully")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

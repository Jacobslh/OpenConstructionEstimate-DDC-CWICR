"""
Rebuild all 14 translate-tracks with the expanded TEXT_COLS, no
LLM_TOP_N cap, and a cleaned BG cache. After each track, run the
6-layer QA module and apply purge-and-retry until script_block flag
rate is 0%.

Usage:
    python rebuild_quality.py                  # all 14 tracks, serial
    python rebuild_quality.py --pilot IT_ROME  # one track only
    python rebuild_quality.py --resume         # skip tracks whose QA already shows 0 script_block
    python rebuild_quality.py --max-retries 2  # default 2 purge-retry rounds per track

Wall time: ~30-50 min/track × 14 = 7-12 hours sequential. Memory peak
~12 GB during translation (single process). Cost: ~$5-10 OpenAI total.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import env_loader  # noqa: F401  — auto-loads .env

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "configs"
REPORTS_DIR = HERE / "validation_reports"

# Order: smallest blast-radius first. Latin-target tracks are quick
# wins; Cyrillic/CJK go later because they have more pending work.
SERIAL_ORDER = [
    "IT_ROME",        # pilot — Latin, big cache hit
    "NL_AMSTERDAM",
    "SV_STOCKHOLM",
    "HR_ZAGREB",
    "CS_PRAGUE",
    "PL_WARSAW",
    "RO_BUCHAREST",
    "TR_ISTANBUL",
    "BG_SOFIA",       # Cyrillic — cache pre-cleaned for RU leakage
    "ID_JAKARTA",
    "VI_HANOI",
    "JA_TOKYO",
    "KO_SEOUL",
    "TH_BANGKOK",
]

REGION_TO_CONFIG = {r: f"{r}.yaml" for r in [
    "IT_ROME", "NL_AMSTERDAM", "SV_STOCKHOLM", "HR_ZAGREB", "CS_PRAGUE",
    "PL_WARSAW", "RO_BUCHAREST", "TR_ISTANBUL", "BG_SOFIA",
    "ID_JAKARTA", "VI_HANOI", "JA_TOKYO", "KO_SEOUL", "TH_BANGKOK",
]}


def already_clean(region: str) -> bool:
    """True if QA report shows 0 script_block flagged."""
    p = REPORTS_DIR / f"qa_{region}.json"
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    sb = d.get("layer_stats", {}).get("script_block", {})
    return sb.get("flagged", -1) == 0


def run_translate(region: str) -> int:
    """Run add_country_track.py for one region. Returns exit code."""
    config = REGION_TO_CONFIG.get(region)
    if not config:
        print(f"  no config for {region}; skip")
        return 0
    cfg_path = CONFIG_DIR / config
    if not cfg_path.exists():
        print(f"  config {cfg_path.name} missing; skip")
        return 0
    cmd = [
        sys.executable, "-u",
        str(HERE / "add_country_track.py"),
        "--config", str(cfg_path),
        "--skip-embeddings",
    ]
    print(f"  $ {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=HERE).returncode
    return rc


def run_qa(region: str) -> dict:
    """Run translation_validators.py for one region (skip round-trip).
    Returns the parsed JSON report.
    """
    cmd = [
        sys.executable, "-u",
        "-m", "quality_check.translation_validators",
        "--track", region, "--skip-round-trip",
    ]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=HERE, check=False)
    p = REPORTS_DIR / f"qa_{region}.json"
    if not p.exists():
        return {"layer_stats": {}, "flagged_count": 0}
    return json.loads(p.read_text(encoding="utf-8"))


def run_qa_purge(region: str) -> int:
    """Run translation_validators with --purge to remove flagged hashes
    from the cache so the next translate pass re-translates them."""
    cmd = [
        sys.executable, "-u",
        "-m", "quality_check.translation_validators",
        "--track", region, "--skip-round-trip", "--purge",
    ]
    print(f"  $ {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=HERE).returncode
    return rc


def build_one(region: str, max_retries: int = 2) -> dict:
    """Translate -> QA -> purge -> translate again until clean or
    max_retries exhausted. Returns final QA report."""
    print(f"\n{'='*70}\nRebuilding {region}\n{'='*70}")
    t0 = time.time()
    rc = run_translate(region)
    print(f"  translate exit={rc}")
    qa = run_qa(region)
    sb = qa.get("layer_stats", {}).get("script_block", {})
    flagged = sb.get("flagged", 0)
    print(f"  QA pass 1: script_block flagged={flagged}")

    for attempt in range(1, max_retries + 1):
        if flagged == 0:
            break
        print(f"\n  retry {attempt}: purging {flagged} flagged hashes ...")
        run_qa_purge(region)
        run_translate(region)
        qa = run_qa(region)
        sb = qa.get("layer_stats", {}).get("script_block", {})
        flagged = sb.get("flagged", 0)
        print(f"  QA pass {attempt + 1}: script_block flagged={flagged}")

    elapsed = time.time() - t0
    qa["_elapsed_min"] = elapsed / 60.0
    qa["_final_script_block"] = flagged
    return qa


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", help="single region (e.g. IT_ROME)")
    p.add_argument("--resume", action="store_true",
                   help="skip tracks whose QA already shows 0 script_block")
    p.add_argument("--max-retries", type=int, default=2,
                   help="purge-retry rounds per track")
    args = p.parse_args()

    regions = [args.pilot] if args.pilot else SERIAL_ORDER
    if args.resume:
        regions = [r for r in regions if not already_clean(r)]
        print(f"resume: {len(regions)} tracks remaining")

    if not regions:
        print("nothing to do")
        return 0

    summary: list[tuple[str, int, float]] = []
    for region in regions:
        report = build_one(region, max_retries=args.max_retries)
        summary.append((
            region,
            report.get("_final_script_block", 0),
            report.get("_elapsed_min", 0.0),
        ))

    print("\n" + "=" * 70)
    print(f"{'Region':22s} {'script_block':14s} {'elapsed':>10s}")
    print("-" * 70)
    for region, flagged, elapsed in summary:
        print(f"{region:22s} {flagged:<14d} {elapsed:.1f} min")
    n_clean = sum(1 for _, f, _ in summary if f == 0)
    print(f"\n{n_clean}/{len(summary)} tracks reached 0 script_block flags")
    return 0 if n_clean == len(summary) else 1


if __name__ == "__main__":
    sys.exit(main())

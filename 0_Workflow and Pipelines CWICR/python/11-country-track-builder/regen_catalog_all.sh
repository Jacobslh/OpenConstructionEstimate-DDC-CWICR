#!/usr/bin/env bash
# One subprocess per track to avoid OOM accumulation across iterations.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
LOG="$HERE/logs/regen_catalog.log"
mkdir -p "$(dirname "$LOG")"
TRACKS=(
  IT_ROME NL_AMSTERDAM SV_STOCKHOLM HR_ZAGREB
  CS_PRAGUE PL_WARSAW RO_BUCHAREST TR_ISTANBUL BG_SOFIA
  ID_JAKARTA VI_HANOI JA_TOKYO KO_SEOUL TH_BANGKOK
)
echo "=== regen_catalog start: $(date) ===" | tee -a "$LOG"
for t in "${TRACKS[@]}"; do
  python -u regen_catalog_xlsx.py "$t" 2>&1 | tee -a "$LOG"
  echo "[$t exit=${PIPESTATUS[0]}]" | tee -a "$LOG"
done
echo "=== regen_catalog done: $(date) ===" | tee -a "$LOG"

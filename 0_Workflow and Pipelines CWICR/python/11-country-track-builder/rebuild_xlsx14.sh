#!/usr/bin/env bash
# Wave 2: regenerate real xlsx tables (--full-xlsx) for all 14 translate tracks.
# Cache fully populated, LLM disabled, embeddings skipped — only xlsx + catalog
# xlsx artifacts get rewritten with real data instead of LFS-pointer placeholders.
set -u
export PYTHONUNBUFFERED=1
export LLM_TOP_N=0
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
LOG="$HERE/logs/rebuild_xlsx14.log"
mkdir -p "$(dirname "$LOG")"
TRACKS=(
  IT_ROME NL_AMSTERDAM SV_STOCKHOLM HR_ZAGREB
  CS_PRAGUE PL_WARSAW RO_BUCHAREST TR_ISTANBUL BG_SOFIA
  ID_JAKARTA VI_HANOI JA_TOKYO KO_SEOUL TH_BANGKOK
)
echo "=== rebuild_xlsx14 (--full-xlsx) start: $(date) ===" | tee -a "$LOG"
for t in "${TRACKS[@]}"; do
  echo "" | tee -a "$LOG"
  echo "=== $t @ $(date '+%H:%M:%S') ===" | tee -a "$LOG"
  python -u add_country_track.py --config "configs/$t.yaml" --skip-embeddings --full-xlsx 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  echo "[$t exit=$rc]" | tee -a "$LOG"
done
echo "" | tee -a "$LOG"
echo "=== rebuild_xlsx14 done: $(date) ===" | tee -a "$LOG"

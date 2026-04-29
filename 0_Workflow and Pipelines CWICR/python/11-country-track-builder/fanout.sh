#!/usr/bin/env bash
# Parallel fan-out for 13 remaining translate-tracks (IT pilot is built
# separately first). 4 workers per wave; LLM_CONCURRENCY=3 keeps OpenAI
# Tier 1 within 200K TPM.
#
# Usage:
#   bash fanout.sh                 # all remaining 13 tracks
#   bash fanout.sh --waves 1 2     # only waves 1 and 2
#
# Wall time on cache-warm rebuild: ~12 min/track × 4 waves ≈ 50-90 min.
# Failed tracks list at logs/_failed.txt for retry.

set -u  # but NOT -e: we want to continue past per-track failures.

cd "$(dirname "$0")"
mkdir -p logs

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export LLM_CONCURRENCY=3
export LLM_TOP_N=200000

run_track() {
  local r="$1"
  local log="logs/${r}.log"
  echo "[$(date +%H:%M:%S)] START $r" | tee -a logs/_master.log
  python -u add_country_track.py --config "configs/${r}.yaml" --skip-embeddings \
      > "$log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] END $r rc=$rc" | tee -a logs/_master.log
  if [ "$rc" -ne 0 ]; then
    echo "$r" >> logs/_failed.txt
  fi
  return "$rc"
}
export -f run_track

# 2-worker waves (xargs -P2). Memory peak ~7 GB total — safe on 16 GB.
WAVES=(
  "NL_AMSTERDAM HR_ZAGREB"
  "PL_WARSAW CS_PRAGUE"
  "BG_SOFIA RO_BUCHAREST"
  "SV_STOCKHOLM TR_ISTANBUL"
  "ID_JAKARTA VI_HANOI"
  "JA_TOKYO KO_SEOUL"
  "TH_BANGKOK"
)

PICK_WAVES=()
if [ "${1:-}" = "--waves" ]; then
  shift
  for n in "$@"; do
    idx=$((n - 1))
    PICK_WAVES+=("${WAVES[$idx]}")
  done
else
  PICK_WAVES=("${WAVES[@]}")
fi

> logs/_failed.txt
echo "[$(date +%H:%M:%S)] FAN-OUT START — ${#PICK_WAVES[@]} waves" | tee -a logs/_master.log

for w in "${PICK_WAVES[@]}"; do
  echo "[$(date +%H:%M:%S)] WAVE: $w" | tee -a logs/_master.log
  printf '%s\n' $w | xargs -n1 -P2 -I{} bash -c 'run_track "$@"' _ {}
  echo "[$(date +%H:%M:%S)] WAVE END" | tee -a logs/_master.log
done

n_ok=0
n_fail=0
for r in $(printf '%s\n' "${PICK_WAVES[@]}"); do
  for region in $r; do
    if grep -q "END $region rc=0" logs/_master.log; then
      n_ok=$((n_ok + 1))
    else
      n_fail=$((n_fail + 1))
    fi
  done
done

echo
echo "=== FAN-OUT SUMMARY ==="
echo "OK:   $n_ok"
echo "FAIL: $n_fail"
if [ -s logs/_failed.txt ]; then
  echo "Failed tracks:"
  cat logs/_failed.txt
fi

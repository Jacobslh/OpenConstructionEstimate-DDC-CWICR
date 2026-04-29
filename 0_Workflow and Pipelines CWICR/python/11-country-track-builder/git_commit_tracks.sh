#!/usr/bin/env bash
# Per-track git commit + push for the 14 retranslated tracks.
# Run from the builder dir; navigates up to repo root.
#
# Each track folder contains regenerated:
#   - <REGION>_workitems_costs_resources_DDC_CWICR.parquet  (data, ~25 MB)
#   - DDC_CWICR_<REGION>_Catalog.csv  (~2.7 MB)
#   - <REGION>_..._SIMPLE.xlsx, ..._FORMATTED.xlsx, Catalog.xlsx  (LFS placeholders)
#   - README.md, ..._TABULAR_*.txt, ..._TABULAR_*.pdf
#
# Commit per-folder so reviewers / git-blame stays clean.
# We do NOT push automatically — author should review before pushing.

set -e
cd "$(dirname "$0")"
REPO_ROOT="$(cd ../../../.. && pwd)"
cd "$REPO_ROOT"

TRACKS=(
  "IT___DDC_CWICR"   "IT_ROME"      "Italian"
  "NL___DDC_CWICR"   "NL_AMSTERDAM" "Dutch"
  "HR___DDC_CWICR"   "HR_ZAGREB"    "Croatian"
  "PL___DDC_CWICR"   "PL_WARSAW"    "Polish"
  "CS___DDC_CWICR"   "CS_PRAGUE"    "Czech"
  "RO___DDC_CWICR"   "RO_BUCHAREST" "Romanian"
  "SV___DDC_CWICR"   "SV_STOCKHOLM" "Swedish"
  "TR___DDC_CWICR"   "TR_ISTANBUL"  "Turkish"
  "ID___DDC_CWICR"   "ID_JAKARTA"   "Indonesian"
  "VI___DDC_CWICR"   "VI_HANOI"     "Vietnamese"
  "JA___DDC_CWICR"   "JA_TOKYO"     "Japanese"
  "KO___DDC_CWICR"   "KO_SEOUL"     "Korean"
  "TH___DDC_CWICR"   "TH_BANGKOK"   "Thai"
  "BG___DDC_CWICR"   "BG_SOFIA"     "Bulgarian"
)

n=${#TRACKS[@]}
i=0
while [ $i -lt $n ]; do
  folder="${TRACKS[$i]}"
  region="${TRACKS[$((i+1))]}"
  language="${TRACKS[$((i+2))]}"

  echo
  echo "=== $folder ($region, $language) ==="
  cd "$REPO_ROOT"

  if [ ! -d "$folder" ]; then
    echo "  folder missing, skip"
    i=$((i + 3))
    continue
  fi

  git add "$folder/"
  if git diff --cached --quiet "$folder/"; then
    echo "  no staged changes, skip"
    i=$((i + 3))
    continue
  fi

  git commit -m "$(cat <<EOF
data($region): retranslate with expanded TEXT_COLS, full long-tail, BG cache cleanup

Rebuild $region track ($language) with the v2 translation pipeline:
- TEXT_COLS expanded 22 -> 30 to translate department_type, section_type,
  row_type, category_type, personnel_operator_grade,
  masterformat_section_title, masterformat_division,
  price_abstract_resource_unit (closes the German/English leakage gap
  in non-source tracks).
- LLM_TOP_N=200000 — full long-tail re-translation, no Google fallback.
- BG cache pre-cleaned for 23 RU-leakage entries.
- Regenerated parquet, Catalog CSV, SIMPLE/FORMATTED.xlsx, README.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
  echo "  committed $folder"
  i=$((i + 3))
done

echo
echo "=== DONE ==="
echo "Run 'git log --oneline -20' to review."
echo "Run 'git push origin main' when ready to publish."

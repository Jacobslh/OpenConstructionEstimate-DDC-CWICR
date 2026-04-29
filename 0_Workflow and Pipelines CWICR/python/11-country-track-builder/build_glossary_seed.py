"""
Build construction_glossary.json from existing tracks.

For each (src_lang, tgt_lang) pair we have data for, extract parallel
strings aligned on rate_code / resource_code. Output is consumed by
text_pipeline.py as in-context vocabulary for Claude translations of
new tracks.

Produces a hierarchical dict:

    {
      "<src_lang>_<tgt_lang>": {
        "<src_text>": "<tgt_text>",
        ...
      },
      ...
    }

We dedup at the source-text level (last occurrence wins; in practice
they're identical because the dataset is one harmonised table).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

from tracks import EXISTING_TRACKS, get_track


import env_loader  # noqa: F401  — auto-loads .env into os.environ


HERE = Path(__file__).resolve().parent
GLOSSARY_DIR = HERE / "glossary"
SEED_GLOSSARY = GLOSSARY_DIR / "construction_glossary.json"


# Pairs we extract — short list to keep file size manageable. Add more
# as needed; we only really care about pairs whose source matches a
# real source_track and whose target spans a target_lang.
PAIRS = [
    # English source pairs (cover AU, NZ, NG, ZA, JA, KO, ID, VI, TH targets)
    ("UK_GBP", "DE_BERLIN"),    # en -> de
    ("UK_GBP", "FR_PARIS"),     # en -> fr
    ("UK_GBP", "ES_BARCELONA"), # en -> es (existing folder is ES___DDC_CWICR)
    ("UK_GBP", "RU_STPETERSBURG"),
    ("UK_GBP", "ZH_SHANGHAI"),
    ("UK_GBP", "AR_DUBAI"),
    ("UK_GBP", "HI_MUMBAI"),
    # German source pairs (cover HR, BG, IT, NL, PL, SV, CS, TR, RO targets)
    ("DE_BERLIN", "UK_GBP"),    # de -> en
    ("DE_BERLIN", "FR_PARIS"),
    ("DE_BERLIN", "ES_BARCELONA"),
    ("DE_BERLIN", "RU_STPETERSBURG"),
    # Spanish source pair (covers MX target)
    ("ES_BARCELONA", "UK_GBP"),
    ("ES_BARCELONA", "DE_BERLIN"),
]

# Text columns to extract pairs from.
GLOSSARY_TEXT_COLS = (
    "rate_original_name", "rate_final_name", "rate_unit",
    "resource_name", "resource_unit",
    "department_name", "section_name", "subsection_name",
    "labor_class", "labor_title", "operator_class",
    "machine_class3_name", "machine_class2_name",
)

# Cap pair count per language pair; the head terms by frequency are what
# matters for Claude's in-context glossary.
PAIR_CAP = 2000

# 14 production pairs that have no parallel-data source — they are the
# new translate-tracks. Seeded by pivoting through EN: take top-N
# construction terms from the existing en_de glossary, translate the EN
# side once per target via Claude, then attach the result to every
# DE-source pair via the de->en inverse.
PIVOT_TARGET_LANGS_FROM_DE = ("bg", "cs", "hr", "it", "nl", "pl", "ro", "sv", "tr")
PIVOT_TARGET_LANGS_FROM_EN = ("id", "ja", "ko", "th", "vi")
PIVOT_TOP_N = 2000


def _llm_translate_terms(terms: list[str], tgt_lang: str) -> list[str]:
    """Translate construction terms EN -> tgt_lang via LLM in batches.

    Picks Claude when ANTHROPIC_API_KEY is set (preferred), otherwise
    falls back to OpenAI gpt-4o-mini. Returns aligned list. On failure
    keeps the source term verbatim so the caller can drop it.
    """
    sys_p_template = (
        "You translate construction-industry technical terms from "
        "English to {tgt}. Preserve all numbers, units, codes, "
        "abbreviations, and dimensions exactly. Do not paraphrase. "
        "Use industry-standard terminology. Respond as a numbered "
        "list, one translation per input line, no preamble."
    )

    use_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
    use_openai = bool(os.environ.get("OPENAI_API_KEY"))
    if not (use_claude or use_openai):
        raise RuntimeError(
            "Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set"
        )

    if use_claude:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        model = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
        backend = "claude"
    else:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model = os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-4o-mini")
        backend = "openai"

    out: list[str] = []
    BATCH = 50
    for i in range(0, len(terms), BATCH):
        chunk = terms[i:i + BATCH]
        numbered = "\n".join(f"{j+1}. {t}" for j, t in enumerate(chunk))
        sys_p = sys_p_template.format(tgt=tgt_lang)
        try:
            if backend == "claude":
                msg = client.messages.create(
                    model=model, max_tokens=4096, system=sys_p,
                    messages=[{"role": "user", "content": numbered}],
                )
                text = msg.content[0].text
            else:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_p},
                        {"role": "user", "content": numbered},
                    ],
                    max_tokens=4096,
                    temperature=0.0,
                    timeout=60,
                )
                text = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"      {backend} error on batch {i}: {e}; keeping source terms")
            out.extend(chunk)
            continue

        parsed = [""] * len(chunk)
        for ln in text.splitlines():
            body = ln.lstrip()
            if not body:
                continue
            for sep in (".", ")", " -", ":"):
                if sep in body[:6]:
                    head, _, rest = body.partition(sep)
                    if head.strip().isdigit():
                        idx = int(head.strip()) - 1
                        if 0 <= idx < len(chunk):
                            parsed[idx] = rest.strip()
                            break
        for j, p in enumerate(parsed):
            if not p:
                parsed[j] = chunk[j]
        out.extend(parsed)
        print(f"      en->{tgt_lang} ({backend}): {min(i + BATCH, len(terms))}/{len(terms)}")
    return out


# Back-compat alias.
_claude_translate_terms = _llm_translate_terms


def pivot_seed(glossary: dict[str, dict[str, str]]) -> None:
    """Mutate `glossary` in place: add 14 new pairs by pivoting EN seeds.

    Strategy:
      1. Take top-N EN terms from en_de glossary (these are real
         construction vocabulary, dataset-frequency-ranked).
      2. For each target lang in PIVOT_TARGET_LANGS_FROM_EN: translate
         the EN list once and store as `en_<tgt>`.
      3. For each target lang in PIVOT_TARGET_LANGS_FROM_DE: chain
         DE->EN (via en_de inverse) -> target. Store as `de_<tgt>`.
    """
    en_de = glossary.get("en_de", {})
    if not en_de:
        print("\nPivot seed: en_de glossary is empty — skipping.")
        return

    en_terms = list(en_de.keys())[:PIVOT_TOP_N]
    print(f"\n=== EN-pivot glossary seed ({len(en_terms)} terms) ===")

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set; skipping pivot seed.")
        return

    # Build de->en inverse once.
    de_en_inverse: dict[str, str] = {}
    for en, de in en_de.items():
        if isinstance(de, str) and de.strip() and de not in de_en_inverse:
            de_en_inverse[de] = en

    all_targets = list(PIVOT_TARGET_LANGS_FROM_EN) + list(PIVOT_TARGET_LANGS_FROM_DE)
    for tgt_lang in all_targets:
        en_to_tgt_key = f"en_{tgt_lang}"
        if glossary.get(en_to_tgt_key) and len(glossary[en_to_tgt_key]) >= PIVOT_TOP_N // 2:
            print(f"  {en_to_tgt_key}: already populated ({len(glossary[en_to_tgt_key])}); skip")
            tgt_translations = list(glossary[en_to_tgt_key].values())[:len(en_terms)]
            # Pad if shorter
            tgt_translations += en_terms[len(tgt_translations):]
        else:
            print(f"  translating {len(en_terms)} EN terms -> {tgt_lang} via LLM ...")
            tgt_translations = _llm_translate_terms(en_terms, tgt_lang)

        en_pairs: dict[str, str] = {}
        for en, tgt in zip(en_terms, tgt_translations):
            if not isinstance(tgt, str) or not tgt.strip():
                continue
            if tgt == en:  # likely Claude refused/unable; skip
                continue
            en_pairs[en] = tgt
        glossary[en_to_tgt_key].update(en_pairs)
        print(f"      en_{tgt_lang}: {len(en_pairs)} pairs")

        # For DE-targets only: chain DE->EN->target.
        if tgt_lang in PIVOT_TARGET_LANGS_FROM_DE:
            de_to_tgt_key = f"de_{tgt_lang}"
            de_pairs: dict[str, str] = {}
            for de, en in de_en_inverse.items():
                tgt = en_pairs.get(en)
                if not tgt:
                    continue
                de_pairs[de] = tgt
            glossary[de_to_tgt_key].update(de_pairs)
            print(f"      de_{tgt_lang}: {len(de_pairs)} pairs (via DE->EN->{tgt_lang})")


def _load_track(region: str) -> pd.DataFrame | None:
    track = get_track(region)
    if not track.parquet_path.exists():
        print(f"  skip {region}: parquet not found")
        return None
    df = pd.read_parquet(track.parquet_path)
    return df


def build_pair(src_df: pd.DataFrame, tgt_df: pd.DataFrame) -> dict[str, str]:
    """
    Align src and tgt by rate_code, build a {src: tgt} dict.

    Memory-friendly approach: for each text column, collapse each source
    dataframe to one row per rate_code (first non-null value per code),
    then dict-lookup target by source's rate_code. Avoids the full N×M
    merge that explodes RAM with 900K-row inputs.
    """
    if "rate_code" not in src_df.columns or "rate_code" not in tgt_df.columns:
        return {}

    pairs: dict[str, str] = {}
    for col in GLOSSARY_TEXT_COLS:
        if col not in src_df.columns or col not in tgt_df.columns:
            continue

        # First non-null per rate_code on each side.
        src_map = (
            src_df.dropna(subset=[col])
                  .drop_duplicates(subset=["rate_code"])[["rate_code", col]]
                  .set_index("rate_code")[col]
                  .to_dict()
        )
        tgt_map = (
            tgt_df.dropna(subset=[col])
                  .drop_duplicates(subset=["rate_code"])[["rate_code", col]]
                  .set_index("rate_code")[col]
                  .to_dict()
        )

        for code, s in src_map.items():
            if not isinstance(s, str) or not s.strip():
                continue
            t = tgt_map.get(code)
            if not isinstance(t, str) or not t.strip():
                continue
            if s == t:
                continue
            if s not in pairs:
                pairs[s] = t
            if len(pairs) >= PAIR_CAP:
                return pairs
    return pairs


def main() -> int:
    GLOSSARY_DIR.mkdir(parents=True, exist_ok=True)

    # ES___ folder is recorded in tracks.py as SP_BARCELONA (filename prefix).
    # The pairs list above uses ES_BARCELONA as the registry key — fix it.
    pair_aliases = {"ES_BARCELONA": "SP_BARCELONA"}

    glossary: dict[str, dict[str, str]] = defaultdict(dict)

    # Cache loaded dataframes.
    df_cache: dict[str, pd.DataFrame] = {}

    def _load(region: str) -> pd.DataFrame | None:
        region = pair_aliases.get(region, region)
        if region not in df_cache:
            print(f"  loading {region} ...")
            df = _load_track(region)
            if df is not None:
                df_cache[region] = df
        return df_cache.get(region)

    for src, tgt in PAIRS:
        print(f"\n{src} -> {tgt}")
        sdf = _load(src)
        tdf = _load(tgt)
        if sdf is None or tdf is None:
            continue

        src_lang = get_track(pair_aliases.get(src, src)).language
        tgt_lang = get_track(pair_aliases.get(tgt, tgt)).language
        key = f"{src_lang}_{tgt_lang}"

        pairs = build_pair(sdf, tdf)
        glossary[key].update(pairs)
        print(f"  {key}: +{len(pairs)} pairs (total {len(glossary[key])})")

    # Phase 2: pivot-seed the 14 production pairs that have no parallel
    # data. Uses the freshly built en_de glossary as backbone. Free no-op
    # when ANTHROPIC_API_KEY is not set or seeds already exist.
    if os.environ.get("DDC_SEED_PIVOT", "1") != "0":
        pivot_seed(glossary)

    # Atomic write — concurrent translate jobs may be reading this file.
    payload = json.dumps(glossary, ensure_ascii=False, indent=2)
    tmp = SEED_GLOSSARY.with_suffix(SEED_GLOSSARY.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(SEED_GLOSSARY)
    total = sum(len(v) for v in glossary.values())
    print(
        f"\nWrote {SEED_GLOSSARY} with {len(glossary)} language pairs, "
        f"{total} total pairs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

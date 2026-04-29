"""
Quality validation for translated DDC CWICR tracks.

Six audit layers run against either a track's parquet or its translation
cache (glossary/translations/<src>_<tgt>.json). Each layer emits Flagged
records that can be fed back into the cache by `purge_flagged()` so the
next translation run regenerates them.

Layers:
  (a) script_block      — value uses the expected Unicode block(s) for
                           the target language (Cyrillic for BG, Hangul
                           for KO, etc.). Catches leftover source text.
  (b) ru_leakage         — BG-specific. Flags Russian orthographic
                           markers absent in true Bulgarian.
  (c) round_trip         — sample N random translations, back-translate
                           target -> source via a different model, and
                           compute embedding cosine to original source.
                           Flags below threshold.
  (d) glossary_consistency — for every (src_term, tgt_term) in the seed
                           glossary, check that translations of strings
                           containing src_term contain tgt_term.
  (e) numeric_preservation — every digit/unit token in the source must
                           also appear in the target. Catches dropped
                           dimensions and hallucinated numbers.
  (f) length_sanity      — ratio target_len/source_len within [0.3, 3].
                           Catches truncation and run-on translation.

Usage:
  python -m quality_check.translation_validators --track BG_SOFIA
  python -m quality_check.translation_validators --all
  python -m quality_check.translation_validators --track BG_SOFIA --purge

The --purge flag removes flagged hashes from the translation cache so a
re-run of add_country_track.py picks them up as pending.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import unicodedata
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable

import pandas as pd

# Make sibling modules importable when run as `python -m`.
_HERE = Path(__file__).resolve().parent
_BUILDER = _HERE.parent
if str(_BUILDER) not in sys.path:
    sys.path.insert(0, str(_BUILDER))

import env_loader  # noqa: E402, F401  — auto-loads .env into os.environ
from text_pipeline import TEXT_COLS, _hash, TranslationCache  # noqa: E402
from tracks import ALL_TRACKS, get_track  # noqa: E402


GLOSSARY_DIR = _BUILDER / "glossary"
SEED_GLOSSARY = GLOSSARY_DIR / "construction_glossary.json"
TRANSLATIONS_DIR = GLOSSARY_DIR / "translations"
REPORTS_DIR = _BUILDER / "validation_reports"


# ---------------------------------------------------------------------------
# Per-language script expectations
# ---------------------------------------------------------------------------

SCRIPT_BLOCKS = {
    "Cyrillic": ("CYRILLIC",),
    "Latin": ("LATIN",),
    "Hangul": ("HANGUL",),
    "Thai": ("THAI",),
    "Hiragana": ("HIRAGANA",),
    "Katakana": ("KATAKANA",),
    "CJK": ("CJK",),
    "Arabic": ("ARABIC",),
    "Devanagari": ("DEVANAGARI",),
    "Hebrew": ("HEBREW",),
}

# Per-target-language: which Unicode block prefixes are acceptable.
# A value passes if >= MIN_BLOCK_RATIO of its letter chars belong to
# any allowed block.
LANG_RULES: dict[str, tuple[str, ...]] = {
    "bg": ("CYRILLIC",),
    "ru": ("CYRILLIC",),
    "ja": ("HIRAGANA", "KATAKANA", "CJK"),
    "ko": ("HANGUL", "CJK"),
    "th": ("THAI",),
    "zh": ("CJK",),
    "ar": ("ARABIC",),
    "he": ("HEBREW",),
    "hi": ("DEVANAGARI",),
    # Latin-script languages — diacritics are fine, but block must be Latin.
    "en": ("LATIN",),
    "de": ("LATIN",),
    "fr": ("LATIN",),
    "es": ("LATIN",),
    "pt": ("LATIN",),
    "it": ("LATIN",),
    "nl": ("LATIN",),
    "sv": ("LATIN",),
    "cs": ("LATIN",),
    "pl": ("LATIN",),
    "hr": ("LATIN",),
    "ro": ("LATIN",),
    "tr": ("LATIN",),
    "vi": ("LATIN",),
    "id": ("LATIN",),
}
# For non-Latin target languages: any presence of target-script letters
# is enough to assume the LLM translated. Without this, mixed strings
# like "243 kW (330 к.с.)" trip the ratio test even though they ARE
# correctly partially-translated. Latin-script targets have a separate
# code path that skips the script-block check entirely (since source is
# also Latin).
NON_LATIN_LANGS = {"bg", "ru", "ja", "ko", "th", "zh", "ar", "he", "hi"}
# 15% target-script-letters threshold. Catches cases like
# "Temperature sensor 0...100°C, модель RT100" where only 2 words are
# translated (5%-ish ratio), while still letting through legitimate
# mixed strings like "243 kW (330 к.с.)".
MIN_BLOCK_RATIO = 0.15

# RU-leakage detector for BG.
RU_MARKERS = re.compile(
    r"(сс|ё|[а-я]ый\b|[а-я]ого\b|[а-я]ые\b|[а-я]ться\b|жн[а-я])",
    re.UNICODE | re.IGNORECASE,
)
BG_NEGATIVE = re.compile(
    r"(ще|щ[ае]|ъ|[а-я]ия\b|[а-я]ите\b|[а-я]ват\b)",
    re.UNICODE | re.IGNORECASE,
)

# Numeric/unit token extraction. Each match must survive translation.
NUMERIC_TOKEN_RE = re.compile(
    r"(\d+(?:[.,]\d+)?(?:\s*[xх×*]\s*\d+(?:[.,]\d+)?)*"
    r"(?:\s*[mкM]?м[²³2³]?|"
    r"\s*kg|\s*т|\s*l|\s*мм|\s*см|\s*м[²³2³]?|"
    r"\s*%|\s*°|\s*Вт|\s*кВт)?)",
    re.UNICODE,
)
PURE_NUMBER_RE = re.compile(r"^[\d\s.,/x*×²³%·\-:_+()]+$", re.UNICODE)

# International technical specs that are language-independent: a number
# (with optional decimal/comma) followed by one of the known SI / domain
# unit abbreviations. These pass through translation untouched and would
# trigger a script-block false positive otherwise.
INTL_UNITS = (
    "kVA", "kW", "MW", "GW", "Wh", "kWh", "MWh",
    "V", "A", "mA", "Hz", "kHz", "MHz",
    "kg", "g", "mg", "t", "kt", "Mt",
    "m", "cm", "mm", "km", "nm", "pm",
    "m²", "m³", "cm²", "cm³", "mm²", "mm³",
    "m2", "m3", "cm2", "cm3", "mm2", "mm3",
    "m3/h", "m2/h", "m3/s",
    "l", "L", "ml", "cl", "dl", "hl",
    "h", "min", "s", "ms", "kHz",
    "kN", "MN", "Pa", "kPa", "MPa", "GPa", "bar",
    "°C", "°F", "K", "%", "ppm", "Bq", "Sv",
    "rpm", "Nm", "Wb", "T",
    "DN", "PN", "PE", "PVC", "ABS", "PP", "HDPE",
)
_UNIT_RE_BODY = "|".join(re.escape(u) for u in sorted(INTL_UNITS, key=len, reverse=True))
TECH_SPEC_RE = re.compile(
    rf"^\s*\d+(?:[.,]\d+)?(?:\s*[xх×*]\s*\d+(?:[.,]\d+)?)*\s*"
    rf"(?:{_UNIT_RE_BODY})(?:\s*[/×x]\s*(?:{_UNIT_RE_BODY}))*\s*$",
    re.UNICODE,
)


def _is_passthrough(s: str) -> bool:
    """True if the string is language-independent (numeric, code,
    international unit spec) and should not be flagged on script-block."""
    if PURE_NUMBER_RE.match(s):
        return True
    if TECH_SPEC_RE.match(s):
        return True
    # Codes like "DN-200", "PN16", "M12x40" — uppercase letters + digits.
    if re.match(r"^[A-Z0-9][A-Z0-9\-/×x×*.,\s]*$", s) and len(s) <= 30:
        return True
    return False


@dataclass
class Flagged:
    layer: str
    column: str
    hash: str
    source_value: str
    target_value: str
    reason: str
    severity: str = "error"  # error | warn | info


@dataclass
class TrackAudit:
    region: str
    target_lang: str
    source_lang: str
    rows: int
    flagged: list[Flagged] = field(default_factory=list)
    layer_stats: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, flag: Flagged) -> None:
        self.flagged.append(flag)
        s = self.layer_stats.setdefault(flag.layer, {"flagged": 0, "checked": 0})
        s["flagged"] += 1

    def bump_checked(self, layer: str, n: int = 1) -> None:
        s = self.layer_stats.setdefault(layer, {"flagged": 0, "checked": 0})
        s["checked"] += n


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dominant_block(s: str) -> str:
    """Return the dominant Unicode block prefix among letter chars."""
    counts: dict[str, int] = {}
    for ch in s:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        for prefix in ("CYRILLIC", "LATIN", "HANGUL", "THAI", "HIRAGANA",
                       "KATAKANA", "CJK", "ARABIC", "HEBREW", "DEVANAGARI"):
            if name.startswith(prefix):
                counts[prefix] = counts.get(prefix, 0) + 1
                break
    if not counts:
        return ""
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _block_ratio(s: str, allowed: Iterable[str]) -> float:
    """Fraction of letter chars in `s` that belong to any allowed block."""
    total = 0
    hits = 0
    for ch in s:
        if not ch.isalpha():
            continue
        total += 1
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if any(name.startswith(p) for p in allowed):
            hits += 1
    return hits / total if total else 1.0


def _extract_numeric_tokens(s: str) -> list[str]:
    """Numeric tokens that must survive translation.

    Excludes:
      - single digits glued to letters (m2, m3, cm3, IPE300) — these are
        unit suffixes, not standalone quantities, and the translation
        commonly rewrites m3 -> m³ which would otherwise drop the "3".
      - bare 1-digit numbers without surrounding context — too noisy.

    Keeps:
      - multi-digit numbers (>= 2 digits)
      - decimal numbers (with . or , separator)
    """
    out = []
    # Match digit sequences that are NOT preceded by a letter (so "m3"
    # is excluded but "1,6 m3" still extracts "1,6"). Use a negative
    # lookbehind for a letter character.
    for m in re.finditer(
        r"(?<![A-Za-zА-Яа-я])"          # not preceded by a letter
        r"\d+(?:[.,]\d+)?"               # number, possibly decimal
        r"(?![A-Za-zА-Яа-я])",          # not followed by a letter
        s,
    ):
        token = m.group(0)
        # Drop bare 1-digit "noise". They are rarely critical and
        # frequently appear in unit fragments after stripping.
        if len(token) == 1 and "." not in token and "," not in token:
            continue
        out.append(token)
    return out


# ---------------------------------------------------------------------------
# Layer (a) — Unicode script-block check
# ---------------------------------------------------------------------------

def layer_script_block(audit: TrackAudit, df: pd.DataFrame) -> None:
    allowed = LANG_RULES.get(audit.target_lang)
    if not allowed:
        print(f"  layer (a): no rule for {audit.target_lang}; skipping")
        return
    seen_hashes: set[str] = set()
    for col in TEXT_COLS:
        if col not in df.columns:
            continue
        for v in df[col].dropna().unique():
            if not isinstance(v, str) or not v.strip():
                continue
            if _is_passthrough(v):
                continue
            h = _hash(v)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            audit.bump_checked("script_block")
            ratio = _block_ratio(v, allowed)
            if ratio < MIN_BLOCK_RATIO:
                dom = _dominant_block(v) or "none"
                audit.add(Flagged(
                    layer="script_block",
                    column=col,
                    hash=h,
                    source_value="",
                    target_value=v,
                    reason=f"only {ratio:.0%} chars in {allowed}, dominant={dom}",
                ))


# ---------------------------------------------------------------------------
# Layer (b) — RU-leakage detector for BG
# ---------------------------------------------------------------------------

def layer_ru_leakage(audit: TrackAudit, df: pd.DataFrame) -> None:
    if audit.target_lang != "bg":
        return
    seen_hashes: set[str] = set()
    for col in TEXT_COLS:
        if col not in df.columns:
            continue
        for v in df[col].dropna().unique():
            if not isinstance(v, str) or not v.strip():
                continue
            h = _hash(v)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            audit.bump_checked("ru_leakage")
            ru = len(RU_MARKERS.findall(v))
            bg = len(BG_NEGATIVE.findall(v))
            if ru >= 2 and bg == 0:
                audit.add(Flagged(
                    layer="ru_leakage",
                    column=col,
                    hash=h,
                    source_value="",
                    target_value=v,
                    reason=f"RU markers={ru} BG markers={bg}",
                ))


# ---------------------------------------------------------------------------
# Layer (c) — Round-trip back-translation cosine (sample-based)
# ---------------------------------------------------------------------------

def layer_round_trip(
    audit: TrackAudit,
    cache: TranslationCache,
    sample_size: int = 500,
    cosine_threshold: float = 0.7,
) -> None:
    """Sample N random translations, back-translate target -> source via a
    different model, embed both and compare cosine. Flag low scores.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        print("  layer (c): OPENAI_API_KEY not set; skipping round-trip")
        return
    items = list(cache.map.items())
    if not items:
        print("  layer (c): empty cache; skipping")
        return
    # We need both src and tgt — but cache maps hash -> tgt. We need src.
    # Recover src from in-memory unique-strings extraction by the caller.
    # Caller passes src_strings via audit.layer_stats setup? Cleaner: do
    # round-trip on the (src, tgt) pairs extracted from the dataframe.
    # That is done in audit_track().
    pass  # actual implementation in audit_track() — needs df context


def _round_trip_on_pairs(
    audit: TrackAudit,
    pairs: list[tuple[str, str, str]],  # (hash, src_str, tgt_str)
    sample_size: int,
    cosine_threshold: float,
) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("  layer (c): OPENAI_API_KEY not set; skipping round-trip")
        return
    if not pairs:
        return

    rng = random.Random(42)
    sampled = pairs if len(pairs) <= sample_size else rng.sample(pairs, sample_size)
    print(f"  layer (c): round-trip on {len(sampled)} samples ...")

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    backtrans_model = os.getenv("BACKTRANS_MODEL", "gpt-4o-mini")
    embed_model = os.getenv("EMBED_MODEL_QA", "text-embedding-3-small")

    BATCH = 25
    for i in range(0, len(sampled), BATCH):
        chunk = sampled[i:i + BATCH]
        targets = [t for _, _, t in chunk]
        numbered = "\n".join(f"{j+1}. {t}" for j, t in enumerate(targets))
        sys_p = (
            f"You translate construction-industry technical strings from "
            f"{audit.target_lang} back to {audit.source_lang}. Preserve "
            f"all numbers, units, codes, abbreviations exactly. Numbered "
            f"list, one translation per line, no preamble."
        )
        try:
            resp = client.chat.completions.create(
                model=backtrans_model,
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
            print(f"      backtrans error on batch {i}: {e}; skipping batch")
            continue

        backs: list[str] = [""] * len(chunk)
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
                            backs[idx] = rest.strip()
                            break

        # Embed both originals and backs in one call.
        all_texts = [s for _, s, _ in chunk] + backs
        try:
            emb_resp = client.embeddings.create(
                model=embed_model, input=all_texts,
            )
        except Exception as e:
            print(f"      embed error: {e}; skipping batch")
            continue
        vecs = [d.embedding for d in emb_resp.data]
        n = len(chunk)
        srcs_v, backs_v = vecs[:n], vecs[n:]

        for j, (h, src, tgt) in enumerate(chunk):
            if not backs[j]:
                continue
            cos = _cosine(srcs_v[j], backs_v[j])
            audit.bump_checked("round_trip")
            if cos < cosine_threshold:
                audit.add(Flagged(
                    layer="round_trip",
                    column="*",
                    hash=h,
                    source_value=src,
                    target_value=tgt,
                    reason=f"backtrans cosine={cos:.2f} < {cosine_threshold}",
                ))
        print(f"      progress: {min(i + BATCH, len(sampled))}/{len(sampled)}")


def _cosine(a: list[float], b: list[float]) -> float:
    import math
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(x * x for x in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


# ---------------------------------------------------------------------------
# Layer (d) — Glossary consistency
# ---------------------------------------------------------------------------

def layer_glossary_consistency(
    audit: TrackAudit, pairs: list[tuple[str, str, str]],
) -> None:
    if not SEED_GLOSSARY.exists():
        return
    glossary = json.loads(SEED_GLOSSARY.read_text(encoding="utf-8"))
    pair_glossary = glossary.get(f"{audit.source_lang}_{audit.target_lang}", {})
    if not pair_glossary:
        return
    # Lower-case lookup.
    rules = [(k.lower(), v) for k, v in pair_glossary.items()
             if isinstance(k, str) and isinstance(v, str) and len(k) >= 3]
    for h, src, tgt in pairs:
        src_l = src.lower()
        for src_term, tgt_term in rules:
            if src_term not in src_l:
                continue
            audit.bump_checked("glossary_consistency")
            if tgt_term.lower() not in tgt.lower():
                audit.add(Flagged(
                    layer="glossary_consistency",
                    column="*",
                    hash=h,
                    source_value=src,
                    target_value=tgt,
                    reason=f"glossary requires '{tgt_term}' for '{src_term}'",
                    severity="warn",
                ))
                break


# ---------------------------------------------------------------------------
# Layer (e) — Numeric/unit preservation
# ---------------------------------------------------------------------------

def _normalise_decimal(s: str) -> str:
    """Treat "1.6" and "1,6" as equivalent. Many target languages use
    a comma decimal separator (DE, FR, IT, NL, ID, RU, BG, ...)."""
    return s.replace(".", ",")


def layer_numeric_preservation(
    audit: TrackAudit, pairs: list[tuple[str, str, str]],
) -> None:
    for h, src, tgt in pairs:
        src_nums = _extract_numeric_tokens(src)
        if not src_nums:
            continue
        audit.bump_checked("numeric_preservation")
        # Normalise decimal separator on BOTH sides before search.
        # Source may use "1.6" or "1,6"; target may convert in either
        # direction, so we compare on a single canonical form.
        tgt_norm = _normalise_decimal(tgt)
        missing = [
            n for n in src_nums
            if _normalise_decimal(n) not in tgt_norm
        ]
        if missing:
            audit.add(Flagged(
                layer="numeric_preservation",
                column="*",
                hash=h,
                source_value=src,
                target_value=tgt,
                reason=f"missing numeric tokens: {missing}",
            ))


# ---------------------------------------------------------------------------
# Layer (f) — Length sanity
# ---------------------------------------------------------------------------

def layer_length_sanity(
    audit: TrackAudit, pairs: list[tuple[str, str, str]],
    min_ratio: float = 0.3, max_ratio: float = 3.0,
) -> None:
    # CJK and Thai are dramatically more compact than Latin source — a
    # 50-char EN sentence often becomes a 12-char JA/KO sentence. Use a
    # relaxed lower bound; they rarely become longer.
    if audit.target_lang in {"ja", "ko", "zh", "th"}:
        min_ratio, max_ratio = 0.1, 2.0
    for h, src, tgt in pairs:
        if len(src) < 5 or len(tgt) < 1:
            continue
        audit.bump_checked("length_sanity")
        ratio = len(tgt) / len(src)
        if ratio < min_ratio or ratio > max_ratio:
            audit.add(Flagged(
                layer="length_sanity",
                column="*",
                hash=h,
                source_value=src,
                target_value=tgt,
                reason=f"len ratio={ratio:.2f} outside [{min_ratio}, {max_ratio}]",
                severity="warn",
            ))


# ---------------------------------------------------------------------------
# Track audit orchestration
# ---------------------------------------------------------------------------

def audit_track(
    region: str,
    sample_size: int = 500,
    skip_round_trip: bool = False,
) -> TrackAudit:
    track = get_track(region)
    if not track.parquet_path.exists():
        raise FileNotFoundError(f"target parquet missing: {track.parquet_path}")
    if track.source_track is None:
        raise ValueError(f"target track {region} has no source_track")
    source = get_track(track.source_track)
    if source.language == track.language:
        print(f"  {region}: same language ({track.language}); skip")
        return TrackAudit(region, track.language, source.language, 0)

    print(f"\n=== {region} ({source.language} -> {track.language}) ===")
    # Memory-friendly load: only TEXT_COLS, not all 93 columns. The full
    # parquet is ~900K rows × 93 cols of objects; loading everything
    # plus pandas sort_values doubles peak memory and OOMs on the
    # bigger tracks.
    import pyarrow.parquet as pq
    tgt_cols_avail = set(pq.ParquetFile(track.parquet_path).schema.names)
    src_cols_avail = set(pq.ParquetFile(source.parquet_path).schema.names)
    common_text = [c for c in TEXT_COLS if c in tgt_cols_avail and c in src_cols_avail]
    target_df = pd.read_parquet(track.parquet_path, columns=common_text)
    source_df = pd.read_parquet(source.parquet_path, columns=common_text)

    audit = TrackAudit(
        region=region,
        target_lang=track.language,
        source_lang=source.language,
        rows=len(target_df),
    )

    # Pairs collected without sort — both dataframes share row order
    # (same source-of-truth, same processing pipeline). Skipping
    # sort_values saves the doubling copy that OOM'd on large rebuilds.
    pairs: list[tuple[str, str, str]] = []
    seen_hashes: set[str] = set()
    for col in common_text:
        ts = target_df[col].astype("string").fillna("")
        ss = source_df[col].astype("string").fillna("")
        for sv, tv in zip(ss, ts):
            if not sv or not tv:
                continue
            if sv == tv:
                continue
            h = _hash(sv)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            pairs.append((h, sv, tv))
    print(f"  {len(pairs):,} unique translation pairs to inspect")

    layer_script_block(audit, target_df)
    layer_ru_leakage(audit, target_df)
    layer_glossary_consistency(audit, pairs)
    layer_numeric_preservation(audit, pairs)
    layer_length_sanity(audit, pairs)
    if not skip_round_trip:
        _round_trip_on_pairs(
            audit, pairs, sample_size=sample_size, cosine_threshold=0.7,
        )

    # Free the dataframes ASAP — caller might iterate over many regions.
    del target_df, source_df
    import gc
    gc.collect()
    return audit


def write_report(audit: TrackAudit) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"qa_{audit.region}.json"
    # Stratify the sample so every layer is represented (up to 50 per
    # layer). Reading just the first 200 gave only the layer that ran
    # first.
    by_layer: dict[str, list[Flagged]] = {}
    for f in audit.flagged:
        by_layer.setdefault(f.layer, []).append(f)
    sample: list[Flagged] = []
    for layer, flags in by_layer.items():
        sample.extend(flags[:50])
    out.write_text(
        json.dumps({
            "region": audit.region,
            "target_lang": audit.target_lang,
            "source_lang": audit.source_lang,
            "rows": audit.rows,
            "layer_stats": audit.layer_stats,
            "flagged_count": len(audit.flagged),
            "flagged_sample": [asdict(f) for f in sample],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


def write_html_report(audit: TrackAudit) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"qa_{audit.region}.html"
    by_layer: dict[str, list[Flagged]] = {}
    for f in audit.flagged:
        by_layer.setdefault(f.layer, []).append(f)
    rows = []
    for layer, flags in by_layer.items():
        rows.append(f"<h3>{layer} — {len(flags)} flagged</h3>")
        rows.append("<table><tr><th>col</th><th>source</th><th>target</th><th>reason</th></tr>")
        for f in flags[:200]:
            rows.append(
                f"<tr><td>{_html_e(f.column)}</td>"
                f"<td>{_html_e(f.source_value)}</td>"
                f"<td>{_html_e(f.target_value)}</td>"
                f"<td>{_html_e(f.reason)}</td></tr>"
            )
        rows.append("</table>")
    body = "\n".join(rows) or "<p>No flags raised.</p>"
    html = f"""<!doctype html><meta charset="utf-8">
<title>QA report — {audit.region}</title>
<style>
  body{{font-family:sans-serif;max-width:1100px;margin:2em auto;padding:0 1em}}
  table{{border-collapse:collapse;width:100%;margin:1em 0}}
  th,td{{border:1px solid #ddd;padding:6px;font-size:12px;vertical-align:top}}
  th{{background:#f2f2f2}}
  tr:nth-child(even){{background:#fafafa}}
</style>
<h1>QA — {audit.region} ({audit.source_lang} → {audit.target_lang})</h1>
<p>{audit.rows:,} rows, {len(audit.flagged):,} flagged across {len(audit.layer_stats)} layers.</p>
<pre>{json.dumps(audit.layer_stats, indent=2)}</pre>
{body}
"""
    out.write_text(html, encoding="utf-8")
    return out


def _html_e(s: str) -> str:
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


def purge_flagged(audit: TrackAudit) -> int:
    """Remove flagged hashes from the cache so the next run re-translates them."""
    cache = TranslationCache.load(audit.source_lang, audit.target_lang)
    before = len(cache.map)
    flagged_hashes = {f.hash for f in audit.flagged if f.severity == "error"}
    for h in flagged_hashes:
        cache.map.pop(h, None)
    cache.save()
    after = len(cache.map)
    return before - after


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

TARGETS = [
    "BG_SOFIA", "CS_PRAGUE", "HR_ZAGREB", "IT_ROME", "NL_AMSTERDAM",
    "PL_WARSAW", "RO_BUCHAREST", "SV_STOCKHOLM", "TR_ISTANBUL",
    "ID_JAKARTA", "JA_TOKYO", "KO_SEOUL", "TH_BANGKOK", "VI_HANOI",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--track", help="single region, e.g. BG_SOFIA")
    p.add_argument("--all", action="store_true",
                   help="run against all 14 translate-tracks")
    p.add_argument("--sample-size", type=int, default=500,
                   help="round-trip sample size per track")
    p.add_argument("--skip-round-trip", action="store_true")
    p.add_argument("--purge", action="store_true",
                   help="after audit, delete flagged hashes from cache")
    args = p.parse_args()

    regions: list[str] = []
    if args.all:
        regions = TARGETS
    elif args.track:
        regions = [args.track]
    else:
        p.error("specify --track <REGION> or --all")

    summary = []
    overall_failed = False
    for region in regions:
        try:
            audit = audit_track(
                region,
                sample_size=args.sample_size,
                skip_round_trip=args.skip_round_trip,
            )
        except FileNotFoundError as e:
            print(f"  skip {region}: {e}")
            continue
        json_path = write_report(audit)
        html_path = write_html_report(audit)
        purged = 0
        if args.purge:
            purged = purge_flagged(audit)
        # Per-layer summary
        layer_lines = []
        for layer, stat in audit.layer_stats.items():
            pct = (stat["flagged"] / stat["checked"] * 100
                   if stat["checked"] else 0.0)
            layer_lines.append(
                f"    {layer}: {stat['flagged']}/{stat['checked']} ({pct:.2f}%)"
            )
        print(f"\n  {region}: {len(audit.flagged)} flagged across "
              f"{len(audit.layer_stats)} layers")
        for ln in layer_lines:
            print(ln)
        print(f"    json: {json_path}")
        print(f"    html: {html_path}")
        if args.purge:
            print(f"    purged from cache: {purged}")
        summary.append((region, len(audit.flagged), purged))

        # Overall fail only on (a) script_block — that's the hard gate.
        # Other layers are warnings.
        a_stats = audit.layer_stats.get("script_block", {})
        if a_stats.get("flagged", 0) > 0:
            overall_failed = True

    print("\n=== SUMMARY ===")
    for region, n, purged in summary:
        print(f"  {region}: {n} flagged, {purged} purged")
    return 1 if overall_failed else 0


if __name__ == "__main__":
    sys.exit(main())

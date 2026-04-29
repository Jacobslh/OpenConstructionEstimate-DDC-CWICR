"""Force re-translate values where the cache stored target == source verbatim.

These are LLM-verbatim leftovers — gpt-4o-mini conservatively returned the
input string unchanged for items it judged to be codes. We now force a
proper translation with a stronger prompt.

Usage:
    python force_translate_verbatim.py            # all tracks needing it
    python force_translate_verbatim.py ID_JAKARTA # one track
"""
from __future__ import annotations
import sys, os, json, hashlib, re, time
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import env_loader  # noqa
import pandas as pd
from validators import _is_passthrough_value
from text_pipeline import TEXT_COLS

# pair → (source_parquet, target_short, src_lang, tgt_lang, tgt_name)
TRACKS = {
    'ID_JAKARTA':   ('UK_GBP',    'ID', 'en', 'id', 'Indonesian'),
    'VI_HANOI':     ('UK_GBP',    'VI', 'en', 'vi', 'Vietnamese'),
    'JA_TOKYO':     ('UK_GBP',    'JA', 'en', 'ja', 'Japanese'),
    'KO_SEOUL':     ('UK_GBP',    'KO', 'en', 'ko', 'Korean'),
    'TH_BANGKOK':   ('UK_GBP',    'TH', 'en', 'th', 'Thai'),
    'IT_ROME':      ('DE_BERLIN', 'IT', 'de', 'it', 'Italian'),
    'NL_AMSTERDAM': ('DE_BERLIN', 'NL', 'de', 'nl', 'Dutch'),
    'SV_STOCKHOLM': ('DE_BERLIN', 'SV', 'de', 'sv', 'Swedish'),
    'HR_ZAGREB':    ('DE_BERLIN', 'HR', 'de', 'hr', 'Croatian'),
    'CS_PRAGUE':    ('DE_BERLIN', 'CS', 'de', 'cs', 'Czech'),
    'PL_WARSAW':    ('DE_BERLIN', 'PL', 'de', 'pl', 'Polish'),
    'RO_BUCHAREST': ('DE_BERLIN', 'RO', 'de', 'ro', 'Romanian'),
    'TR_ISTANBUL':  ('DE_BERLIN', 'TR', 'de', 'tr', 'Turkish'),
    'BG_SOFIA':     ('DE_BERLIN', 'BG', 'de', 'bg', 'Bulgarian'),
}

REPO = r'C:\Users\Artem Boiko\Desktop\CodeProjects\legal-restructure-2026-04\OpenConstructionEstimate-DDC-CWICR'
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, 'logs', 'force_translate_verbatim.log')
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

API_URL = 'https://api.openai.com/v1/chat/completions'


def strhash(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:16]


def log(msg: str) -> None:
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def parquet_path(region: str) -> str:
    if region == 'UK_GBP':
        return os.path.join(REPO, 'UK___DDC_CWICR',
                            'UK_GBP_workitems_costs_resources_DDC_CWICR.parquet')
    if region == 'DE_BERLIN':
        return os.path.join(REPO, 'DE___DDC_CWICR',
                            'DE_BERLIN_workitems_costs_resources_DDC_CWICR.parquet')
    folder = f'{region.split("_")[0]}___DDC_CWICR'
    return os.path.join(REPO, folder,
                        f'{region}_workitems_costs_resources_DDC_CWICR.parquet')


def find_verbatim_targets(src_path: str, tgt_path: str) -> set[str]:
    """Return values where target == source verbatim and not passthrough."""
    src = pd.read_parquet(src_path)
    tgt = pd.read_parquet(tgt_path)
    sort_keys = ['rate_code', 'resource_code']
    s = src.sort_values(sort_keys).reset_index(drop=True)
    t = tgt.sort_values(sort_keys).reset_index(drop=True)
    flagged: set[str] = set()
    for col in TEXT_COLS:
        if col not in t.columns or col not in s.columns:
            continue
        ts = t[col].astype('string').fillna('')
        ss = s[col].astype('string').fillna('')
        same = (ts == ss) & ts.str.len().gt(0)
        if not same.any():
            continue
        same_vals = ts[same].unique()
        for v in same_vals:
            if not _is_passthrough_value(v):
                flagged.add(v)
    return flagged


def build_prompt(texts: list[str], tgt_name: str) -> tuple[str, str]:
    sys_prompt = (
        f"You are a professional construction-industry translator. The strings "
        f"below are descriptions of construction work, machinery, materials, "
        f"equipment, units of measure, or labels — they are NOT product codes. "
        f"They MUST be translated into {tgt_name}.\n"
        f"Rules:\n"
        f"- Preserve numeric values, ratios, dimensions, and SI units (kg, m, "
        f"mm, m2, m3, kW, V, Hz, etc.).\n"
        f"- 'Nr' is German abbreviation for 'Nummer' (number/each) — translate "
        f"to the {tgt_name} equivalent for piece/each count.\n"
        f"- Words like 'terminal', 'normal', 'Stand:', 'Press:', 'Unit:', "
        f"'Stacker' must be translated into {tgt_name} (not kept as English).\n"
        f"- Numbered loanwords like '1680 pcs.' translate as '1680 <word for "
        f"piece in {tgt_name}>'.\n"
        f"- Composite unit lists like 'kg, t' or 'set, Nr' — translate each "
        f"word, keep commas and SI units verbatim.\n"
        f"- NEVER return the input unchanged unless it is purely a code or "
        f"dimensionless number with no descriptive language.\n"
        f"- Respond as a numbered list, one translation per input, no preamble, "
        f"no commentary."
    )
    user_prompt = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    return sys_prompt, user_prompt


def call_openai(sys_prompt: str, user_prompt: str, *,
                api_key: str, model: str = 'gpt-4o-mini',
                timeout: float = 45.0) -> str:
    body = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        'max_tokens': 4096,
        'temperature': 0.0,
    }
    req = urllib.request.Request(
        API_URL,
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        },
        data=json.dumps(body).encode('utf-8'),
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        d = json.loads(resp.read().decode('utf-8'))
    return d['choices'][0]['message']['content'] or ''


def parse_numbered(raw: str, n: int, fallback: list[str]) -> list[str]:
    out = [''] * n
    for ln in raw.splitlines():
        body = ln.lstrip()
        if not body:
            continue
        for sep in ('.', ')'):
            if sep in body[:6]:
                head, _, rest = body.partition(sep)
                if head.strip().isdigit():
                    idx = int(head.strip()) - 1
                    if 0 <= idx < n:
                        out[idx] = rest.strip()
                        break
    for i, p in enumerate(out):
        if not p:
            out[i] = fallback[i]
    return out


def process_track(region: str, api_key: str) -> int:
    src_region, _tgt_short, src_lang, tgt_lang, tgt_name = TRACKS[region]
    pair = f'{src_lang}_{tgt_lang}'
    cache_fp = os.path.join(HERE, 'glossary', 'translations', f'{pair}.json')
    src_path = parquet_path(src_region)
    tgt_path = parquet_path(region)

    if not os.path.exists(tgt_path):
        log(f'{region}: target parquet missing, skip')
        return 0

    log(f'{region}: scanning verbatim leftovers...')
    flagged = find_verbatim_targets(src_path, tgt_path)
    log(f'{region}: {len(flagged)} unique verbatim values to re-translate')
    if not flagged:
        return 0

    cache = json.load(open(cache_fp, encoding='utf-8'))
    m = cache.setdefault('map', {})

    BATCH = 20
    CONCURRENCY = int(os.environ.get('LLM_CONCURRENCY', 30))
    chunks = list(flagged)
    chunks = [chunks[i:i+BATCH] for i in range(0, len(chunks), BATCH)]

    def do_chunk(chunk: list[str]) -> tuple[list[str], list[str]]:
        sys_p, user_p = build_prompt(chunk, tgt_name)
        try:
            raw = call_openai(sys_p, user_p, api_key=api_key)
        except Exception as e:
            return chunk, [f'__ERR__:{type(e).__name__}'] * len(chunk)
        return chunk, parse_numbered(raw, len(chunk), chunk)

    n_done = n_err = n_unchanged = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = [ex.submit(do_chunk, c) for c in chunks]
        for k, fut in enumerate(as_completed(futures)):
            chunk, outs = fut.result()
            if outs and outs[0].startswith('__ERR__'):
                n_err += 1
                continue
            for src, tgt in zip(chunk, outs):
                if tgt == src:
                    n_unchanged += 1
                m[strhash(src)] = tgt
            n_done += len(chunk)
            if (k + 1) % 50 == 0:
                tmp = cache_fp + '.tmp'
                json.dump(cache, open(tmp, 'w', encoding='utf-8'),
                          ensure_ascii=False, indent=2)
                os.replace(tmp, cache_fp)

    tmp = cache_fp + '.tmp'
    json.dump(cache, open(tmp, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    os.replace(tmp, cache_fp)
    log(f'{region}: done ({n_done}/{len(flagged)}, errors={n_err}, '
        f'still-verbatim={n_unchanged}, {time.time()-t0:.0f}s)')
    return n_done


def main() -> int:
    api_key = os.environ['OPENAI_API_KEY']
    regions = sys.argv[1:] or list(TRACKS)
    grand = 0
    for r in regions:
        if r not in TRACKS:
            log(f'unknown region {r}; skip')
            continue
        grand += process_track(r, api_key)
    log(f'TOTAL: {grand} entries re-translated')
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""Direct re-translate of purged verbatim entries via gpt-4o-mini.

Uses raw urllib (not OpenAI SDK — known Windows-httpx hang issue) with
ThreadPoolExecutor for concurrent API calls (default 20 workers).

Saves progress to logs/retranslate_purged.log for monitoring.
"""
from __future__ import annotations
import sys, os, json, hashlib, re, time
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import env_loader  # noqa
import pandas as pd
from text_pipeline import TEXT_COLS

DE_PARQUET = r'C:\Users\Artem Boiko\Desktop\CodeProjects\legal-restructure-2026-04\OpenConstructionEstimate-DDC-CWICR\DE___DDC_CWICR\DE_BERLIN_workitems_costs_resources_DDC_CWICR.parquet'
EN_PARQUET = r'C:\Users\Artem Boiko\Desktop\CodeProjects\legal-restructure-2026-04\OpenConstructionEstimate-DDC-CWICR\EN___DDC_CWICR\ENG_TORONTO_workitems_costs_resources_DDC_CWICR.parquet'

DE_MARKERS = re.compile(
    r'[äöüÄÖÜß]|'
    r'\b(?:der|die|das|den|dem|des|ein|mit|von|für|zur|zum|aus|auf|über|unter|durch|als|und|oder)\b|'
    r'\b\w+(?:ung|heit|keit|nis|chaft|tum|lich|isch|enn|tech|maschine|gerät|werkzeug|werk)\b',
    re.UNICODE)
EN_MARKERS = re.compile(
    r'\b(?:the|of|with|and|or|for|by|from|to|in|on|at|into|using|over|under)\b|'
    r'\b\w+(?:ing|tion|ment|ness|able|less|ful|ish)\b|'
    r'\b(?:plate|sheet|tube|pipe|valve|pump|motor|tank|wire|cable|mesh|drill|machine|installation)\b',
    re.IGNORECASE)

LANG_NAMES = {
    'it':'Italian','ro':'Romanian','bg':'Bulgarian','hr':'Croatian','cs':'Czech',
    'pl':'Polish','nl':'Dutch','sv':'Swedish','tr':'Turkish',
    'id':'Indonesian','ja':'Japanese','ko':'Korean','th':'Thai','vi':'Vietnamese',
    'de':'German','en':'English',
}

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'glossary', 'translations')
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'logs', 'retranslate_purged.log')
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

DE_PAIRS = ['de_it','de_ro','de_bg','de_hr','de_cs','de_pl','de_nl','de_sv','de_tr']
EN_PAIRS = ['en_id','en_ja','en_ko','en_th','en_vi']
API_URL = 'https://api.openai.com/v1/chat/completions'


def strhash(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:16]


def get_uniques(parquet_path: str) -> set[str]:
    df = pd.read_parquet(parquet_path)
    s: set[str] = set()
    for c in TEXT_COLS:
        if c in df.columns:
            for v in df[c].dropna().astype(str).unique():
                if v: s.add(v)
    return s


def log(msg: str) -> None:
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def build_prompt(texts: list[str], src_lang: str, tgt_lang: str) -> tuple[str, str]:
    src_name = LANG_NAMES[src_lang]
    tgt_name = LANG_NAMES[tgt_lang]
    sys_prompt = (
        f"You are a professional construction-industry translator. Translate "
        f"every input string from {src_name} into {tgt_name}. The strings are "
        f"descriptions of construction work, machinery, materials, and "
        f"equipment — they ARE NOT product codes. They MUST be translated.\n"
        f"Rules:\n"
        f"- Preserve numeric values, units (kg, m, mm, kW, etc.), and "
        f"identifier codes that contain digits (e.g. 'DN-200', 'KS-069-1', "
        f"'EP-0199', 'Typ MM-01-1').\n"
        f"- Translate ALL natural-language words including technical names "
        f"and equipment terms into {tgt_name}.\n"
        f"- NEVER return the input unchanged unless it is purely a product "
        f"code with no surrounding descriptive language.\n"
        f"- Respond as a numbered list, one translation per input, no "
        f"preamble, no commentary."
    )
    user_prompt = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    return sys_prompt, user_prompt


def call_openai(sys_prompt: str, user_prompt: str, *,
                api_key: str, model: str = 'gpt-4o-mini',
                timeout: float = 30.0) -> str:
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


def main() -> int:
    log('Loading source parquets...')
    de_uniq = get_uniques(DE_PARQUET)
    en_uniq = get_uniques(EN_PARQUET)
    log(f'DE source: {len(de_uniq)} unique strings')
    log(f'EN source: {len(en_uniq)} unique strings')

    pairs = ([(p, de_uniq, DE_MARKERS, 'de') for p in DE_PAIRS]
             + [(p, en_uniq, EN_MARKERS, 'en') for p in EN_PAIRS])

    api_key = os.environ['OPENAI_API_KEY']
    BATCH = 20
    CONCURRENCY = int(os.environ.get('LLM_CONCURRENCY', 20))
    t0 = time.time()
    grand_total = 0

    def translate_one_batch(chunk: list[str], src_lang: str, tgt_lang: str
                            ) -> tuple[list[str], list[str]]:
        sys_p, user_p = build_prompt(chunk, src_lang, tgt_lang)
        try:
            raw = call_openai(sys_p, user_p, api_key=api_key, timeout=45)
        except Exception as e:
            return chunk, [f'__ERR__:{type(e).__name__}'] * len(chunk)
        outs = parse_numbered(raw, len(chunk), chunk)
        return chunk, outs

    for pair, uniq, marker, src_lang in pairs:
        fp = os.path.join(BASE_DIR, f'{pair}.json')
        d = json.load(open(fp, encoding='utf-8'))
        m = d.setdefault('map', {})
        missing = [s for s in uniq
                   if strhash(s) not in m and marker.search(s)]
        log(f'{pair}: {len(missing)} missing entries (concurrency={CONCURRENCY})')
        if not missing:
            continue
        tgt_lang = pair.split('_')[1]
        chunks = [missing[i:i+BATCH] for i in range(0, len(missing), BATCH)]
        n_done = 0
        n_err = 0
        n_save_every = 50  # save cache every 50 batches
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            futures = [ex.submit(translate_one_batch, c, src_lang, tgt_lang)
                       for c in chunks]
            for k, fut in enumerate(as_completed(futures)):
                chunk, outs = fut.result()
                if outs and outs[0].startswith('__ERR__'):
                    n_err += 1
                    if n_err <= 3:
                        log(f'  {pair} batch err: {outs[0]}')
                    continue
                for src, tgt in zip(chunk, outs):
                    m[strhash(src)] = tgt
                n_done += len(chunk)
                if (k + 1) % n_save_every == 0:
                    tmp = fp + '.tmp'
                    json.dump(d, open(tmp, 'w', encoding='utf-8'),
                              ensure_ascii=False, indent=2)
                    os.replace(tmp, fp)
                    log(f'  {pair}: progress {n_done}/{len(missing)} '
                        f'({time.time()-t0:.0f}s, err={n_err})')

        # Final save
        tmp = fp + '.tmp'
        json.dump(d, open(tmp, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
        os.replace(tmp, fp)
        grand_total += n_done
        log(f'{pair}: done ({n_done}/{len(missing)}, errors={n_err}, '
            f'total {grand_total}, {time.time()-t0:.0f}s)')

    log(f'TOTAL: {grand_total} entries in {time.time()-t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())

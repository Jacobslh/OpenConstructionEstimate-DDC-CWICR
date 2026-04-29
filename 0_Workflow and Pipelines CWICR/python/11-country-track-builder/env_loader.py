"""Auto-load .env into os.environ on import.

Saves users from having to export ANTHROPIC_API_KEY / OPENAI_API_KEY
on every shell. Searches a few likely places, plain KEY=value parser
so we avoid the python-dotenv hard dependency.

Import this once near the top of any script that needs API keys.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_env() -> None:
    here = Path(__file__).resolve().parent
    candidates = [
        here / ".env",
        here.parent / ".env",
        here.parent.parent / ".env",
        here.parent.parent.parent / ".env",
        here.parent.parent.parent.parent / ".env",
    ]
    for env_path in candidates:
        if not env_path.exists():
            continue
        for ln in env_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:
                continue
            k, _, v = ln.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v
        break


load_env()

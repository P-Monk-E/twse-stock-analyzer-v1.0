# /mnt/data/names_store.py
from __future__ import annotations
import json, os
from typing import Dict, Optional

NAMES_PATH = "names.json"

def _load() -> Dict[str, str]:
    if not os.path.exists(NAMES_PATH):
        return {}
    try:
        with open(NAMES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return {str(k).upper(): str(v) for k, v in data.items() if v}
    except Exception:
        pass
    return {}

def _save(data: Dict[str, str]) -> None:
    with open(NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get(symbol: str, fallback: Optional[str] = None) -> Optional[str]:
    if not symbol:
        return fallback
    data = _load()
    return data.get(str(symbol).upper(), fallback)

def set(symbol: str, name: str) -> None:
    sym = str(symbol).upper().strip()
    nm = str(name).strip()
    if not sym or not nm:
        return
    data = _load()
    data[sym] = nm
    _save(data)

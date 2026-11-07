"""Универсальный извлекатель метрик из benchmark JSON.

Поддерживает:
  - реальную схему benchmark_diarization.py:
        {timestamp, audio_dir, configs, summaries: [{mean_der, mean_rtf, n_cases, ...}]}
  - плоскую схему {mean_der, mean_rtf, n_scored}
  - list[dict] (по конфигу на элемент)
DER нормализуется в ПРОЦЕНТЫ, если в JSON он в долях (<1).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_DER_KEYS = ("mean_der", "der", "der_pct", "mean_DER", "DER")
_RTF_KEYS = ("mean_rtf", "rtf", "rtf_mean", "RTF")
_N_KEYS   = ("n_scored", "n_cases", "n_files", "scored", "n", "count")


def _pick(d: dict, keys: tuple) -> Any:
    for k in keys:
        if k in d and isinstance(d[k], (int, float)):
            return d[k]
    return None


def _walk(d: Any):
    if isinstance(d, dict):
        yield d
        for v in d.values():
            yield from _walk(v)
    elif isinstance(d, list):
        for v in d:
            yield from _walk(v)


def load_metrics(path: str | Path) -> dict:
    raw = json.load(open(path))

    # 1) новая схема: {summaries: [...]}
    if isinstance(raw, dict) and "summaries" in raw and raw["summaries"]:
        s = raw["summaries"][0]
        der = _pick(s, _DER_KEYS)
        rtf = _pick(s, _RTF_KEYS)
        n   = _pick(s, _N_KEYS)
        if der is not None:
            der_pct = der * 100 if der <= 1.0 else der
            return {
                "config_name": s.get("config_name"),
                "mean_der_pct": der_pct,
                "median_der_pct": (s.get("median_der") or 0) * 100,
                "mean_rtf": rtf,
                "max_rtf": s.get("max_rtf"),
                "n_scored": n,
                "mean_speakers_error": s.get("mean_speakers_error"),
                "raw": s,
            }

    # 2) список конфигов
    if isinstance(raw, list):
        for c in raw:
            if isinstance(c, dict):
                der = _pick(c, _DER_KEYS)
                if der is not None:
                    der_pct = der * 100 if der <= 1.0 else der
                    return {
                        "config_name": c.get("config_name"),
                        "mean_der_pct": der_pct,
                        "mean_rtf": _pick(c, _RTF_KEYS),
                        "n_scored": _pick(c, _N_KEYS),
                        "raw": c,
                    }

    # 3) плоский
    der = _pick(raw, _DER_KEYS)
    if der is not None:
        return {
            "mean_der_pct": der * 100 if der <= 1.0 else der,
            "mean_rtf": _pick(raw, _RTF_KEYS),
            "n_scored": _pick(raw, _N_KEYS),
            "raw": raw,
        }

    # 4) последняя попытка: обойти дерево
    for node in _walk(raw):
        der = _pick(node, _DER_KEYS)
        if der is not None:
            return {
                "mean_der_pct": der * 100 if der <= 1.0 else der,
                "mean_rtf": _pick(node, _RTF_KEYS),
                "n_scored": _pick(node, _N_KEYS),
                "raw": node,
            }

    raise ValueError(f"no metrics found in {path}: top keys={list(raw)[:10]}")


def latest_result(glob_pat: str) -> Path:
    """Найти самый свежий benchmark-*.json или *.result.json."""
    import glob as _g
    cands = sorted(_g.glob(glob_pat), key=lambda p: Path(p).stat().st_mtime)
    if not cands:
        raise FileNotFoundError(glob_pat)
    return Path(cands[-1])


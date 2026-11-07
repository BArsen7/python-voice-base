#!/usr/bin/env python3
"""
grid_search.py — последовательный (staged) grid search оптимального конфига.

Стратегия: coordinate descent по группам параметров.
  1. Оптимизируем стадию 1 (например, clustering) → берём лучший конфиг.
  2. На нём оптимизируем стадию 2 (embeddings) → обновляем конфиг.
  3. ... и т.д.
  4. Если --repeat > 1, проходим все стадии ещё раз с новым конфигом.

Использование:
    python scripts/grid_search.py \
        --audio-dir data/short \
        --base-config configs/baseline.yaml \
        --search-space configs/search_space.yaml \
        --output grid_results \
        --limit 10 \
        --objective der_rtf --rtf-weight 0.1 \
        --repeat 2
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger("grid")


# ---------------------------------------------------------------------------
# Кеш моделей — критично для скорости grid search
# ---------------------------------------------------------------------------

def install_model_caches() -> None:
    """Сделать загрузку тяжёлых моделей идемпотентной между pipeline'ами.

    Без этого каждая DiarizationPipeline заново грузит Whisper (~10–30 с),
    WeSpeaker (~2–5 с), Silero (~1 с). При 100 прогонах это часы впустую.
    """
    try:
        from scripts.models import asr as asr_mod
        from scripts.models import embeddings as emb_mod
        from scripts.models import overlap as overlap_mod
        from scripts.models import vad as vad_mod
    except ImportError as e:
        logger.warning("install_model_caches: %s (кэш не установлен)", e)
        return

    def wrap(cls, attr_name: str, key_fn) -> None:
        if not hasattr(cls, "_lazy_load"):
            return
        orig = cls._lazy_load
        cache: Dict[str, Any] = {}

        def patched(self):
            if getattr(self, attr_name, None) is not None:
                return
            key = key_fn(self)
            if key is None:
                return orig(self)
            if key in cache:
                setattr(self, attr_name, cache[key])
                return
            orig(self)
            cache[key] = getattr(self, attr_name, None)

        cls._lazy_load = patched

    wrap(emb_mod.WeSpeakerEmbeddingExtractor, "_model", lambda s: "wespeaker")
    wrap(emb_mod.ECAPAEmbeddingExtractor,     "_model", lambda s: "ecapa")
    wrap(asr_mod.FasterWhisperASR,            "_model",
         lambda s: f"fwhisper:{s.model_size}:{s.device}")
    wrap(asr_mod.SherpaOnnxASR,               "_recognizer",
         lambda s: f"sherpa:{s.model_dir or 'explicit'}")
    wrap(vad_mod.SileroVAD,                   "_vad", lambda s: "silero")
    wrap(overlap_mod.PyannoteOverlapDetector, "_pipeline", lambda s: "pyannote_ovl")

    logger.info("Model caches installed (heavy models reload only once)")


# ---------------------------------------------------------------------------
# Работа с вложенным YAML
# ---------------------------------------------------------------------------

def set_nested(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def get_nested(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    keys = dotted.split(".")
    d: Any = cfg
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


# ---------------------------------------------------------------------------
# Скоринг и агрегация
# ---------------------------------------------------------------------------

def summarize_cases(cases: List[Any]):
    """Агрегировать CaseResult одного конфига в ConfigSummary."""
    from benchmark_diarization import ConfigSummary

    ok = [c for c in cases if c.error is None and c.der is not None]
    ders = [c.der for c in ok]
    rtfs = [c.rtf for c in ok]
    spk = [
        abs(c.n_speakers_pred - c.n_speakers_ref)
        for c in ok
        if c.n_speakers_pred is not None and c.n_speakers_ref is not None
    ]
    return ConfigSummary(
        config_name="current",
        n_cases=len(cases),
        mean_der=statistics.mean(ders) if ders else None,
        median_der=statistics.median(ders) if ders else None,
        mean_rtf=statistics.mean(rtfs) if rtfs else None,
        max_rtf=max(rtfs) if rtfs else None,
        mean_speakers_error=statistics.mean(spk) if spk else None,
        cases=ok,
    )


def score_summary(s, objective: str, rtf_weight: float) -> float:
    der = s.mean_der if s.mean_der is not None else 1.0
    rtf = s.mean_rtf if s.mean_rtf is not None else 10.0
    spk = s.mean_speakers_error if s.mean_speakers_error is not None else 10.0
    if objective == "der":
        return der
    if objective == "der_rtf":
        return der + rtf_weight * rtf
    if objective == "der_spk":
        return der + 0.05 * spk
    if objective == "combined":
        return der + rtf_weight * rtf + 0.05 * spk
    raise ValueError(f"Unknown objective: {objective}")


def run_one_config(cfg: Dict[str, Any], cases, label: str = "current"):
    from benchmark_diarization import run_single_case

    results = [run_single_case(wav, rttm, cfg, label) for wav, rttm in cases]
    return summarize_cases(results)


# ---------------------------------------------------------------------------
# Одна стадия: grid по одной группе параметров
# ---------------------------------------------------------------------------

def run_stage(
    stage_name: str,
    base_cfg: Dict[str, Any],
    param_space: Dict[str, List[Any]],
    cases,
    objective: str,
    rtf_weight: float,
) -> Tuple[Dict[str, Any], Any, List[Tuple[float, Tuple, Any]]]:
    keys = list(param_space.keys())
    values = [param_space[k] for k in keys]
    combos = list(itertools.product(*values))

    if "clustering.n_clusters" in keys and any(
        k in keys for k in ("clustering.threshold", "clustering.max_speakers")
    ):
        logger.warning(
            "Stage '%s': n_clusters в одной стадии с threshold/max_speakers — "
            "n_clusters перезаписывает остальные, комбинации тратятся впустую. "
            "Разнесите по разным стадиям.",
            stage_name,
        )

    logger.info("=" * 72)
    logger.info(
        "Stage '%s': %d combos × %d cases", stage_name, len(combos), len(cases)
    )
    logger.info("Params: %s", keys)

    all_summaries: List[Tuple[float, Tuple, Any]] = []
    best_score = float("inf")
    best_cfg: Optional[Dict[str, Any]] = None
    best_summary = None

    for i, combo in enumerate(combos, 1):
        cfg = copy.deepcopy(base_cfg)
        for k, v in zip(keys, combo):
            set_nested(cfg, k, v)

        label = ", ".join(f"{k}={v}" for k, v in zip(keys, combo))
        t0 = time.perf_counter()
        try:
            summary = run_one_config(cfg, cases, label=stage_name)
        except Exception as e:
            logger.warning("  [%d/%d] %s → FAILED: %s", i, len(combos), label, e)
            continue
        elapsed = time.perf_counter() - t0

        s = score_summary(summary, objective, rtf_weight)
        all_summaries.append((s, combo, summary))

        marker = ""
        if s < best_score:
            best_score = s
            best_cfg = cfg
            best_summary = summary
            marker = "  ← best"

        logger.info(
            "  [%d/%d] %-55s DER=%.2f%% RTF=%.3f Δspk=%.1f score=%.4f (%.1fs)%s",
            i, len(combos), label,
            (summary.mean_der or 0) * 100,
            summary.mean_rtf or 0,
            summary.mean_speakers_error or 0,
            s, elapsed, marker,
        )

    # Top-3
    if all_summaries:
        logger.info("Top-3 for stage '%s':", stage_name)
        for s, combo, summary in sorted(all_summaries, key=lambda x: x[0])[:3]:
            logger.info(
                "  score=%.4f  DER=%.2f%%  RTF=%.3f  %s",
                s,
                (summary.mean_der or 0) * 100,
                summary.mean_rtf or 0,
                dict(zip(keys, combo)),
            )

    if best_cfg is None:
        logger.error("Stage '%s' produced no valid results", stage_name)
        return base_cfg, None, all_summaries

    logger.info(
        "Stage '%s' WINNER: DER=%.2f%%  RTF=%.3f  Δspk=%.1f  score=%.4f",
        stage_name,
        (best_summary.mean_der or 0) * 100,
        best_summary.mean_rtf or 0,
        best_summary.mean_speakers_error or 0,
        best_score,
    )
    return best_cfg, best_summary, all_summaries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--search-space", required=True)
    ap.add_argument("--output", default="grid_results")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--objective", default="der",
                    choices=["der", "der_rtf", "der_spk", "combined"])
    ap.add_argument("--rtf-weight", type=float, default=0.1)
    ap.add_argument("--repeat", type=int, default=1,
                    help="Сколько раз пройти все стадии (coordinate descent)")
    ap.add_argument("--no-cache", action="store_true",
                    help="Отключить кэш моделей (только для отладки)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.no_cache:
        install_model_caches()

    from benchmark_diarization import discover_cases

    cases = discover_cases(Path(args.audio_dir), limit=args.limit)
    logger.info("Cases: %d", len(cases))
    if not cases:
        raise SystemExit("Нет кейсов — проверь --audio-dir")

    with open(args.base_config, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    with open(args.search_space, encoding="utf-8") as f:
        space = yaml.safe_load(f)
    stages = space["stages"]

    # Оценка времени
    est_runs = sum(
        int(np.prod([len(v) for v in st["params"].values()])) for st in stages
    ) * len(cases) * args.repeat
    logger.info("Estimated runs: %d", est_runs)
    if est_runs > 200:
        logger.warning(
            "Много прогонов (%d). Рассмотрите --limit или --repeat 1.", est_runs
        )

    current_cfg = copy.deepcopy(base_cfg)
    final_summary = None
    history: List[Dict[str, Any]] = []
    total_start = time.perf_counter()

    for iteration in range(args.repeat):
        if args.repeat > 1:
            logger.info(
                "\n%s\nIteration %d / %d\n%s",
                "=" * 72, iteration + 1, args.repeat, "=" * 72,
            )
        for stage in stages:
            name = stage["name"]
            params = stage["params"]
            best_cfg, best_sum, _ = run_stage(
                name, current_cfg, params, cases, args.objective, args.rtf_weight,
            )
            if best_cfg is not None and best_sum is not None:
                current_cfg = best_cfg
                final_summary = best_sum
                history.append({
                    "iteration": iteration,
                    "stage": name,
                    "score": score_summary(best_sum, args.objective, args.rtf_weight),
                    "der": best_sum.mean_der,
                    "rtf": best_sum.mean_rtf,
                    "spk_err": best_sum.mean_speakers_error,
                    "config": copy.deepcopy(current_cfg),
                })

    total_s = time.perf_counter() - total_start

    # Сохранение
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    best_path = out / f"best_config-{ts}.yaml"
    with open(best_path, "w", encoding="utf-8") as f:
        yaml.dump(current_cfg, f, allow_unicode=True, sort_keys=False)

    hist_path = out / f"history-{ts}.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "objective": args.objective,
                "rtf_weight": args.rtf_weight,
                "total_seconds": total_s,
                "history": history,
            },
            f, ensure_ascii=False, indent=2, default=str,
        )

    # Итог
    print("\n" + "=" * 72)
    print("GRID SEARCH COMPLETE")
    print("=" * 72)
    if final_summary:
        print(f"Objective:   {args.objective}")
        print(f"Best DER:    {(final_summary.mean_der or 0) * 100:.2f}%")
        print(f"Best RTF:    {(final_summary.mean_rtf or 0):.3f}")
        print(f"Δspk:        {(final_summary.mean_speakers_error or 0):.1f}")
    print(f"Total time:  {total_s / 60:.1f} min")
    print(f"Best config: {best_path}")
    print(f"History:     {hist_path}")
    print("\nИтоговый config:")
    print(yaml.dump(current_cfg, allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()

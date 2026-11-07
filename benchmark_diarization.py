#!/usr/bin/env python3
"""
benchmark_diarization.py — автоматический бенчмарк пайплайна диаризации.

Что делает:
  1. Читает config.yaml и набор WAV + RTTM из --audio-dir.
  2. Для каждой конфигурации моделей из --configs прогоняет DiarizationPipeline.
  3. Считает DER (pyannote.metrics) против RTTM.
  4. Замеряет wall-time и RTF.
  5. Пишет benchmark_results/<timestamp>.json и печатает сводную таблицу.

Ключевые флаги:
  --min-ref-ratio 0.6   — только кейсы, где ref_speech/audio >= 0.6
                          (отсекает кейсы с неполной разметкой → честный DER)
  --uem-from-ref        — UEM = объединение ref-сегментов.
                          Стандартная практика: неаннотированные участки
                          не считаются false alarm.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("benchmark")


# ---------------------------------------------------------------------------
# Структуры
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    audio_path: str
    audio_duration_s: float
    config_name: str
    der: Optional[float] = None
    jer: Optional[float] = None
    wall_time_s: float = 0.0
    rtf: float = 0.0
    n_speakers_ref: Optional[int] = None
    n_speakers_pred: Optional[int] = None
    error: Optional[str] = None
    der_breakdown: Optional[Dict[str, float]] = None


@dataclass
class ConfigSummary:
    config_name: str
    n_cases: int
    mean_der: Optional[float] = None
    median_der: Optional[float] = None
    mean_rtf: Optional[float] = None
    max_rtf: Optional[float] = None
    mean_speakers_error: Optional[float] = None
    cases: List[CaseResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Утилиты (после import — важно для from __future__)
# ---------------------------------------------------------------------------

def _rttm_total_seconds(rttm_path: Path) -> float:
    """Суммарная длительность речи в RTTM (секунды)."""
    total = 0.0
    with open(rttm_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9 or parts[0] != "SPEAKER":
                continue
            total += float(parts[4])
    return total


def load_rttm_as_annotation(rttm_path: Path):
    from pyannote.core import Annotation, Segment
    ann = Annotation(uri=rttm_path.stem)
    with open(rttm_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9 or parts[0] != "SPEAKER":
                continue
            tbeg = float(parts[3])
            tdur = float(parts[4])
            ann[Segment(tbeg, tbeg + tdur)] = parts[7]
    return ann


def entries_to_annotation(entries: Sequence[Any], uri: str = "hypothesis"):
    from pyannote.core import Annotation, Segment
    ann = Annotation(uri=uri)
    for e in entries:
        start = e.start_ms / 1000.0
        end = e.end_ms / 1000.0
        if end <= start:
            continue
        ann[Segment(start, end)] = e.speaker_id
    return ann


# ---------------------------------------------------------------------------
# Прогон одного кейса
# ---------------------------------------------------------------------------

def run_single_case(
    audio_path: Path,
    rttm_path: Path,
    config: Dict[str, Any],
    config_name: str,
    min_ref_ratio: float = 0.0,
    uem_from_ref: bool = False,
) -> CaseResult:
    # Импорт внутри функции — чтобы grid_search мог подменять модули
    from scripts.pipeline.orchestrator import DiarizationPipeline, load_wav

    waveform, sr = load_wav(
        audio_path,
        target_sr=int(config.get("pipeline", {}).get("sample_rate", 16000)),
    )
    duration_s = len(waveform) / sr

    # --- Фильтр min_ref_ratio ---
    if min_ref_ratio > 0:
        ref_total_pre = _rttm_total_seconds(rttm_path)
        ratio = ref_total_pre / max(duration_s, 1e-9)
        if ratio < min_ref_ratio:
            return CaseResult(
                audio_path=str(audio_path),
                audio_duration_s=duration_s,
                config_name=config_name,
                error=f"SKIP: ref/audio={ratio:.2f} < {min_ref_ratio:.2f}",
            )

    pipe = DiarizationPipeline(config)

    t0 = time.perf_counter()
    try:
        entries = pipe.run(waveform, sr)
    except Exception as e:
        logger.exception("Pipeline failed on %s", audio_path)
        return CaseResult(
            audio_path=str(audio_path),
            audio_duration_s=duration_s,
            config_name=config_name,
            error=f"{type(e).__name__}: {e}",
        )
    wall_time = time.perf_counter() - t0

    result = CaseResult(
        audio_path=str(audio_path),
        audio_duration_s=duration_s,
        config_name=config_name,
        wall_time_s=wall_time,
        rtf=wall_time / duration_s if duration_s > 0 else 0.0,
    )

    try:
        from pyannote.core import Segment
        from pyannote.metrics.diarization import DiarizationErrorRate

        ref = load_rttm_as_annotation(rttm_path)
        hyp = entries_to_annotation(entries, uri=audio_path.stem)

        # UEM
        if uem_from_ref:
            uem = ref.get_timeline().support()
        else:
            uem = Segment(0.0, duration_s)

        metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
        der = metric(ref, hyp, uem=uem, detailed=False)
        breakdown = metric(ref, hyp, uem=uem, detailed=True)

        try:
            from pyannote.metrics.diarization import JaccardErrorRate
            jer_metric = JaccardErrorRate()
            jer = jer_metric(ref, hyp)
        except Exception:
            jer = None

        # Нормализуем breakdown к ref_total, чтобы "%" были осмысленными
        ref_total = _rttm_total_seconds(rttm_path)
        norm: Dict[str, float] = {}
        for k, v in breakdown.items():
            if isinstance(v, (int, float)):
                norm[k] = float(v) / max(ref_total, 1e-9)
        result.der_breakdown = norm

        result.der = float(der)
        result.jer = float(jer) if jer is not None else None
        result.n_speakers_ref = len(ref.labels())
        result.n_speakers_pred = len(hyp.labels())
    except Exception as e:
        logger.warning("DER computation failed for %s: %s", audio_path, e)
        result.error = f"DER: {type(e).__name__}: {e}"

    return result


# ---------------------------------------------------------------------------
# Обход кейсов
# ---------------------------------------------------------------------------

def discover_cases(
    audio_dir: Path, limit: Optional[int] = None
) -> List[Tuple[Path, Path]]:
    wavs = sorted(audio_dir.glob("*.wav"))
    pairs = []
    for wav in wavs:
        rttm = wav.with_suffix(".rttm")
        if not rttm.exists():
            logger.warning("Пропускаю %s — нет %s", wav.name, rttm.name)
            continue
        pairs.append((wav, rttm))
        if limit and len(pairs) >= limit:
            break
    if not pairs:
        raise SystemExit(f"В {audio_dir} не найдено пар *.wav + *.rttm")
    return pairs


def summarize(results: List[CaseResult]) -> List[ConfigSummary]:
    by_cfg: Dict[str, List[CaseResult]] = {}
    for r in results:
        by_cfg.setdefault(r.config_name, []).append(r)

    summaries = []
    for name, cases in by_cfg.items():
        ok = [c for c in cases if c.error is None and c.der is not None]
        ders = [c.der for c in ok if c.der is not None]
        rtfs = [c.rtf for c in ok]
        spk_err = [
            abs(c.n_speakers_pred - c.n_speakers_ref)
            for c in ok
            if c.n_speakers_pred is not None and c.n_speakers_ref is not None
        ]
        summaries.append(ConfigSummary(
            config_name=name,
            n_cases=len(cases),
            mean_der=statistics.mean(ders) if ders else None,
            median_der=statistics.median(ders) if ders else None,
            mean_rtf=statistics.mean(rtfs) if rtfs else None,
            max_rtf=max(rtfs) if rtfs else None,
            mean_speakers_error=statistics.mean(spk_err) if spk_err else None,
            cases=cases,
        ))
    return summaries


def print_summary_table(summaries: List[ConfigSummary]) -> None:
    header = (
        f"{'config':<24} {'N':>3} {'mean DER':>10} {'median DER':>11} "
        f"{'mean RTF':>9} {'max RTF':>8} {'Δspk':>5}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for s in summaries:
        def fmt(v, p=1):
            return f"{v:.{p}f}" if v is not None else "—"
        print(
            f"{s.config_name:<24} {s.n_cases:>3} "
            f"{fmt(s.mean_der * 100 if s.mean_der is not None else None, 2):>10} "
            f"{fmt(s.median_der * 100 if s.median_der is not None else None, 2):>11} "
            f"{fmt(s.mean_rtf, 3):>9} {fmt(s.max_rtf, 3):>8} "
            f"{fmt(s.mean_speakers_error, 1):>5}"
        )
    print("=" * len(header))
    print("DER в %, RTF = processing_time / audio_duration (меньше = быстрее).")
    print("Δspk = средняя |pred_speakers - ref_speakers|.")


def print_per_case_breakdown(payload: Dict[str, Any]) -> None:
    """Per-case разложение DER (miss / FA / confusion), нормированное к ref."""
    print("\nPer-case DER breakdown (% of ref speech):")
    print(f"{'config':<18} {'case':<22} {'DER':>7} {'miss':>7} {'FA':>7} {'conf':>7}")
    print("-" * 80)
    for s in payload["summaries"]:
        for c in s["cases"]:
            b = c.get("der_breakdown") or {}
            err = c.get("error") or ""
            fname = c["audio_path"].split("/")[-1]
            if err.startswith("SKIP"):
                print(f"{s['config_name']:<18} {fname:<22} {err}")
                continue
            if err:
                print(f"{s['config_name']:<18} {fname:<22} ERR: {err[:50]}")
                continue

            def g(k):
                return f"{b.get(k, 0) * 100:>6.2f}%"

            print(
                f"{s['config_name']:<18} {fname:<22} "
                f"{(c.get('der') or 0) * 100:>6.2f}% "
                f"{g('missed detection')} {g('false alarm')} {g('confusion')}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio-dir", required=True,
                    help="Директория с парами *.wav + *.rttm")
    ap.add_argument("--configs", nargs="+", required=True,
                    help="Один или несколько YAML-конфигов")
    ap.add_argument("--output", default="benchmark_results",
                    help="Куда сохранять JSON")
    ap.add_argument("--limit", type=int, default=None,
                    help="Ограничить число кейсов")
    ap.add_argument("--min-ref-ratio", type=float, default=0.0,
                    help="Только кейсы с ref_speech/audio >= X (0.6 — разумно)")
    ap.add_argument("--uem-from-ref", action="store_true",
                    help="UEM = только ref-сегменты (исключает неаннотированные из FA)")
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)
    cases = discover_cases(audio_dir, limit=args.limit)
    logger.info("Найдено кейсов: %d", len(cases))

    loaded_configs: List[Tuple[str, Dict[str, Any]]] = []
    for cfg_path in args.configs:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        loaded_configs.append((Path(cfg_path).stem, cfg))
        logger.info("Загружен конфиг: %s", Path(cfg_path).stem)

    all_results: List[CaseResult] = []
    for cfg_name, cfg in loaded_configs:
        logger.info("=== Конфигурация: %s ===", cfg_name)
        for wav, rttm in cases:
            logger.info("  Кейс: %s", wav.name)
            res = run_single_case(
                wav, rttm, cfg, cfg_name,
                min_ref_ratio=args.min_ref_ratio,
                uem_from_ref=args.uem_from_ref,
            )
            all_results.append(res)
            if res.error:
                logger.warning("    %s", res.error)
            else:
                logger.info(
                    "    DER=%.2f%%  RTF=%.3f  spk=%d/%d",
                    (res.der or 0) * 100,
                    res.rtf,
                    res.n_speakers_pred or 0,
                    res.n_speakers_ref or 0,
                )

    summaries = summarize(all_results)
    print_summary_table(summaries)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"benchmark-{ts}.json"
    payload = {
        "timestamp": ts,
        "audio_dir": str(audio_dir),
        "configs": [name for name, _ in loaded_configs],
        "min_ref_ratio": args.min_ref_ratio,
        "uem_from_ref": args.uem_from_ref,
        "summaries": [
            {**{k: v for k, v in asdict(s).items() if k != "cases"},
             "cases": [asdict(c) for c in s.cases]}
            for s in summaries
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Результаты сохранены: %s", out_path)

    print_per_case_breakdown(payload)


if __name__ == "__main__":
    main()

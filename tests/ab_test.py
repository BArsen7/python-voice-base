#!/usr/bin/env python3
"""A/B: baseline vs accuracy через штатный DiarizationPipeline."""
import argparse, json, pathlib, sys, time, tracemalloc, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts.pipeline.orchestrator import DiarizationPipeline, load_config, load_wav


def run_mode(name, cfg_path, wav_path):
    print(f"\n=== Режим: {name}  (config={cfg_path}) ===")
    cfg = load_config(cfg_path)
    wf, sr = load_wav(wav_path, target_sr=cfg.get("pipeline", {}).get("sample_rate", 16000))
    dur_ms = int(len(wf) / sr * 1000)

    tracemalloc.start()
    t0 = time.time()
    try:
        pipe = DiarizationPipeline(cfg)
        entries = pipe.run(wf, sr)
    except Exception as e:
        tracemalloc.stop()
        import traceback; traceback.print_exc()
        return {"mode": name, "status": "FAIL", "error": f"{type(e).__name__}: {e}"}
    elapsed = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Валидация схемы
    rows = [e.__dict__ if hasattr(e, "__dict__") else e for e in entries]
    required = {"speaker_id", "start_ms", "end_ms", "text"}
    for i, r in enumerate(rows):
        miss = required - set(r)
        assert not miss, f"[{name}] seg#{i}: нет полей {miss}"
        assert 0 <= r["start_ms"] <= r["end_ms"] <= dur_ms + 1000, \
            f"[{name}] seg#{i}: таймкоды вне файла ({r['start_ms']}-{r['end_ms']} vs dur={dur_ms})"

    n_speakers = len({r["speaker_id"] for r in rows})
    chars = sum(len(r["text"] or "") for r in rows)
    print(f"  ✅ сегментов: {len(rows)}, спикеров: {n_speakers}, "
          f"символов: {chars}")
    print(f"  ⏱  {elapsed:.2f} c,  🧠 peak {peak/1e6:.0f} MB")

    return {
        "mode": name, "status": "PASS",
        "rows": rows,
        "n_segments": len(rows),
        "n_speakers": n_speakers,
        "text_chars": chars,
        "time_s": elapsed,
        "peak_mb": peak / 1e6,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--baseline-config", default="configs/baseline.yaml")
    ap.add_argument("--accuracy-config", default="configs/accuracy_mode.yaml")
    ap.add_argument("--out", default="tests/out")
    args = ap.parse_args()

    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)

    base = run_mode("baseline", args.baseline_config, args.wav)
    acc  = run_mode("accuracy", args.accuracy_config, args.wav)

    # Сохраняем JSON
    for r in (base, acc):
        if r["status"] == "PASS":
            p = pathlib.Path(args.out) / f"transcript_{r['mode']}.json"
            p.write_text(
                json.dumps({"transcript": r["rows"]}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  💾 {p}")

    # Сводка
    print("\n" + "=" * 66)
    print(f"{'метрика':<20} {'baseline':>18} {'accuracy':>18}")
    print("-" * 66)
    for key, label in [("status", "статус"), ("n_segments", "сегментов"),
                       ("n_speakers", "спикеров"), ("text_chars", "символов")]:
        b, a = base.get(key, "—"), acc.get(key, "—")
        print(f"{label:<20} {str(b):>18} {str(a):>18}")
    for key, label in [("time_s", "время, с"), ("peak_mb", "память, MB")]:
        b, a = base.get(key), acc.get(key)
        bs = f"{b:.2f}" if isinstance(b, float) else "—"
        as_ = f"{a:.2f}" if isinstance(a, float) else "—"
        print(f"{label:<20} {bs:>18} {as_:>18}")
    print("=" * 66)

    if base["status"] != "PASS" or acc["status"] != "PASS":
        sys.exit(1)
    print("✅ Оба режима работают E2E через DiarizationPipeline")


if __name__ == "__main__":
    main()

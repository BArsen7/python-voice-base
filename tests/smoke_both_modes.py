#!/usr/bin/env python3
"""Smoke: классы обоих режимов создаются через те же ключи, что и в orchestrator."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import yaml
import numpy as np

results = []
def check(name, fn):
    try:
        fn(); results.append((name, "PASS", ""))
    except Exception as e:
        results.append((name, "FAIL", f"{type(e).__name__}: {e}"))

def _find_std_config():
    for c in ("configs/default.yaml", "configs/base.yaml", "configs/baseline.yaml",
              "configs/config.yaml", "configs/pipeline.yaml",
              "config.yaml", "config.yml"):
        if pathlib.Path(c).exists():
            return pathlib.Path(c)
    return None
STD_CFG_PATH = _find_std_config()
ACC_CFG_PATH = pathlib.Path("configs/accuracy_mode.yaml")

# --- Конфиги + правильные ключи
def t_std_cfg_exists():
    assert STD_CFG_PATH is not None and STD_CFG_PATH.exists(), \
        "baseline-конфиг не найден (см. candidates в _find_std_config)"
def t_acc_cfg_keys():
    cfg = yaml.safe_load(ACC_CFG_PATH.read_text())
    assert cfg["mode"] == "accuracy"
    assert cfg["asr"]["model"] == "sherpa_whisper_medium", cfg["asr"].get("model")
    assert cfg["vad"]["model"] == "hybrid_vad",             cfg["vad"].get("model")
    assert cfg["clustering"]["algorithm"] == "spectral",    cfg["clustering"].get("algorithm")
    assert cfg["embeddings"]["model"] == "wespeaker_resnet34"
    # чужие ключи не должны остаться
    for k in ("asr", "vad", "embeddings"):
        assert "type" not in cfg[k], f"{k}.type устарел, используйте {k}.model"

check("config: standard", t_std_cfg_exists)
check("config: accuracy (model/algorithm)", t_acc_cfg_keys)

# --- Accuracy Mode классы
def t_acc_asr():
    from scripts.models.asr import SherpaOnnxWhisperMediumASR
    SherpaOnnxWhisperMediumASR(yaml.safe_load(ACC_CFG_PATH.read_text())["asr"])

def t_acc_vad():
    from scripts.models.vad import HybridVAD
    HybridVAD(yaml.safe_load(ACC_CFG_PATH.read_text())["vad"])

def t_acc_clu():
    from scripts.models.clustering import SpectralClusteringWrapper
    cfg = yaml.safe_load(ACC_CFG_PATH.read_text())["clustering"]
    SpectralClusteringWrapper(cfg)
    rs = np.random.RandomState(0)
    X = np.vstack([rs.randn(8, 16) * .1,
                   rs.randn(8, 16) * .1 + 5]).astype(np.float32)
    labels = SpectralClusteringWrapper({"n_clusters": 2, "n_neighbors": 5}).fit_predict(X)
    assert len(np.unique(labels)) == 2, f"got {np.unique(labels)}"

def t_acc_emb():
    from scripts.models.embeddings import extract_windowed_embeddings
    class Dummy:
        def extract(self, wf, sr): return np.random.rand(256).astype(np.float32)
    sr = 16000
    wf = np.random.randn(sr * 6).astype(np.float32) * .1
    out = extract_windowed_embeddings(Dummy(), wf, sr, window_sec=2.5, hop_sec=1.0)
    assert 3 <= len(out) <= 5, f"windows={len(out)}"

def t_acc_post():
    from scripts.pipeline.alignment import postprocess_transcript
    segs = [
        {"speaker_id": "S0", "start_ms": 0,    "end_ms": 1000, "text": "привет"},
        {"speaker_id": "S1", "start_ms": 1000, "end_ms": 1500, "text": "два три"},
        {"speaker_id": "S0", "start_ms": 1500, "end_ms": 3000, "text": "пять"},
    ]
    out = postprocess_transcript(segs, smooth_method="median", smooth_window=3)
    assert out[1]["speaker_id"] == "S0", "flip не устранён"
    assert "2" in out[1]["text"] and "3" in out[1]["text"]

check("acc: SherpaOnnxWhisperMediumASR", t_acc_asr)
check("acc: HybridVAD",                  t_acc_vad)
check("acc: SpectralClusteringWrapper",  t_acc_clu)
check("acc: extract_windowed_embeddings", t_acc_emb)
check("acc: postprocess_transcript",     t_acc_post)

# --- КРИТИЧНО: оркестратор знает новые модели
def t_orch_builders():
    import scripts.pipeline.orchestrator as o
    cfg = yaml.safe_load(ACC_CFG_PATH.read_text())
    v = o._build_vad(cfg["vad"])
    from scripts.models.vad import HybridVAD
    assert isinstance(v, HybridVAD), f"VAD = {type(v).__name__}, ожидался HybridVAD"
    c = o._build_clustering(cfg["clustering"])
    from scripts.models.clustering import SpectralClusteringWrapper
    assert isinstance(c, SpectralClusteringWrapper), f"Clustering = {type(c).__name__}"
    # ASR lazy — просто проверяем тип
    a = o._build_asr(cfg["asr"])
    from scripts.models.asr import SherpaOnnxWhisperMediumASR
    assert isinstance(a, SherpaOnnxWhisperMediumASR), f"ASR = {type(a).__name__}"

check("orchestrator: билдеры создают accuracy-классы", t_orch_builders)

# --- Стандартные классы (SMOKE, регрессий нет)
def t_std_classes():
    import scripts.models.asr as asr_m, scripts.models.vad as vad_m
    import scripts.models.clustering as clu_m
    assert hasattr(asr_m, "SherpaOnnxASR")
    assert hasattr(vad_m, "SileroVAD")
    assert hasattr(clu_m, "AgglomerativeClusteringWrapper")

check("std: классы присутствуют", t_std_classes)

# --- Вывод
width = max(len(n) for n, _, _ in results)
fails = 0
for name, status, err in results:
    mark = "✅" if status == "PASS" else "❌"
    print(f"{mark}  {name:<{width}}  {err}")
    if status == "FAIL": fails += 1
print()
if fails:
    print(f"ПРОВАЛЕНО: {fails}/{len(results)}")
    sys.exit(1)
print(f"OK: {len(results)}/{len(results)} smoke-проверок пройдено")

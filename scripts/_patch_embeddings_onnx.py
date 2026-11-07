"""Патч WeSpeakerEmbeddingExtractor._lazy_load — поддержка ONNX.

Ищет метод по маркеру внутри class WeSpeakerEmbeddingExtractor
и заменяет тело. Идемпотентен: повторный запуск ничего не меняет.
"""
from __future__ import annotations  # первой строкой кода

import re
import sys
from pathlib import Path


OLD_BODY = '''\
    def _lazy_load(self):
        if self._model is not None:
            return
        try:
            from pyannote.audio.pipelines.speaker_verification import (
                PretrainedSpeakerEmbedding,
            )
        except ImportError as e:
            raise ImportError(
                "WeSpeakerEmbeddingExtractor requires pyannote.audio. "
                "Install with: pip install pyannote.audio"
            ) from e

        self._model = PretrainedSpeakerEmbedding(
            "hbredin/wespeaker-voxceleb-resnet34-LM"
        )
        logger.info("WeSpeaker ResNet34 embedding model loaded")
'''

NEW_BODY = '''\
    def _lazy_load(self):
        if self._model is not None:
            return

        use_onnx = bool(self.config.get("use_onnx", False))
        onnx_path = self.config.get("onnx_path")

        if use_onnx:
            if not onnx_path:
                raise ValueError(
                    "embeddings.use_onnx=true requires embeddings.onnx_path"
                )
            if not Path(onnx_path).exists():
                raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
            try:
                from pyannote.audio.pipelines.speaker_verification import (
                    ONNXWeSpeakerPretrainedSpeakerEmbedding,
                )
            except ImportError as e:
                raise ImportError(
                    "ONNX path requires pyannote.audio >= 3.3"
                ) from e
            self._model = ONNXWeSpeakerPretrainedSpeakerEmbedding(onnx_path)
            logger.info("WeSpeaker ResNet34 ONNX loaded from %s", onnx_path)
            return

        try:
            from pyannote.audio.pipelines.speaker_verification import (
                PretrainedSpeakerEmbedding,
            )
        except ImportError as e:
            raise ImportError(
                "WeSpeakerEmbeddingExtractor requires pyannote.audio. "
                "Install with: pip install pyannote.audio"
            ) from e

        self._model = PretrainedSpeakerEmbedding(
            "hbredin/wespeaker-voxceleb-resnet34-LM"
        )
        logger.info("WeSpeaker ResNet34 embedding model loaded (pyannote)")
'''


def main() -> None:
    path = Path("scripts/models/embeddings.py")
    src = path.read_text()

    # Идемпотентность: уже пропатчено?
    if "use_onnx" in src and "ONNXWeSpeakerPretrainedSpeakerEmbedding" in src:
        print("[i] already patched — nothing to do")
        return

    # Границы: от "class WeSpeakerEmbeddingExtractor" до "class ECAPAEmbeddingExtractor"
    m = re.search(
        r"(class WeSpeakerEmbeddingExtractor\b.*?)(?=\nclass ECAPAEmbeddingExtractor\b)",
        src, re.DOTALL,
    )
    if not m:
        raise SystemExit("ERR: cannot find WeSpeakerEmbeddingExtractor block")
    wespeaker_block = m.group(1)

    # Внутри блока ищем точный OLD_BODY
    if OLD_BODY not in wespeaker_block:
        raise SystemExit(
            "ERR: OLD_BODY not found verbatim in WeSpeakerEmbeddingExtractor.\n"
            "Возможные причины: тело _lazy_load уже изменено, "
            "или изменены отступы. Пришли `git diff`, подгоним."
        )

    new_block = wespeaker_block.replace(OLD_BODY, NEW_BODY)
    new_src = src[: m.start(1)] + new_block + src[m.end(1):]

    # Валидация синтаксиса до записи
    import ast
    try:
        ast.parse(new_src)
    except SyntaxError as e:
        raise SystemExit(f"ERR: patched source has SyntaxError: {e}")

    # Импорт Path в шапку (если ещё нет)
    if "from pathlib import Path" not in new_src.split("class ")[0]:
        new_src = new_src.replace(
            "from __future__ import annotations\n",
            "from __future__ import annotations\n\nfrom pathlib import Path\n",
            1,
        )

    path.write_text(new_src)
    print(f"[OK] patched {path}")
    print("     _lazy_load supports embeddings.use_onnx + onnx_path")


if __name__ == "__main__":
    main()

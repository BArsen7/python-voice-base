"""Combined patch v2: chunking + blob-fallback using AST boundaries.

Previous version used indent-based scanning, which broke on multi-line
signatures (closing ')' is at same indent as 'def'). This version uses
ast.FunctionDef.lineno / end_lineno for exact method boundaries.
"""
from __future__ import annotations

import ast
from pathlib import Path


NEW_METHODS = '''    MAX_CHUNK_SEC = 30

    def transcribe(
        self, waveform: np.ndarray, sample_rate: int
    ) -> List[WordTimestamp]:
        self._lazy_load()
        if sample_rate != 16000:
            raise ValueError("SherpaOnnxASR expects 16 kHz audio")
        n = len(waveform)
        chunk_samples = self.MAX_CHUNK_SEC * sample_rate
        if n <= chunk_samples:
            return self._transcribe_chunk(waveform, sample_rate, offset_ms=0)
        all_words: List[WordTimestamp] = []
        for start in range(0, n, chunk_samples):
            end = min(start + chunk_samples, n)
            offset_ms = int(start / sample_rate * 1000)
            all_words.extend(
                self._transcribe_chunk(waveform[start:end], sample_rate, offset_ms)
            )
        return all_words

    def _transcribe_chunk(
        self, waveform: np.ndarray, sample_rate: int, offset_ms: int = 0
    ) -> List[WordTimestamp]:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, waveform.astype(np.float32))
        self._recognizer.decode_stream(stream)
        result = stream.result
        if not result.text:
            return []
        words: List[WordTimestamp] = []
        tokens = getattr(result, "tokens", []) or []
        timestamps = getattr(result, "timestamps", []) or []
        if tokens and timestamps and len(tokens) == len(timestamps):
            current_word = ""
            start_ms = None
            last_token_ms = 0
            for tok, ts in zip(tokens, timestamps):
                ts_ms = int(ts * 1000) + offset_ms
                is_new = tok.startswith(" ") or tok.startswith("\\u2581")
                clean = tok.lstrip(" \\u2581")
                if not clean:
                    continue
                if is_new:
                    if current_word:
                        words.append(WordTimestamp(
                            word=current_word,
                            start_ms=start_ms + offset_ms,
                            end_ms=ts_ms,
                        ))
                    current_word = clean
                    start_ms = int(ts * 1000)
                else:
                    if not current_word:
                        current_word = clean
                        start_ms = int(ts * 1000)
                    else:
                        current_word += clean
                last_token_ms = ts_ms
            if current_word and start_ms is not None:
                words.append(WordTimestamp(
                    word=current_word,
                    start_ms=start_ms + offset_ms,
                    end_ms=last_token_ms,
                ))
        # Blob fallback: 600-char word from sherpa tail-degradation
        if any(len(w.word) > 30 for w in words):
            parts = result.text.split()
            if parts:
                total_ms = int(len(waveform) / sample_rate * 1000)
                step = max(1, total_ms // len(parts))
                return [
                    WordTimestamp(
                        word=p,
                        start_ms=offset_ms + i * step,
                        end_ms=offset_ms + (i + 1) * step,
                    )
                    for i, p in enumerate(parts)
                ]
        return words
'''


def main() -> None:
    p = Path("scripts/models/asr.py")
    s = p.read_text()

    if "MAX_CHUNK_SEC = 30" in s and "Blob fallback" in s:
        print("[i] already patched (chunk + blob)")
        return

    lines = s.splitlines(keepends=True)

    # --- AST-based boundary detection ---
    tree = ast.parse(s)

    target_cls = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SherpaOnnxASR":
            target_cls = node
            break
    if target_cls is None:
        raise SystemExit("ERR: class SherpaOnnxASR not found")

    target_method = None
    for node in target_cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == "transcribe":
            target_method = node
            break
    if target_method is None:
        raise SystemExit("ERR: SherpaOnnxASR.transcribe not found")

    tr_start = target_method.lineno - 1     # 0-based, def line
    tr_end = target_method.end_lineno       # 0-based, exclusive (past last body line)

    # trim trailing blank lines within boundary
    while tr_end > tr_start and not lines[tr_end - 1].strip():
        tr_end -= 1

    print(f"[i] class SherpaOnnxASR at line {target_cls.lineno}")
    print(f"[i] transcribe() at lines {tr_start+1}..{tr_end} (AST-based, incl. body)")

    new_lines = lines[:tr_start] + [NEW_METHODS, "\n"] + lines[tr_end:]
    new_src = "".join(new_lines)

    try:
        ast.parse(new_src)
    except SyntaxError as e:
        raise SystemExit(f"ERR: SyntaxError after patch: {e}")

    p.write_text(new_src)
    print("[OK] patched asr.py (chunk 30s + blob fallback)")


if __name__ == "__main__":
    main()

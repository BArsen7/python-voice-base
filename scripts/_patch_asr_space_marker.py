"""Fix SherpaOnnxASR: token word-boundary marker is a SPACE, not ▁.

sherpa-onnx converts SentencePiece '▁' to ' ' when building result.tokens.
So tok.startswith(" ") means 'new word'; otherwise continuation.
"""
from __future__ import annotations

import ast
from pathlib import Path


NEW_BLOCK = '''            # sherpa-onnx converts SentencePiece '▁' to ' ' in result.tokens.
            # A token starting with ' ' is a new word; otherwise continuation.
            for tok, ts in zip(tokens, timestamps):
                ts_ms = int(ts * 1000)
                is_new_word = tok.startswith(" ") or tok.startswith("\\u2581")
                clean = tok.lstrip(" \\u2581")
                if not clean:
                    continue
                if is_new_word:
                    if current_word:
                        words.append(
                            WordTimestamp(
                                word=current_word,
                                start_ms=start_ms,
                                end_ms=ts_ms,
                            )
                        )
                    current_word = clean
                    start_ms = ts_ms
                else:
                    if not current_word:
                        current_word = clean
                        start_ms = ts_ms
                    else:
                        current_word += clean
                last_token_ms = ts_ms
            if current_word and start_ms is not None:
                words.append(
                    WordTimestamp(
                        word=current_word,
                        start_ms=start_ms,
                        end_ms=last_token_ms,
                    )
                )
'''


def main() -> None:
    p = Path("scripts/models/asr.py")
    s = p.read_text()

    if "token starting with ' ' is a new word" in s:
        print("[i] already patched (space-marker) — nothing to do")
        return

    lines = s.splitlines(keepends=True)

    # Найти первый маркер текущего (gap-group) блока
    start = None
    for i, line in enumerate(lines):
        if "# Group subword tokens into words by inter-token gap" in line:
            start = i
            break
    if start is None:
        # Fallback: найти по 'for i, (tok, ts) in enumerate'
        for i, line in enumerate(lines):
            if "for i, (tok, ts) in enumerate" in line or "for tok, ts in zip(tokens, timestamps):" in line:
                start = i
                break
    if start is None:
        raise SystemExit("ERR: token loop not found")

    # Захватить предшествующий init-блок (current_word/start_ms/last_token_ms)
    init_start = start
    while init_start > 0:
        prev = lines[init_start - 1].lstrip()
        if prev.startswith(("current_word", "start_ms", "last_token_ms")):
            init_start -= 1
        else:
            break

    indent = len(lines[start]) - len(lines[start].lstrip())

    # Конец блока: до строки с отступом ≤ indent, которая НЕ комментарий
    end = start + 1
    while end < len(lines):
        l = lines[end]
        if not l.strip() or l.lstrip().startswith("#"):
            end += 1
            continue
        if (len(l) - len(l.lstrip())) <= indent:
            break
        end += 1

    # Также захватить trailing "if current_word and start_ms is not None:" если есть
    tail = end
    while tail < len(lines):
        l = lines[tail]
        if not l.strip():
            tail += 1
            continue
        st = l.lstrip()
        ci = len(l) - len(l.lstrip())
        if ci == indent and st.startswith("if current_word"):
            body_indent = None
            j = tail + 1
            while j < len(lines):
                lj = lines[j]
                if not lj.strip():
                    j += 1
                    continue
                cj = len(lj) - len(lj.lstrip())
                if body_indent is None:
                    body_indent = cj
                if cj < body_indent:
                    break
                j += 1
            tail = j
            continue
        break

    print(f"[i] replacing lines {init_start+1}..{tail} of asr.py")

    new_lines = (
        lines[:init_start]
        + [
            '            current_word = ""\n',
            '            start_ms = None\n',
            '            last_token_ms = 0\n',
        ]
        + [NEW_BLOCK]
        + lines[tail:]
    )
    new_src = "".join(new_lines)

    try:
        ast.parse(new_src)
    except SyntaxError as e:
        raise SystemExit(f"ERR: SyntaxError after patch: {e}")

    p.write_text(new_src)
    print("[OK] patched asr.py (space-marker)")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "=== SHA256 ==="
( cd models && sha256sum -c SHA256SUMS )
echo
echo "=== structure ==="
find models -type f | sort
echo
echo "=== sizes ==="
du -sh models/* | sort -h
echo
echo "=== total ==="
du -sh .
echo "[OK] integrity check passed"

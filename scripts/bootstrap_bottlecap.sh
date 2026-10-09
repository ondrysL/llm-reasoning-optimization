#!/usr/bin/env bash
set -euo pipefail

# Frozen revision reviewed on 2026-10-09. Upgrade explicitly, never implicitly.
BOTTLECAP_SHA="5f9119e8c5e1724a90fd10f86ff83c9b1d775937"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT/external/ThinkingCap-Test"

mkdir -p "$ROOT/external"
if [[ ! -d "$TARGET/.git" ]]; then
  git clone https://github.com/BottleCapAI/ThinkingCap-Test.git "$TARGET"
fi

if ! git -C "$TARGET" cat-file -e "$BOTTLECAP_SHA^{commit}" 2>/dev/null; then
  git -C "$TARGET" fetch origin "$BOTTLECAP_SHA"
fi

git -C "$TARGET" checkout --detach "$BOTTLECAP_SHA"
echo "Pinned BottleCapAI to $(git -C "$TARGET" rev-parse HEAD)"

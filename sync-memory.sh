#!/usr/bin/env bash
# Sync the committed Claude Code memory notes into the local per-project memory
# directory, so Claude Code loads them automatically.
#
# Run from the repo root after `git pull`:
#     bash sync-memory.sh
#
# Claude Code stores memory at ~/.claude/projects/<hash>/memory/, where <hash>
# is the project's absolute path with path separators replaced by dashes.
# Launch Claude Code from this same directory so the hash matches.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HOME/.claude/projects/$(printf '%s' "$ROOT" | sed 's#/#-#g')/memory"

mkdir -p "$DEST"
cp "$ROOT/memory_files/"*.md "$DEST/"
echo "Synced $(ls "$ROOT/memory_files/"*.md | wc -l) memory files -> $DEST"

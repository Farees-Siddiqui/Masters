---
name: dev-server-reload-gotcha
description: "uvicorn --reload is unreliable on this Windows box; restart manually, watch for zombie port sockets"
metadata: 
  node_type: memory
  type: project
  originSessionId: dfd50d75-4b6f-4fbf-b669-a14ff0fef3b3
---

On this Windows 11 setup, `uvicorn app.main:app --reload` (WatchFiles) is
unreliable: it often detects a file change and prints "Reloading…" but never
completes the worker restart (no "Application startup complete", no traceback),
so the server keeps serving stale code and new routes 404. Editing `app/main.py`
frequently did not trigger a reload at all; a new file (e.g. `alignment/search.py`)
did.

**How to apply:** run the dev server WITHOUT `--reload` and restart it manually
after backend edits (`.venv/Scripts/python.exe -m uvicorn app.main:app --port <p>`).
When killing it, the reloader can respawn workers and a dead PID can keep port
8000 in a zombie LISTEN state (`Get-NetTCPConnection -LocalPort 8000` shows an
OwningProcess that no longer exists), so a fresh start silently fails to bind.
Easiest fix: start on a fresh port (we moved 8000 → 8001 → 8002). Verify routes
loaded via `curl /openapi.json`. Frontend assets are cache-busted with `?v=N`
in `static/align.html`; bump on every css/js change. See [[project-overview]].

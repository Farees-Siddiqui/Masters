# Running the Document AST Viewer

## Prerequisites

- Python 3.10+
- A Mistral API key (already present in `API_KEY.txt`, or set `MISTRAL_API_KEY` in the environment)

## One-time setup

From the repo root (`C:\Users\faree\Documents\Masters\AST`):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS / Linux replace the activate line with:

```bash
source .venv/bin/activate
```

## Start the server

```powershell
uvicorn app.main:app --reload
```

You should see:

```
Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

## Use it

1. Open <http://127.0.0.1:8000> in a browser.
2. Click **Choose File**, pick a PDF.
3. Click **Build AST**.
4. The left pane shows a collapsible AST tree; the right pane shows the raw JSON.

## Useful flags

- Bind to all interfaces: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- Disable autoreload (production-ish): drop `--reload`
- Change port: `--port 5050`

## API (direct call)

```powershell
curl.exe -F "file=@path\to\doc.pdf" http://127.0.0.1:8000/api/upload
```

Response shape is documented in [DATASTRUCTS.md](DATASTRUCTS.md) under
*`/api/upload` response*.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `RuntimeError: No Mistral API key found` | Set `MISTRAL_API_KEY` env var, or ensure `API_KEY.txt` exists at the repo root. |
| `OCR failed: 401` | Key is invalid or rate-limited. Verify it on the Mistral console. |
| `Only PDF files are supported in v1.` | The uploader currently only accepts `.pdf`. Convert other formats first. |
| Browser shows old JS/CSS after edits | Hard refresh (Ctrl+Shift+R) — static files are cached. |
| Port 8000 already in use | Pass `--port 5050` (or any free port) to `uvicorn`. |

## Stopping

`Ctrl+C` in the terminal running uvicorn.

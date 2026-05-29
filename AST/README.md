# Document AST Viewer

Upload a PDF → Mistral OCR → nested-section AST, rendered as both a collapsible tree and raw JSON.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The Mistral API key is read from `MISTRAL_API_KEY` if set, otherwise from `API_KEY.txt` at the repo root (gitignored).

## Run

```powershell
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000.

## Layout

- `app/ast_builder.py` — `Node` + markdown → AST
- `app/ocr.py` — Mistral OCR wrapper
- `app/main.py` — FastAPI app + `/api/upload`
- `static/` — frontend (HTML / CSS / JS)
- `DATASTRUCTS.md` — every data structure documented here

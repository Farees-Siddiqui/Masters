#!/usr/bin/env bash
# Regenerate the corpus: gold XML from build_gold.py, PDFs from tex/.
set -euo pipefail
cd "$(dirname "$0")"

python build_gold.py

for f in tex/*.tex; do
    latexmk -pdf -quiet -outdir=Output "$f" >/dev/null
done
latexmk -c -outdir=Output tex/*.tex >/dev/null 2>&1 || true

echo "built $(ls Output/*.pdf | wc -l) pdfs in Output/"

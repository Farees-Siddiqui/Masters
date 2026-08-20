# OCR engine survey -- structural output for G_doc

10 arXiv papers, page 1 only, rendered at 200 DPI. Ground truth is the pdfplumber native PDF text layer, page 1; no LaTeX source is read anywhere in this pipeline. Every engine runs in its own virtualenv on Tesla V100-SXM2-32GB.

Total sweep: 53.2 min wall clock (batched-per-engine invocation).

## Results

| Engine | Mode | Word F1 | Word recall | Avg CER | Content CER | Boxes | Box format | Avg s/page | Peak VRAM |
|---|---|---|---|---|---|---|---|---|---|
| deepseek-ocr | grounding | **0.893** | 0.874 | 0.166 | 0.150 | Yes (11/page) | `xyxy_pixels (from 0-999 normalised grid)` | 62.11 | 9.1 GB |
| deepseek-ocr | text | **0.944** | 0.920 | 0.118 | 0.109 | **No** | `--` | 53.52 | 9.1 GB |
| glm-ocr | grounding | -- | -- | -- | -- | **No** | `--` | -- | -- |
| glm-ocr | text | **0.943** | 0.917 | 0.120 | 0.112 | **No** | `--` | 41.81 | 6.6 GB |
| paddleocr | grounding | **0.844** | 0.788 | 0.199 | 0.184 | Yes (16/page) | `xyxy_pixels (layout blocks)` | 11.90 | 4.7 GB |
| paddleocr | text | **0.978** | 0.978 | 0.281 | 0.268 | Yes (80/page) | `quad_4point_pixels (+ xyxy derived)` | 3.04 | 2.3 GB |
| unlimited-ocr | grounding | **0.856** | 0.849 | 0.208 | 0.191 | Yes (15/page) | `xyxy_pixels (from 0-999 normalised grid)` | 55.56 | 11.4 GB |
| unlimited-ocr | text | **0.886** | 0.949 | 0.225 | 0.192 | **No** | `--` | 57.06 | 11.4 GB |

**Read Word F1 as the headline number, not CER.** CER is a sequential metric: it cannot tell "recognised the wrong characters" apart from "recognised the right characters in a different order". 5 of these 10 pages are two-column, so ordering dominates. Word F1 and recall compare token multisets and are therefore independent of reading order -- including independent of the reading order this harness reconstructs for the ground truth.

- **Word F1 / recall** -- order-independent token overlap. Recall answers "did it read all the text on the page".
- **Avg CER** -- whitespace-collapsed edit distance against ground truth in reconstructed reading order. Also penalises a model for emitting `#` headings and `\(x\)` math the text layer never contained.
- **Content CER** -- markup and punctuation stripped, lowercased.

## Spatial output

Box geometry validated against the rendered page size: a box is bad if it falls outside the page or has non-positive area.

| Engine | Mode | Boxes | Granularity | Out of bounds | Degenerate | Median box (w x h, % of page) |
|---|---|---|---|---|---|---|
| deepseek-ocr | grounding | 112 | layout block | 0 | 0 | 39% x 5.2% |
| paddleocr | grounding | 165 | layout block | 0 | 0 | 36% x 3.0% |
| paddleocr | text | 795 | text line | 0 | 0 | 38% x 1.3% |
| unlimited-ocr | grounding | 146 | layout block | 0 | 0 | 37% x 3.0% |

### Box vocabularies

- **deepseek-ocr / grounding**: equation, image, image_caption, sub_title, text, title
- **paddleocr / grounding**: abstract, aside_text, chart, doc_title, figure_title, footer, footnote, formula, header, image, number, paragraph_title, text
- **paddleocr / text**: text_line
- **unlimited-ocr / grounding**: aside_text, equation, header, image, image_caption, page_footnote, page_number, text, title

## Per-paper Word F1

| Paper | cols | deepseek-ocr/grounding | deepseek-ocr/text | glm-ocr/grounding | glm-ocr/text | paddleocr/grounding | paddleocr/text | unlimited-ocr/grounding | unlimited-ocr/text |
|---|---|---|---|---|---|---|---|---|---|
| adam | 1 | 0.928 | 0.974 | -- | 0.974 | 0.717 | 0.992 | 0.982 | 0.921 |
| attention | 1 | 0.759 | 0.976 | -- | 0.959 | 0.723 | 0.991 | 0.000 | 0.886 |
| batchnorm | 2 | 0.884 | 0.909 | -- | 0.909 | 0.875 | 0.929 | 0.900 | 0.834 |
| bert | 2 | 0.867 | 0.929 | -- | 0.929 | 0.903 | 0.986 | 0.925 | 0.877 |
| densenet | 2 | 0.913 | 0.923 | -- | 0.923 | 0.914 | 0.982 | 0.947 | 0.853 |
| faster-rcnn | 2 | 0.867 | 0.913 | -- | 0.913 | 0.930 | 0.994 | 0.961 | 0.908 |
| resnet | 2 | 0.931 | 0.908 | -- | 0.910 | 0.901 | 0.982 | 0.942 | 0.864 |
| unet | 1 | 0.945 | 0.973 | -- | 0.973 | 0.930 | 0.987 | 0.962 | 0.897 |
| vgg | 1 | 0.908 | 0.963 | -- | 0.962 | 0.684 | 0.960 | 0.968 | 0.916 |
| vit | 1 | 0.926 | 0.971 | -- | 0.973 | 0.859 | 0.980 | 0.976 | 0.905 |

## Notable failure modes

- **deepseek-ocr / text degrades on two-column pages.** CER 0.036 single-column vs 0.200 two-column. Modest next to the detection-order engines above, and partly attributable to hyphenation and to the ground truth's own reconstructed reading order rather than to the model.
- **glm-ocr / text degrades on two-column pages.** CER 0.041 single-column vs 0.199 two-column. Modest next to the detection-order engines above, and partly attributable to hyphenation and to the ground truth's own reconstructed reading order rather than to the model.
- **paddleocr / text has no reading order.** CER 0.011 on single-column pages vs 0.550 on two-column -- a 49x collapse, while its Word F1 stays at 0.978. It recognises the glyphs almost perfectly and returns them in detection order, zipping the two columns together. Any consumer must supply its own reading order.
- **deepseek-ocr / grounding silently drops page content.** Word recall below 0.70 on 1 of 10 pages: attention (0.62). The text it does emit is accurate (precision stays high), so the loss is invisible unless you measure recall.
- **paddleocr / grounding silently drops page content.** Word recall below 0.70 on 3 of 10 pages: vgg (0.55), attention (0.57), adam (0.61). The text it does emit is accurate (precision stays high), so the loss is invisible unless you measure recall.
- **unlimited-ocr / grounding returned nothing at all on `attention`.** Zero characters, zero boxes, no exception -- the model emits EOS immediately. Reproduced deterministically at temperature 0 (3.0 s versus its usual ~55 s). A pipeline that does not check for empty output would record this as a clean run.
- **unlimited-ocr / text over-generates.** Word precision below 0.80 on 3 page(s): batchnorm (0.77), densenet (0.79), attention (0.80), with recall staying high -- the signature of repeated or invented spans rather than missed text.
- **deepseek-ocr pays for structure in coverage.** Word recall drops 0.920 -> 0.874 when asked for boxes instead of plain text: the grounded pass reads less of the page.
- **paddleocr pays for structure in coverage.** Word recall drops 0.978 -> 0.788 when asked for boxes instead of plain text: the grounded pass reads less of the page.

## Errors and skips

- `glm-ocr/grounding` (all pages): GLM-OCR exposes no grounding prompt and no bbox tokens; layout boxes come from a separate PP-DocLayout-V3 stage in the official pipeline.

## Bottom line for G_doc

The survey question was which engine yields both the spatial structure a candidate graph needs and accurate text. No single engine is best at both, and the split is clean:

- **PaddleOCR det+rec has the best text and the finest boxes, and no reading order.** Word F1 0.978 -- the highest here -- with ~80 line-level quadrilaterals per page at 3.0 s, the fastest by 14x. But CER goes 0.011 to 0.550 between single- and two-column pages because lines come back in detection order. For a graph this matters less than it looks: G_doc nodes are spatial, so you would impose reading order from the boxes yourself rather than trust a serialised string.
- **DeepSeek-OCR grounding is the strongest single-pass structured output.** Word F1 0.893 with semantically labelled blocks (title, sub_title, text, equation, image, image_caption) and no invalid geometry -- but 62 s/page and ~11 coarse blocks, so fine-grained nodes need a second pass.
- **GLM-OCR cannot serve this role at all.** Competitive text (F1 0.943) but structurally blind: no grounding prompt, no bbox vocabulary. Its own pipeline gets layout from a separate PP-DocLayout-V3 stage, which is PaddleOCR's detector -- so choosing GLM-OCR means running Paddle anyway.
- **Unlimited-OCR is not yet dependable.** Comparable structure to DeepSeek (F1 0.856), but it over-generates on 3 of 10 pages and returned nothing at all on one, deterministically.

**Suggested pairing:** PaddleOCR det+rec for text and line boxes (fast, highest recall, finest granularity), with PP-StructureV3 or DeepSeek-OCR grounding supplying block-level semantic labels over the top. Check the recall gap before trusting either grounded pass as the sole source of text -- PP-StructureV3 dropped a fifth of the words on average and nearly half on two of the pages.

## Environment caveats

These numbers were measured on Tesla V100 (compute capability 7.0), which constrains the setup in ways worth stating before the latencies are quoted anywhere:

- **FlashAttention-2 could not be used.** It requires Ampere (sm_80+). All transformer engines ran `eager` or `sdpa` attention, so the per-page times here are an upper bound; an A100/H100 would be materially faster.
- **DeepSeek-OCR and Unlimited-OCR are forced into bfloat16**, which Volta has no tensor cores for. Their modeling code hardcodes `.to(torch.bfloat16)` on image tensors and a bf16 autocast, so float16 (about 9x faster on this GPU) fails in `masked_scatter_`. Their latencies are penalised accordingly; GLM-OCR and PaddleOCR were free to use float16/float32.
- **vLLM was not used.** It would have replaced the pinned torch in each venv and broken the three mutually-incompatible transformers versions (DeepSeek 4.46.3, Unlimited 4.57.1, GLM 5.14.1) that make these engines coexist at all.
- Engines ran **sequentially, one at a time**: the container caps RAM at 16 GiB and parallel model loads are OOM-killed.

## Reproducing

```
python3 fetch_arxiv_pdfs.py                                  # 10 PDFs
ocr_venvs/env_eval/bin/python prepare_pages.py               # renders + GT
ocr_venvs/env_eval/bin/python orchestrate_benchmark.py       # full sweep
ocr_venvs/env_eval/bin/python orchestrate_benchmark.py --rescore  # metrics only
```

`--rescore` recomputes every metric from the stored engine outputs in `ocr_survey_results.json`, so changing a CER definition or the ground-truth extraction costs seconds instead of another GPU sweep. Exact package sets for all five virtualenvs are frozen in `requirements/`.
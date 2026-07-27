# Literature review — alignment of logical structure and page layout

Status: **cited** = now in `main.tex`; **candidate** = worth reading, not yet cited.
Verified via web search; arXiv DOIs are deterministic and safe, publisher DOIs
marked (!) still need checking against the publisher page before submission.

---

## Two findings that changed the paper's positioning

**1. "Structure-oriented systems discard geometry" was too strong.**
It holds for end-to-end VLM transcribers (Nougat, GOT-OCR, olmOCR, Donut), which
emit a token sequence with no coordinates. It is **false** for pipeline toolkits:
Docling emits a `DoclingDocument` where *every element retains page and bounding
box*, and MinerU does the same. The claim is now scoped to the end-to-end family,
and the toolkits are handled in the "building structure on top of layout" theme.

**2. The closest prior art was missing entirely, and it is not PubLayNet/DocBank.**
It is **hierarchical document structure analysis**: HRDoc (AAAI 2023),
Detect-Order-Construct + Comp-HRDoc (Pattern Recognition 2024), UniHDSA (2025).
These recover a logical tree from a PDF *and* keep the physical grounding. A
referee in this field would know them immediately, and their absence was the
single largest novelty risk in the draft.

The distinction now argued in the paper: in all of that work the logical structure
is **defined over** the physical units, so the correspondence is an artefact of
construction and cannot be independently wrong — but the structure is then bounded
by the detector, and what the detector misreads (mathematics, cross-column flow)
cannot be repaired higher up. Our setting inverts the dependency: an independent
recogniser may recover exactly that material, at the price of a correspondence
that must now be inferred. That inversion is the paper's claim to novelty and it
should be stated this explicitly wherever the contribution is summarised.

---

## Theme 1 — Document structured extraction (hierarchy, no geometry)

| Work | Venue / ID | Why it matters | Status |
|---|---|---|---|
| Donut | ECCV 2022, arXiv:2111.15664 | OCR-free image→structured output; founds the end-to-end family | cited |
| Nougat | arXiv:2308.13418 | academic papers → markup preserving maths | cited |
| GOT-OCR 2.0 | arXiv:2409.01704 | "OCR-2.0": formulas, tables, charts as characters, one model | cited |
| olmOCR 2 | arXiv:2510.19817 | distilled 7B VLM, corpus-scale conversion | cited |
| DeepSeek-OCR | arXiv:2510.18234 | contexts optical compression; recent alternative | candidate |
| READoc | arXiv:2409.05137 | argues DSE evaluation is fragmented; unified end-to-end benchmark over 3,576 docs | cited |

READoc is the closest thing to a rival evaluation contribution — read it before
finalising the benchmark sections, and position our node-level protocol against it.

## Theme 2 — Layout analysis (geometry, no hierarchy)

| Work | Venue / ID | Why it matters | Status |
|---|---|---|---|
| PubLayNet | ICDAR 2019, DOI 10.1109/ICDAR.2019.00166 (!) | 360k pages, labels from publisher XML | cited |
| DocBank | COLING 2020, DOI 10.18653/v1/2020.coling-main.82 | 500k pages, token labels from LaTeX | cited |
| DocLayNet | KDD 2022, DOI 10.1145/3534678.3539043 | 80,863 *human* annotated pages; explicitly criticises PubLayNet/DocBank for lacking layout variety | cited |
| LayoutLM | KDD 2020, DOI 10.1145/3394486.3403172 (!) | text+position+image fusion | cited |
| DLAFormer | ICDAR 2024, DOI 10.1007/978-3-031-70546-5_3 | detection + logical role + reading order as one relation-prediction transformer | cited |

DocLayNet's own criticism of arXiv/PubMed-only corpora is useful cover for our
ten-paper corpus limitation: cite it *as* the reason the caveat is stated.

## Theme 3 — Building structure on top of layout (closest prior art)

| Work | Venue / ID | Why it matters | Status |
|---|---|---|---|
| HRDoc | AAAI 2023, arXiv:2303.13839 | 2,500 docs, line-level categories + parent relations; unit classification / parent finding / relation classification | cited |
| Detect-Order-Construct | Pattern Recognition 2024, arXiv:2401.11874 | tree construction; introduces Comp-HRDoc (detection + order + ToC + hierarchy) | cited |
| UniHDSA | Pattern Recognition 2025, arXiv:2503.15893 | unified relation prediction for hierarchical structure | candidate |
| Docling | arXiv:2408.09869 (+ toolkit paper 2501.17887) | structured doc with full provenance per element | cited |
| MinerU2.5 | arXiv:2509.22186 | decoupled VLM parsing pipeline | cited |
| Marker | software, no paper | practitioner baseline | candidate |

## Theme 4 — Reading order

| Work | Venue / ID | Why it matters | Status |
|---|---|---|---|
| Nagy & Seth | ICPR 1984 | the original recursive XY cut (no DOI, pre-DOI era) | cited |
| LayoutReader / ReadingBank | EMNLP 2021, DOI 10.18653/v1/2021.emnlp-main.389 | seq2seq ordering; order harvested from Word XML | cited |
| Ordering relations | EMNLP 2024, DOI 10.18653/v1/2024.emnlp-main.540 | recasts order as pairwise relations | cited |
| XY-Cut++ | arXiv:2504.10258 | pre-mask + multi-granularity segmentation; what we implement | cited |

## Theme 5 — Alignment used as supervision

| Work | Venue / ID | Why it matters | Status |
|---|---|---|---|
| Toselli, Wu & Smith | ICDAR 2021, arXiv:2112.12703 | force-aligns TEI digital-edition text to page images as distant supervision over 500k DTA pages | cited |
| PubLayNet / DocBank / ReadingBank | above | each exploits a privileged source | cited |

Toselli et al. matters because it shows the same manoeuvre outside the ML-dataset
tradition, in historical documents, and because forced alignment of a trusted
transcription to an image is the nearest methodological cousin to our positional
aligner. Worth reading in full.

## Theme 6 — Reliability of automatically constructed ground truth

| Work | Venue / ID | Why it matters | Status |
|---|---|---|---|
| Northcutt, Athalye & Mueller | NeurIPS 2021 D&B, arXiv:2103.14749 | ≥3.3% label errors across ten standard test sets, enough to reorder model rankings | cited |

Gives our contamination finding a home in an existing conversation. The argument
now made in the paper: their errors are *scattered human mistakes*, ours is
*systematic* — correlated with exactly the node types that distinguish the methods
under test, so it survives averaging and grows with corpus size.

---

## Reading list, in priority order

1. **Detect-Order-Construct** (arXiv:2401.11874) — closest competing formulation;
   we must be able to say precisely how our task differs from Comp-HRDoc.
2. **HRDoc** (arXiv:2303.13839) — the dataset and task definition behind it.
3. **READoc** (arXiv:2409.05137) — rival evaluation philosophy for structured extraction.
4. **Docling technical report** (arXiv:2408.09869) — check how its provenance model
   compares with our `node -> set of regions`; possible baseline.
5. **Toselli et al.** (arXiv:2112.12703) — forced alignment as distant supervision.

## Open gaps

- No baseline comparison yet. Docling and MinerU both produce node-to-region
  correspondences and could be run on our corpus as a by-construction baseline
  against our two-view aligners. A referee will ask why this was not done.
- Nothing cited on document VQA evidence localisation or RAG citation grounding,
  which is the main *applied* motivation for wanting alignment at all.
- No survey citation for document layout analysis; one exists (ICCV workshops)
  and would economise several sentences.

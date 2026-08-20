#!/usr/bin/env python3
"""CLI for the PaddleOCR + XY-Cut++ document pipeline.

    ocr_venvs/env_paddle/bin/python layout_pipeline/main.py \
        --input_path arxiv_papers/bert.pdf --output_dir out/

Runs under env_paddle (the only venv with paddleocr installed). Model
construction dominates a small job: expect tens of seconds before the first
page is processed, then ~3 s/page.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow both `python layout_pipeline/main.py` and `python -m layout_pipeline.main`.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.ocr_engine import OCREngine
    from src.pipeline import DocumentPipeline, write_outputs
    from src.xycut_plus import XYCutConfig
else:  # pragma: no cover - import-style shim
    from .src.ocr_engine import OCREngine
    from .src.pipeline import DocumentPipeline, write_outputs
    from .src.xycut_plus import XYCutConfig


DOC_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff",
                  ".bmp", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract text with PaddleOCR and recover reading order "
                    "with XY-Cut++ (arXiv:2504.10258).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", nargs="?", default=None,
                   help="PDF, image, or directory of them (positional form of "
                        "--input_path)")
    p.add_argument("--input_path", default=None,
                   help="PDF, image, or directory of them")
    p.add_argument("--output_dir", default="out",
                   help="directory for the JSON / Markdown output")
    p.add_argument("--beta", type=float, default=1.3,
                   help="cross-layout width threshold scale (Eq. 1)")
    p.add_argument("--min_gap", type=float, default=1.0,
                   help="minimum whitespace in px that counts as a split gap")

    p.add_argument("--theta_v", type=float, default=0.9,
                   help="density threshold choosing XY-Cut vs YX-Cut (Eq. 5)")
    p.add_argument("--overlap_threshold", type=float, default=0.3,
                   help="minimum projection-IoU for an aligned anchor (Eq. 9)")
    p.add_argument("--format", default="json,md",
                   help="comma-separated outputs: json, md")
    p.add_argument("--dpi", type=int, default=200, help="PDF render resolution")
    p.add_argument("--device", default="gpu:0", help="paddle device, e.g. gpu:0 or cpu")
    p.add_argument("--batch_size", type=int, default=4,
                   help="pages per inference batch")
    p.add_argument("--layout_model", default="PP-DocLayoutV3",
                   help="PP-DocLayout model name")
    p.add_argument("--no_layout", action="store_true",
                   help="skip layout detection; order raw text lines only "
                        "(Phase 1 masking and Phase 4 priorities go unused)")
    p.add_argument("--mask_titles", action="store_true",
                   help="pre-mask titles in Phase 1 (the paper's behaviour). "
                        "Off by default: measured worse on 8 of 10 arXiv pages "
                        "(mean CER 0.152 vs 0.115) because Phase 4 can re-anchor "
                        "the title behind the author block.")
    p.add_argument("--widest_gap_axis", action="store_true",
                   help="choose the cut axis by widest whitespace (classic "
                        "XY-Cut) rather than by density alone. Helps side-by-side "
                        "rows on single-column pages; can mis-split two-column "
                        "bodies where a masked figure leaves a phantom gap.")
    p.add_argument("--paper_defaults", action="store_true",
                   help="reproduce arXiv:2504.10258 exactly: mask titles, enable "
                        "cross-layout masking, and choose the cut axis by density "
                        "alone. Measured worse on this corpus; provided for "
                        "faithful comparison.")
    p.add_argument("--cross_mask", action="store_true",
                   help="enable full-width spanner masking (Eq. 1-2). Off by "
                        "default: on academic papers it masks the centred "
                        "author/email lines and re-anchors the title after the "
                        "left column. Use it for newspaper-style layouts.")
    p.add_argument("--require_gpu", action="store_true",
                   help="fail rather than silently running on CPU")
    p.add_argument("--extract", action="store_true",
                   help="run the dual-extraction router: crop TABLE/FORMULA/VISION "
                        "regions into <output_dir>/crops/, publish figures into "
                        "<output_dir>/figures/, and write <stem>.blocks.json plus "
                        "<stem>.extracted.md")
    p.add_argument("--crop_padding", type=int, default=5,
                   help="pixels of padding around each cropped region")
    p.add_argument("--formula_model", default="PP-FormulaNet_plus-M",
                   help="local image-to-LaTeX model for FORMULA blocks")
    p.add_argument("--no_formula", action="store_true",
                   help="skip LaTeX extraction; formula blocks fall back to "
                        "their OCR text")
    p.add_argument("--per_file", action="store_true",
                   help="treat each file in a directory as its own document: "
                        "separate outputs, own title, own page numbering. "
                        "Without it a directory is projected as one merged "
                        "document, which is only right if the files really are "
                        "pages of one thing.")
    p.add_argument("--mode", choices=("structural", "semantic", "both"),
                   default="both",
                   help="which XML to produce. 'structural' = layout blocks -> "
                        "<stem>.reconstructed.xml. 'semantic' = local LLM "
                        "schema discovery -> <stem>.semantic.xml. 'both' does "
                        "each. 'structural' is the one that needs no LLM.")
    p.add_argument("--xml", action="store_true",
                   help="deprecated alias kept for older commands; --mode now "
                        "governs XML output and defaults to both")
    p.add_argument("--ie_model", default=None,
                   help="model for semantic extraction (default: the "
                        "LocalLLMClient default, llama3.3:70b)")
    p.add_argument("--ie_base_url", default=None,
                   help="inference endpoint for semantic extraction")
    p.add_argument("--doc_title", default=None,
                   help="title attribute for the XML root; defaults to the "
                        "document's own detected title")
    p.add_argument("--no_table", action="store_true",
                   help="skip table structure recognition; table blocks fall "
                        "back to their OCR text")
    p.add_argument("--export-db", "--export_db", dest="export_db",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="normalise the semantic tree into relational tables and "
                        "write <stem>.db plus a <stem>_tables/ CSV bundle. "
                        "Needs the semantic XML, so it is skipped under "
                        "--mode structural or when the LLM is unreachable")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # Accept both `main.py doc.pdf` and `main.py --input_path doc.pdf`.
    args.input_path = args.input_path or args.input
    if not args.input_path:
        print("an input path is required (positional or --input_path)",
              file=sys.stderr)
        return 2
    if not os.path.exists(args.input_path):
        print(f"input not found: {args.input_path}", file=sys.stderr)
        return 2

    config = XYCutConfig(
        beta=args.beta,
        theta_v=args.theta_v,
        min_gap_px=args.min_gap,
        overlap_threshold=args.overlap_threshold,
        mask_titles=args.mask_titles or args.paper_defaults,
        enable_cross_mask=args.cross_mask or args.paper_defaults,
        axis_by_widest_gap=args.widest_gap_axis and not args.paper_defaults,
    )
    engine = OCREngine(
        device=args.device,
        layout_model=args.layout_model,
        detect_layout=not args.no_layout,
        require_gpu=args.require_gpu,
    )
    pipeline = DocumentPipeline(engine=engine, config=config, dpi=args.dpi,
                               batch_size=args.batch_size)

    targets = [args.input_path]
    if args.per_file and os.path.isdir(args.input_path):
        targets = sorted(
            os.path.join(args.input_path, n)
            for n in os.listdir(args.input_path)
            if os.path.splitext(n)[1].lower() in DOC_EXTENSIONS)
        if not targets:
            print(f"no documents in {args.input_path}", file=sys.stderr)
            return 2
        print(f"{len(targets)} document(s), one output set each\n")

    formats = [f.strip() for f in args.format.split(",") if f.strip()]
    # One extractor for the whole run: it holds the LLM client, and rebuilding
    # it per document would re-probe the endpoint for every file.
    ie_extractor = None
    if args.mode in ("semantic", "both"):
        ie_extractor = build_ie_extractor(args)
    all_written = []
    for target in targets:
        stem = os.path.splitext(os.path.basename(target.rstrip("/")))[0] \
            or "document"
        # Block ids restart per document, so crops/page_0_block_5.png would
        # collide between files. One subdirectory per document keeps every
        # artifact — json, markdown, xml, crops, figures — namespaced.
        doc_out = os.path.join(args.output_dir, stem) if len(targets) > 1 \
            else args.output_dir
        pages = pipeline.run(target)
        written = write_outputs(pages, doc_out, config, stem=stem,
                                formats=formats)

        n_blocks = sum(len(p.blocks) for p in pages)
        n_lines = sum(len(b.lines) for p in pages for b in p.blocks)
        label = f"{stem}: " if len(targets) > 1 else ""
        print(f"{label}{len(pages)} page(s), {n_blocks} ordered blocks, "
              f"{n_lines} text lines")

        if args.extract or args.xml or args.mode:
            written += run_extraction(pages, doc_out, stem,
                                      args.crop_padding,
                                      formula_model=args.formula_model,
                                      no_formula=args.no_formula,
                                      no_table=args.no_table,
                                      device=args.device,
                                      want_xml=args.xml,
                                      doc_title=args.doc_title,
                                      mode=args.mode,
                                      ie_extractor=ie_extractor,
                                      export_db=args.export_db)
        all_written += written

    for path in all_written:
        print(f"  wrote {path}")
    return 0


def build_ie_extractor(args):
    """Construct the semantic extractor, or None if its endpoint is unreachable.

    Probed once here rather than failing per document: a missing LLM should
    degrade the run to structural-only with one clear message, not emit ten
    empty semantic files.
    """
    if __package__ in (None, ""):
        from src.ie_engine import DynamicInformationExtractor, LocalLLMClient
    else:  # pragma: no cover
        from .src.ie_engine import DynamicInformationExtractor, LocalLLMClient

    kwargs = {}
    if args.ie_model:
        kwargs["model"] = args.ie_model
    if args.ie_base_url:
        kwargs["base_url"] = args.ie_base_url
    client = LocalLLMClient(**kwargs)
    if not client.is_available():
        print(f"  semantic extraction disabled: {client.last_error}",
              file=sys.stderr)
        return None
    print(f"  semantic extraction via {client.model} at {client.base_url}")
    return DynamicInformationExtractor(client=client)


def run_extraction(pages, output_dir, stem, crop_padding,
                   formula_model=None, no_formula=False, no_table=False,
                   device="gpu:0", want_xml=False, doc_title=None,
                   mode="both", ie_extractor=None, export_db=True):
    """Route every page through the dual-extraction engine and write its output."""
    if __package__ in (None, ""):
        from src.dual_extractor import BlockType, DualExtractionRouter
        from src.ie_engine import DynamicInformationExtractor, RelationalExporter
        from src.xml_projector import XMLProjector, infer_title
    else:  # pragma: no cover
        from .src.dual_extractor import BlockType, DualExtractionRouter
        from .src.ie_engine import (DynamicInformationExtractor,
                                    RelationalExporter)
        from .src.xml_projector import XMLProjector, infer_title

    router = DualExtractionRouter(crop_padding=crop_padding,
                                  formula_extractor=False if no_formula else None,
                                  table_extractor=False if no_table else None,
                                  formula_model=formula_model, device=device)
    routed = router.route_pages([(p, p.image) for p in pages],
                                output_dir=output_dir)

    counts, statuses = {}, {}
    for blocks in routed:
        for b in blocks:
            counts[b.block_type.value] = counts.get(b.block_type.value, 0) + 1
            s = b.metadata.get("status", "?")
            statuses[s] = statuses.get(s, 0) + 1
    print("  routed: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print("  status: " + ", ".join(f"{k} {v}" for k, v in sorted(statuses.items())))

    blocks_path = os.path.join(output_dir, f"{stem}.blocks.json")
    with open(blocks_path, "w", encoding="utf-8") as fh:
        json.dump({
            "n_pages": len(routed),
            "pages": [{"page": i, "blocks": [b.to_dict() for b in bl]}
                      for i, bl in enumerate(routed)],
        }, fh, indent=2, ensure_ascii=False)

    md_path = os.path.join(output_dir, f"{stem}.extracted.md")
    chunks = []
    for page, blocks in zip(pages, routed):
        if len(pages) > 1:
            chunks.append(f"<!-- {page.source} page {page.index + 1} -->")
        for b in blocks:
            if not b.parsed_content:
                continue
            if b.block_type is BlockType.TITLE:
                depth = "#" if b.metadata.get("label") == "doc_title" else "##"
                chunks.append(f"{depth} {b.parsed_content}")
            else:
                chunks.append(b.parsed_content)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(chunks) + "\n")
    written = [blocks_path, md_path]

    flat = [b for page in routed for b in page]
    projector = XMLProjector()
    title = doc_title or infer_title(flat, default=stem)

    if mode in ("structural", "both") or want_xml:
        xml_path = os.path.join(output_dir, f"{stem}.reconstructed.xml")
        projector.save_xml(projector.project_to_xml(flat, doc_title=title),
                           xml_path)
        written.append(xml_path)

    if mode in ("semantic", "both") and ie_extractor is not None:
        doc = ie_extractor.extract(flat, source=f"{stem}.pdf")
        status = doc.metadata.get("status")
        print(f"  semantic: {status}, {doc.elements} element(s), "
              f"root <{doc.root.tag_name}>")
        if status not in ("extracted", "partial"):
            # Say why rather than leaving an empty file to be discovered later.
            print(f"    reason: {doc.metadata.get('reason')}")
        sem_path = os.path.join(output_dir, f"{stem}.semantic.xml")
        projector.save_xml(
            projector.project_dynamic_xml(doc.root, doc_title=title,
                                          source=f"{stem}.pdf"), sem_path)
        written.append(sem_path)

        if export_db:
            # Export from the saved XML, not from doc.root, so the database
            # holds exactly what the .semantic.xml holds — including the title
            # and source attributes the projector adds to the root.
            exporter = RelationalExporter(sem_path)
            db_path = os.path.join(output_dir, f"{stem}.db")
            csv_dir = os.path.join(output_dir, f"{stem}_tables")
            written.append(exporter.to_sqlite(db_path))
            written += exporter.to_csv_bundle(csv_dir)
            print(f"  relational: {len(exporter.tables)} table(s) -> "
                  f"{exporter.summary()}")
    return written


if __name__ == "__main__":
    sys.exit(main())

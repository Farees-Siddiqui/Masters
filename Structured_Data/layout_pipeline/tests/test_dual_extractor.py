"""Tests for the Dual Extraction Engine router (step 1).

Pure data plumbing: no paddle, no GPU, no models. Blocks are mocked as the
reading-ordered output of XY-Cut++.
"""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dual_extractor import (BlockType, DualExtractionRouter,  # noqa: E402
                                SemanticBlock)
from src.ocr_engine import LayoutBlock, TextLine  # noqa: E402
from src.pipeline import OrderedBlock  # noqa: E402


def line(text, x0=100, y0=200, x1=400, y1=230, score=0.95):
    return TextLine(text=text, bbox=[x0, y0, x1, y1], score=score)


def blk(label, order=0, texts=(), bbox=(80, 200, 480, 400), **kw):
    """A reading-ordered block, as XY-Cut++ hands them over."""
    return OrderedBlock(
        order=order, label=label, bbox=list(bbox),
        lines=[line(t, y0=200 + 40 * i, y1=230 + 40 * i)
               for i, t in enumerate(texts)],
        **kw,
    )


class Classification(unittest.TestCase):
    def setUp(self):
        self.r = DualExtractionRouter()

    def test_known_labels_map_to_expected_types(self):
        cases = {
            "doc_title": BlockType.TITLE,
            "paragraph_title": BlockType.TITLE,
            "text": BlockType.TEXT,
            "abstract": BlockType.TEXT,
            "table": BlockType.TABLE,
            "formula": BlockType.FORMULA,
            "display_formula": BlockType.FORMULA,
            "image": BlockType.VISION,
            "chart": BlockType.VISION,
        }
        for label, expected in cases.items():
            self.assertEqual(self.r.classify(label), expected, f"label={label}")

    def test_captions_are_prose_not_headings(self):
        """figure_title is a caption; rendering it as a heading would be wrong."""
        self.assertEqual(self.r.classify("figure_title"), BlockType.TEXT)
        self.assertEqual(self.r.classify("table_title"), BlockType.TEXT)

    def test_page_furniture_routes_as_text(self):
        for label in ("aside_text", "header", "footer", "footnote", "number"):
            self.assertEqual(self.r.classify(label), BlockType.TEXT, label)

    def test_unknown_none_and_blank_labels(self):
        for label in (None, "", "   ", "some_future_label"):
            self.assertEqual(self.r.classify(label), BlockType.UNKNOWN, repr(label))

    def test_label_matching_is_case_insensitive(self):
        self.assertEqual(self.r.classify("Table"), BlockType.TABLE)
        self.assertEqual(self.r.classify("DOC_TITLE"), BlockType.TITLE)

    def test_custom_label_map(self):
        r = DualExtractionRouter(label_map={"weird": BlockType.TABLE})
        self.assertEqual(r.classify("weird"), BlockType.TABLE)
        self.assertEqual(r.classify("text"), BlockType.UNKNOWN)


class Routing(unittest.TestCase):
    """Requirement 1: each label reaches the right handler."""

    def setUp(self):
        self.r = DualExtractionRouter()

    def _handler_called(self, label):
        target = self.r.handler_for(self.r.classify(label))
        with mock.patch.object(DualExtractionRouter, target.__name__,
                               autospec=True,
                               return_value={"content": "x", "metadata": {}}) as spy:
            self.r.route([blk(label, texts=("a",))])
        self.assertEqual(spy.call_count, 1, f"{target.__name__} not called for {label}")
        return target.__name__

    def test_table_reaches_the_table_handler(self):
        self.assertEqual(self._handler_called("table"), "_handle_table")

    def test_formula_reaches_the_formula_handler(self):
        self.assertEqual(self._handler_called("display_formula"), "_handle_formula")

    def test_vision_reaches_the_vision_handler(self):
        self.assertEqual(self._handler_called("image"), "_handle_vision")

    def test_text_reaches_the_text_handler(self):
        self.assertEqual(self._handler_called("text"), "_handle_text")

    def test_title_reaches_the_text_handler_but_keeps_its_type(self):
        """There is no _handle_title: a heading is extracted as text, and the
        TITLE type carries the rendering distinction downstream."""
        self.assertEqual(self._handler_called("doc_title"), "_handle_text")
        out = self.r.route([blk("doc_title", texts=("A Title",))])
        self.assertEqual(out[0].block_type, BlockType.TITLE)
        self.assertEqual(out[0].parsed_content, "A Title")

    def test_each_block_hits_exactly_one_handler(self):
        blocks = [blk("text", 0, ("t",)), blk("table", 1, ("tb",)),
                  blk("display_formula", 2, ("f",)), blk("image", 3)]
        with mock.patch.object(DualExtractionRouter, "_handle_text", autospec=True,
                               return_value={"content": "", "metadata": {}}) as t, \
             mock.patch.object(DualExtractionRouter, "_handle_table", autospec=True,
                               return_value={"content": "", "metadata": {}}) as tb, \
             mock.patch.object(DualExtractionRouter, "_handle_formula", autospec=True,
                               return_value={"content": "", "metadata": {}}) as f, \
             mock.patch.object(DualExtractionRouter, "_handle_vision", autospec=True,
                               return_value={"content": "", "metadata": {}}) as v:
            self.r.route(blocks)
        self.assertEqual((t.call_count, tb.call_count, f.call_count, v.call_count),
                         (1, 1, 1, 1))

    def test_handler_name_is_recorded_in_metadata(self):
        out = self.r.route([blk("table", texts=("cell",))])
        self.assertEqual(out[0].metadata["handler"], "_handle_table")


class FallbackBehaviour(unittest.TestCase):
    """Requirement 3: unlabeled / ambiguous blocks fall back to _handle_text."""

    def setUp(self):
        self.r = DualExtractionRouter()

    def test_unlabeled_block_falls_back_to_text(self):
        with mock.patch.object(DualExtractionRouter, "_handle_text", autospec=True,
                               return_value={"content": "", "metadata": {}}) as spy:
            self.r.route([blk(None, texts=("orphan text",))])
        self.assertEqual(spy.call_count, 1)

    def test_unknown_label_falls_back_to_text_and_keeps_content(self):
        out = self.r.route([blk("brand_new_label", texts=("line one", "line two"))])
        self.assertEqual(out[0].block_type, BlockType.UNKNOWN)
        self.assertEqual(out[0].metadata["handler"], "_handle_text")
        self.assertEqual(out[0].parsed_content, "line one\nline two")

    def test_fallback_never_loses_text(self):
        """The whole point of defaulting to text: content survives."""
        out = self.r.route([blk(None, texts=("keep", "me"))])
        self.assertIn("keep", out[0].parsed_content)
        self.assertIn("me", out[0].parsed_content)

    def test_block_with_no_lines_is_handled(self):
        out = self.r.route([blk("text")])
        self.assertEqual(out[0].parsed_content, "")
        self.assertEqual(out[0].metadata["n_lines"], 0)

    def test_empty_input(self):
        self.assertEqual(self.r.route([]), [])

    def test_plain_dicts_route_too(self):
        """Duck-typing so callers are not forced to build OrderedBlocks."""
        out = self.r.route([{"label": "text", "bbox": [0, 0, 10, 10],
                             "lines": [{"text": "hi", "score": 0.9}]}])
        self.assertEqual(out[0].parsed_content, "hi")
        self.assertEqual(out[0].block_type, BlockType.TEXT)


class OrderPreservation(unittest.TestCase):
    """Requirement 2: reading order is strictly preserved."""

    def test_output_sequence_matches_input_sequence(self):
        labels = ["doc_title", "text", "table", "text", "display_formula",
                  "image", "footnote"]
        blocks = [blk(l, order=i, texts=(f"content {i}",))
                  for i, l in enumerate(labels)]
        out = DualExtractionRouter().route(blocks)
        self.assertEqual(len(out), len(blocks))
        self.assertEqual([b.metadata["reading_position"] for b in out],
                         list(range(len(labels))))
        self.assertEqual([b.metadata["label"] for b in out], labels)

    def test_order_is_preserved_even_when_types_interleave(self):
        blocks = [blk("table", 0, ("T0",)), blk("text", 1, ("X1",)),
                  blk("table", 2, ("T2",)), blk("text", 3, ("X3",))]
        out = DualExtractionRouter().route(blocks)
        self.assertEqual([b.raw_lines[0].text for b in out],
                         ["T0", "X1", "T2", "X3"])

    def test_router_does_not_resort_a_misordered_input(self):
        """Re-sorting here would mask an upstream ordering bug."""
        blocks = [blk("text", order=5, texts=("later",)),
                  blk("text", order=1, texts=("earlier",))]
        out = DualExtractionRouter().route(blocks)
        self.assertEqual([b.metadata["source_order"] for b in out], [5, 1])
        self.assertEqual(out[0].parsed_content, "later")

    def test_block_ids_are_sequential_and_unique(self):
        out = DualExtractionRouter().route([blk("text", i) for i in range(5)])
        self.assertEqual([b.block_id for b in out], [0, 1, 2, 3, 4])

    def test_ids_stay_unique_across_pages(self):
        r = DualExtractionRouter()
        p1 = r.route([blk("text", 0), blk("text", 1)], page_number=0)
        p2 = r.route([blk("text", 0), blk("text", 1)], page_number=1)
        ids = [b.block_id for b in p1 + p2]
        self.assertEqual(ids, [0, 1, 2, 3])
        self.assertEqual([b.metadata["page"] for b in p2], [1, 1])

    def test_reset_restarts_ids(self):
        r = DualExtractionRouter()
        r.route([blk("text")])
        r.reset()
        self.assertEqual(r.route([blk("text")])[0].block_id, 0)


class Stubs(unittest.TestCase):
    """The specialised handlers are placeholders, and say so."""

    def setUp(self):
        self.r = DualExtractionRouter()

    def test_table_without_a_page_image_falls_back(self):
        """Step 4: a table is no longer a stub either. With no pixels it
        degrades to the OCR text and says so."""
        out = DualExtractionRouter(table_extractor=False).route(
            [blk("table", texts=("a b c",))])[0]
        self.assertEqual(out.metadata["status"], "fallback")
        self.assertIn("WARNING: Table extraction failed", out.parsed_content)
        self.assertIn("a b c", out.parsed_content)

    def test_vision_without_a_page_image_stays_pending(self):
        out = self.r.route([blk("image")])[0]
        self.assertEqual(out.metadata["status"], "pending")
        self.assertIn("VISION", out.parsed_content)

    def test_formula_without_a_page_image_falls_back(self):
        """Step 3: a formula is no longer a stub. With no pixels to run the
        model on it degrades to the OCR text and says so."""
        out = DualExtractionRouter(formula_extractor=False).route(
            [blk("display_formula", texts=("x = 1",))])[0]
        self.assertEqual(out.metadata["status"], "fallback")
        self.assertIn("WARNING: LaTeX extraction failed", out.parsed_content)
        self.assertIn("x = 1", out.parsed_content)

    def test_text_handler_is_not_marked_pending(self):
        out = self.r.route([blk("text", texts=("real prose",))])[0]
        self.assertEqual(out.metadata["status"], "extracted")
        self.assertEqual(out.parsed_content, "real prose")

    def test_stub_keeps_ocr_text_so_content_is_not_lost(self):
        out = self.r.route([blk("table", texts=("row one", "row two"))])[0]
        self.assertEqual(out.metadata["ocr_fallback_text"], "row one\nrow two")
        self.assertIn("row one", out.parsed_content)

    def test_no_page_image_means_no_crop(self):
        """Without pixels there is nothing to crop, and the vision handler must
        not emit a Markdown tag pointing at a file that was never written."""
        out = self.r.route([blk("image", bbox=(10, 20, 110, 220))])[0]
        self.assertIsNone(out.crop_path)
        self.assertEqual(out.metadata["status"], "pending")
        self.assertEqual(out.metadata["crop_bbox"], [10, 20, 110, 220])
        self.assertNotIn("![Figure]", out.parsed_content)


class Model(unittest.TestCase):
    def test_semantic_block_fields(self):
        b = SemanticBlock(block_id=3, block_type=BlockType.TEXT,
                          bbox=[1.0, 2.0, 3.0, 4.0])
        self.assertEqual((b.block_id, b.block_type), (3, BlockType.TEXT))
        self.assertEqual(b.raw_lines, [])
        self.assertEqual(b.parsed_content, "")
        self.assertEqual(b.metadata, {})

    def test_block_type_serialises_as_a_plain_string(self):
        self.assertEqual(BlockType.TABLE.value, "TABLE")
        self.assertEqual(json.dumps({"t": BlockType.TABLE}), '{"t": "TABLE"}')

    def test_to_dict_is_json_serialisable(self):
        out = DualExtractionRouter().route([blk("text", texts=("hello",))])[0]
        payload = json.dumps(out.to_dict())
        self.assertIn("hello", payload)
        self.assertIn('"block_type": "TEXT"', payload)

    def test_metadata_carries_confidence_and_provenance(self):
        out = DualExtractionRouter().route([blk("text", texts=("a", "b"))],
                                           page_number=7)[0]
        self.assertEqual(out.metadata["page"], 7)
        self.assertEqual(out.metadata["n_lines"], 2)
        self.assertAlmostEqual(out.metadata["mean_line_confidence"], 0.95)

    def test_marginal_blocks_are_flagged(self):
        out = DualExtractionRouter().route([blk("aside_text", texts=("stamp",))])[0]
        self.assertTrue(out.metadata["is_marginal"])
        self.assertFalse(
            DualExtractionRouter().route([blk("text", texts=("x",))])[0]
            .metadata["is_marginal"])

    def test_synthetic_blocks_are_flagged(self):
        b = blk("text", texts=("orphan",), synthetic=True)
        self.assertTrue(DualExtractionRouter().route([b])[0].metadata["synthetic"])

    def test_raw_lines_are_preserved_on_the_block(self):
        out = DualExtractionRouter().route([blk("text", texts=("a", "b"))])[0]
        self.assertEqual(len(out.raw_lines), 2)
        self.assertIsInstance(out.raw_lines[0], TextLine)


class RoutePages(unittest.TestCase):
    def test_route_pages_returns_one_list_per_page(self):
        from src.pipeline import OrderedPage

        pages = [
            OrderedPage("d.pdf", 0, 1000, 1400, [blk("text", 0, ("p0",))]),
            OrderedPage("d.pdf", 1, 1000, 1400,
                        [blk("text", 0, ("p1a",)), blk("table", 1, ("p1b",))]),
        ]
        out = DualExtractionRouter().route_pages(pages)
        self.assertEqual([len(p) for p in out], [1, 2])
        self.assertEqual([b.block_id for p in out for b in p], [0, 1, 2])
        self.assertEqual([b.metadata["page"] for p in out for b in p], [0, 1, 1])


if __name__ == "__main__":
    unittest.main()

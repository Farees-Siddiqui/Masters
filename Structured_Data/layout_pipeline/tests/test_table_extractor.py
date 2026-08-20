"""Tests for table structure extraction and the table handler (step 4).

Backends are mocked with HTML strings, so nothing here needs paddle, model
weights or a GPU.
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

from src.dual_extractor import DualExtractionRouter  # noqa: E402
from src.extractors.table_extractor import (TableExtractor,  # noqa: E402
                                            clean_html, has_merged_cells,
                                            rows_to_markdown)
from src.ocr_engine import TextLine  # noqa: E402
from src.pipeline import OrderedBlock  # noqa: E402

PAGE_W, PAGE_H = 500, 400

SIMPLE = ("<html><body><table><thead><tr><td>Model</td><td>Top-1</td></tr>"
          "</thead><tbody><tr><td>ResNet-50</td><td>76.1</td></tr>"
          "<tr><td>DenseNet-121</td><td>75.0</td></tr></tbody></table>"
          "</body></html>")

MERGED = ('<html><body><table><thead><tr><td>Layers</td>'
          '<td colspan="2">DenseNet-121</td></tr></thead>'
          '<tr><td>Convolution</td><td colspan="2">7x7 conv</td></tr>'
          '</table></body></html>')


def page():
    return Image.new("RGB", (PAGE_W, PAGE_H), (255, 255, 255))


def blk(label="table", bbox=(50, 50, 450, 350), texts=()):
    return OrderedBlock(order=0, label=label, bbox=list(bbox),
                        lines=[TextLine(text=t, bbox=[60, 60, 400, 90], score=0.9)
                               for t in texts])


def fixed_backend(html):
    def backend(image):
        backend.calls.append(image)
        return html
    backend.calls = []
    backend.name = "fake"
    return backend


def raising_backend(exc):
    def backend(image):
        raise exc
    backend.name = "raising"
    return backend


class HtmlHelpers(unittest.TestCase):
    def test_clean_html_extracts_the_table_element(self):
        self.assertTrue(clean_html(SIMPLE).startswith("<table>"))
        self.assertTrue(clean_html(SIMPLE).endswith("</table>"))
        self.assertNotIn("<body>", clean_html(SIMPLE))

    def test_clean_html_passes_through_bare_tables(self):
        bare = "<table><tr><td>x</td></tr></table>"
        self.assertEqual(clean_html(bare), bare)

    def test_merged_cell_detection(self):
        self.assertFalse(has_merged_cells([[{"colspan": 1, "rowspan": 1}]]))
        self.assertTrue(has_merged_cells([[{"colspan": 2, "rowspan": 1}]]))
        self.assertTrue(has_merged_cells([[{"colspan": 1, "rowspan": 3}]]))

    def test_rows_to_markdown_shape(self):
        rows = [[{"text": "a"}, {"text": "b"}], [{"text": "1"}, {"text": "2"}]]
        md = rows_to_markdown(rows)
        self.assertEqual(md.splitlines(), ["| a | b |", "|---|---|", "| 1 | 2 |"])

    def test_pipes_in_cells_are_escaped(self):
        md = rows_to_markdown([[{"text": "a|b"}], [{"text": "c"}]])
        self.assertIn(r"a\|b", md)

    def test_ragged_rows_are_padded(self):
        rows = [[{"text": "a"}, {"text": "b"}], [{"text": "1"}]]
        lines = rows_to_markdown(rows).splitlines()
        self.assertEqual(lines[0].count("|"), lines[2].count("|"))


class Extractor(unittest.TestCase):
    """Requirement 1: image -> table string, no live weights."""

    def test_simple_table_becomes_markdown(self):
        r = TableExtractor(backend=fixed_backend(SIMPLE)).extract_table(page())
        self.assertEqual(r["format"], "markdown")
        self.assertIn("| Model | Top-1 |", r["content"])
        self.assertIn("| ResNet-50 | 76.1 |", r["content"])
        self.assertEqual((r["rows"], r["columns"]), (3, 2))

    def test_merged_cells_stay_html(self):
        """Markdown has no colspan; flattening would misstate the data."""
        r = TableExtractor(backend=fixed_backend(MERGED)).extract_table(page())
        self.assertEqual(r["format"], "html")
        self.assertTrue(r["merged_cells"])
        self.assertIn('colspan="2"', r["content"])
        self.assertTrue(r["content"].startswith("<table>"))

    def test_backend_receives_the_image(self):
        b = fixed_backend(SIMPLE)
        img = page()
        TableExtractor(backend=b).extract_table(img)
        self.assertIs(b.calls[0], img)

    def test_prefer_markdown_false_keeps_html(self):
        r = TableExtractor(backend=fixed_backend(SIMPLE),
                           prefer_markdown=False).extract_table(page())
        self.assertEqual(r["format"], "html")

    def test_none_image_returns_none(self):
        e = TableExtractor(backend=fixed_backend(SIMPLE))
        self.assertIsNone(e.extract_table(None))
        self.assertIn("no crop", e.last_error)

    def test_tiny_crop_returns_none(self):
        e = TableExtractor(backend=fixed_backend(SIMPLE))
        self.assertIsNone(e.extract_table(Image.new("RGB", (5, 5))))
        self.assertIn("too small", e.last_error)

    def test_empty_markup_returns_none(self):
        e = TableExtractor(backend=fixed_backend("   "))
        self.assertIsNone(e.extract_table(page()))
        self.assertIn("no markup", e.last_error)

    def test_table_without_cell_text_returns_none(self):
        e = TableExtractor(backend=fixed_backend(
            "<table><tr><td></td><td></td></tr></table>"))
        self.assertIsNone(e.extract_table(page()))
        self.assertIn("no cell text", e.last_error)

    def test_backend_exception_returns_none_not_raise(self):
        e = TableExtractor(backend=raising_backend(RuntimeError("CUDA oom")))
        self.assertIsNone(e.extract_table(page()))
        self.assertIn("CUDA oom", e.last_error)

    def test_missing_dependency_is_not_retried(self):
        calls = []

        def backend(image):
            calls.append(image)
            raise ImportError("no paddleocr")

        e = TableExtractor(backend=backend)
        for _ in range(4):
            self.assertIsNone(e.extract_table(page()))
        self.assertEqual(len(calls), 1)

    def test_default_backend_is_paddle_and_lazy(self):
        e = TableExtractor()
        self.assertEqual(e.backend.name, "paddle:TableRecognitionPipelineV2")
        self.assertIsNone(e.backend._pipeline)

    def test_entities_are_decoded(self):
        e = TableExtractor(backend=fixed_backend(
            "<table><tr><td>a &amp; b</td></tr><tr><td>c</td></tr></table>"))
        self.assertIn("a & b", e.extract_table(page())["content"])


class TableHandler(unittest.TestCase):
    """Requirement 2: status transitions and format handling."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _route(self, backend, texts=("noise",)):
        r = DualExtractionRouter(
            table_extractor=TableExtractor(backend=backend),
            formula_extractor=False)
        return r.route([blk(texts=texts)], page_image=page(),
                       output_dir=self.tmp)[0]

    def test_markdown_table_completes(self):
        out = self._route(fixed_backend(SIMPLE))
        self.assertEqual(out.status, "completed")
        self.assertEqual(out.metadata["table_format"], "markdown")
        self.assertIn("| Model | Top-1 |", out.parsed_content)
        self.assertNotIn("WARNING", out.parsed_content)

    def test_html_table_completes(self):
        out = self._route(fixed_backend(MERGED))
        self.assertEqual(out.status, "completed")
        self.assertEqual(out.metadata["table_format"], "html")
        self.assertTrue(out.metadata["merged_cells"])
        self.assertIn("<table>", out.parsed_content)

    def test_dimensions_are_recorded(self):
        out = self._route(fixed_backend(SIMPLE))
        self.assertEqual((out.metadata["table_rows"],
                          out.metadata["table_columns"]), (3, 2))

    def test_crop_is_still_written(self):
        out = self._route(fixed_backend(SIMPLE))
        self.assertTrue(os.path.isfile(out.crop_path))

    def test_failure_falls_back_with_warning_and_ocr_text(self):
        out = self._route(raising_backend(RuntimeError("boom")),
                          texts=("Layers 112 x 112",))
        self.assertEqual(out.status, "fallback")
        self.assertIn("<!-- WARNING: Table extraction failed -->",
                      out.parsed_content)
        self.assertIn("Layers 112 x 112", out.parsed_content)
        self.assertIn("boom", out.metadata["table_error"])
        self.assertIsNone(out.metadata["table_format"])

    def test_empty_result_falls_back(self):
        out = self._route(fixed_backend(""))
        self.assertEqual(out.status, "fallback")

    def test_disabled_extractor_falls_back(self):
        r = DualExtractionRouter(table_extractor=False, formula_extractor=False)
        out = r.route([blk(texts=("x",))], page_image=page(),
                      output_dir=self.tmp)[0]
        self.assertEqual(out.status, "fallback")
        self.assertEqual(out.metadata["table_error"], "table extraction disabled")

    def test_no_page_image_falls_back(self):
        r = DualExtractionRouter(
            table_extractor=TableExtractor(backend=fixed_backend(SIMPLE)))
        out = r.route([blk(texts=("x",))])[0]
        self.assertEqual(out.status, "fallback")
        self.assertIsNone(out.crop_path)

    def test_ocr_text_kept_on_success(self):
        out = self._route(fixed_backend(SIMPLE), texts=("garbled",))
        self.assertEqual(out.metadata["ocr_fallback_text"], "garbled")

    def test_handler_sees_the_crop_not_the_page(self):
        b = fixed_backend(SIMPLE)
        self._route(b)
        self.assertEqual(b.calls[0].size, (410, 310))  # 400x300 box + 5px pad

    def test_only_table_blocks_reach_the_extractor(self):
        b = fixed_backend(SIMPLE)
        r = DualExtractionRouter(table_extractor=TableExtractor(backend=b),
                                 formula_extractor=False)
        r.route([blk("text", texts=("prose",)), blk("image"),
                 blk("display_formula")],
                page_image=page(), output_dir=self.tmp)
        self.assertEqual(len(b.calls), 0)

    def test_table_and_formula_extractors_are_independent(self):
        tb = fixed_backend(SIMPLE)
        r = DualExtractionRouter(table_extractor=TableExtractor(backend=tb),
                                 formula_extractor=False)
        out = r.route([blk("table"), blk("display_formula", texts=("x",))],
                      page_image=page(), output_dir=self.tmp)
        self.assertEqual(out[0].status, "completed")
        self.assertEqual(out[1].status, "fallback")



class EmptyGridGuard(unittest.TestCase):
    """A grid with no cell text must not report success.

    Structure recognition can find a correct grid and attach nothing to it --
    observed on the DenseNet ImageNet-results table, 1 filled cell of 9 from a
    perfectly legible crop.
    """

    EMPTY_BODY = ("<table><tr><td>Model</td><td>top-1</td><td>top-5</td></tr>"
                  "<tr><td></td><td>7.71</td><td></td></tr>"
                  "<tr><td></td><td></td><td></td></tr>"
                  "<tr><td></td><td></td><td></td></tr></table>")

    def test_mostly_empty_body_is_rejected(self):
        e = TableExtractor(backend=fixed_backend(self.EMPTY_BODY))
        self.assertIsNone(e.extract_table(page()))
        self.assertIn("body cells", e.last_error)

    def test_well_filled_table_passes(self):
        e = TableExtractor(backend=fixed_backend(SIMPLE))
        self.assertIsNotNone(e.extract_table(page()))
        self.assertEqual(e.extract_table(page())["cell_fill"], 1.0)

    def test_threshold_is_configurable(self):
        e = TableExtractor(backend=fixed_backend(self.EMPTY_BODY),
                           min_cell_fill=0.0)
        self.assertIsNotNone(e.extract_table(page()))

    def test_empty_grid_routes_to_fallback_not_completed(self):
        tmp = tempfile.mkdtemp()
        try:
            r = DualExtractionRouter(
                table_extractor=TableExtractor(backend=fixed_backend(self.EMPTY_BODY)),
                formula_extractor=False)
            out = r.route([blk(texts=("DenseNet-121 25.02 / 23.61",))],
                          page_image=page(), output_dir=tmp)[0]
            self.assertEqual(out.status, "fallback")
            self.assertIn("DenseNet-121", out.parsed_content)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_fill_ratio_helper(self):
        from src.extractors.table_extractor import body_fill_ratio
        rows = [[{"text": "h"}], [{"text": "a"}], [{"text": ""}]]
        self.assertEqual(body_fill_ratio(rows), 0.5)
        self.assertEqual(body_fill_ratio([]), 0.0)

class BackendConfig(unittest.TestCase):
    def test_layout_detection_is_off_by_default(self):
        """Regression: the input is already a table crop.

        Re-running layout detection on it made the pipeline find no table at
        all -- 6 of 6 borderless tables in the StudentRecord corpus returned
        empty markup, and all 6 extract cleanly with it disabled.
        """
        from src.extractors.table_extractor import PaddleTableBackend
        self.assertFalse(PaddleTableBackend().use_layout_detection)

    def test_layout_detection_can_be_re_enabled(self):
        from src.extractors.table_extractor import PaddleTableBackend
        self.assertTrue(
            PaddleTableBackend(use_layout_detection=True).use_layout_detection)


if __name__ == "__main__":
    unittest.main()

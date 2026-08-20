"""Tests for image-to-LaTeX extraction and the formula handler (step 3).

Every test injects a fake backend, so the suite needs no model weights, no
paddle and no GPU -- the point of the backend abstraction.
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

from src.dual_extractor import (BlockType, DualExtractionRouter,  # noqa: E402
                                RouteContext)
from src.extractors.formula_extractor import (FormulaExtractor,  # noqa: E402
                                              strip_delimiters)
from src.ocr_engine import TextLine  # noqa: E402
from src.pipeline import OrderedBlock  # noqa: E402

PAGE_W, PAGE_H = 400, 600
LATEX = r"\Theta = \frac{1}{N}\sum_{i=1}^{N}\ell(x_i,\Theta)"


def page():
    return Image.new("RGB", (PAGE_W, PAGE_H), (255, 255, 255))


def blk(label, bbox=(100, 100, 300, 160), texts=()):
    return OrderedBlock(
        order=0, label=label, bbox=list(bbox),
        lines=[TextLine(text=t, bbox=[100, 100, 300, 130], score=0.9)
               for t in texts])


def fixed_backend(value):
    """A backend returning ``value`` and recording what it was given."""
    calls = []

    def backend(image):
        calls.append(image)
        return value

    backend.calls = calls
    backend.name = "fake"
    return backend


def raising_backend(exc):
    def backend(image):
        raise exc
    backend.name = "raising"
    return backend


class Delimiters(unittest.TestCase):
    """The router adds its own $$ fences, so the model's must come off."""

    def test_double_dollar_wrapper_removed(self):
        self.assertEqual(strip_delimiters("$$ x = 1 $$"), "x = 1")

    def test_single_dollar_wrapper_removed(self):
        self.assertEqual(strip_delimiters("$x = 1$"), "x = 1")

    def test_bracket_and_paren_wrappers_removed(self):
        self.assertEqual(strip_delimiters(r"\[ x = 1 \]"), "x = 1")
        self.assertEqual(strip_delimiters(r"\( x = 1 \)"), "x = 1")

    def test_equation_environment_removed(self):
        self.assertEqual(
            strip_delimiters(r"\begin{equation}x = 1\end{equation}"), "x = 1")

    def test_markdown_fence_removed(self):
        self.assertEqual(strip_delimiters("```latex\nx = 1\n```"), "x = 1")

    def test_nested_wrappers_removed(self):
        self.assertEqual(strip_delimiters(r"$$\[ x = 1 \]$$"), "x = 1")

    def test_inner_dollars_are_left_alone(self):
        """Only wrappers around the whole expression come off."""
        self.assertEqual(strip_delimiters(r"a $x$ b"), r"a $x$ b")

    def test_bare_expression_is_untouched(self):
        self.assertEqual(strip_delimiters(LATEX), LATEX)

    def test_empty_and_whitespace(self):
        self.assertEqual(strip_delimiters(""), "")
        self.assertEqual(strip_delimiters("   "), "")

    def test_delimiters_only_input_unwraps_to_leftovers(self):
        # Documented raw behaviour; extract_latex rejects it as contentless.
        self.assertEqual(strip_delimiters("$$$$"), "$$")


class Extractor(unittest.TestCase):
    """Requirement 1: image -> string contract, with no live weights."""

    def test_returns_backend_output(self):
        e = FormulaExtractor(backend=fixed_backend(LATEX))
        self.assertEqual(e.extract_latex(page()), LATEX)

    def test_backend_receives_the_image(self):
        b = fixed_backend(LATEX)
        img = page()
        FormulaExtractor(backend=b).extract_latex(img)
        self.assertEqual(len(b.calls), 1)
        self.assertIs(b.calls[0], img)

    def test_delimiters_are_stripped_from_backend_output(self):
        e = FormulaExtractor(backend=fixed_backend(f"$${LATEX}$$"))
        self.assertEqual(e.extract_latex(page()), LATEX)

    def test_none_image_returns_none(self):
        e = FormulaExtractor(backend=fixed_backend(LATEX))
        self.assertIsNone(e.extract_latex(None))
        self.assertIn("no crop", e.last_error)

    def test_tiny_crop_returns_none(self):
        e = FormulaExtractor(backend=fixed_backend(LATEX))
        self.assertIsNone(e.extract_latex(Image.new("RGB", (3, 3))))
        self.assertIn("too small", e.last_error)

    def test_empty_output_returns_none(self):
        e = FormulaExtractor(backend=fixed_backend("   "))
        self.assertIsNone(e.extract_latex(page()))
        self.assertIn("empty", e.last_error)

    def test_contentless_output_returns_none(self):
        """Stray delimiters are not an expression."""
        for junk in ("$$$$", "\\[\\]", "   $  $  "):
            e = FormulaExtractor(backend=fixed_backend(junk))
            self.assertIsNone(e.extract_latex(page()), junk)

    def test_non_string_output_returns_none(self):
        e = FormulaExtractor(backend=fixed_backend(42))
        self.assertIsNone(e.extract_latex(page()))
        self.assertIn("expected str", e.last_error)

    def test_backend_exception_returns_none_not_raise(self):
        e = FormulaExtractor(backend=raising_backend(RuntimeError("CUDA blew up")))
        self.assertIsNone(e.extract_latex(page()))
        self.assertIn("CUDA blew up", e.last_error)

    def test_missing_dependency_is_not_retried(self):
        """An ImportError repeats for every block, so it is latched once."""
        calls = []

        def backend(image):
            calls.append(image)
            raise ImportError("no module named paddleocr")

        e = FormulaExtractor(backend=backend)
        for _ in range(4):
            self.assertIsNone(e.extract_latex(page()))
        self.assertEqual(len(calls), 1, "backend should not be retried")

    def test_ordinary_failures_are_retried(self):
        """A transient failure on one crop must not disable the rest."""
        e = FormulaExtractor(backend=raising_backend(ValueError("bad crop")))
        for _ in range(3):
            e.extract_latex(page())
        self.assertFalse(e._failed)

    def test_unknown_backend_name_raises_at_construction(self):
        with self.assertRaises(ValueError):
            FormulaExtractor(backend_name="nonsense")

    def test_transformers_backend_requires_a_model_id(self):
        with self.assertRaises(ValueError):
            FormulaExtractor(backend_name="transformers")

    def test_default_backend_is_paddle_and_is_lazy(self):
        """Constructing must not import paddle -- only calling it does."""
        e = FormulaExtractor()
        self.assertEqual(e.backend.name, "paddle:PP-FormulaNet_plus-M")
        self.assertIsNone(e.backend._model)


class FormulaHandler(unittest.TestCase):
    """Requirement 2: display-math wrapping and graceful fallback."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _route(self, backend, texts=("noise",)):
        r = DualExtractionRouter(
            formula_extractor=FormulaExtractor(backend=backend))
        return r.route([blk("display_formula", texts=texts)],
                       page_image=page(), output_dir=self.tmp)[0]

    def test_latex_is_wrapped_in_display_delimiters(self):
        out = self._route(fixed_backend(LATEX))
        self.assertEqual(out.parsed_content, f"$$\n{LATEX}\n$$")

    def test_status_is_completed_on_success(self):
        out = self._route(fixed_backend(LATEX))
        self.assertEqual(out.status, "completed")
        self.assertEqual(out.metadata["latex"], LATEX)

    def test_model_delimiters_are_not_doubled(self):
        out = self._route(fixed_backend(f"$${LATEX}$$"))
        self.assertEqual(out.parsed_content.count("$$"), 2)

    def test_crop_is_still_written(self):
        out = self._route(fixed_backend(LATEX))
        self.assertTrue(os.path.isfile(out.crop_path))

    def test_failure_falls_back_to_ocr_text_with_a_warning(self):
        out = self._route(raising_backend(RuntimeError("boom")),
                          texts=("D N 1 sum l(xi,)",))
        self.assertIn("<!-- WARNING: LaTeX extraction failed -->",
                      out.parsed_content)
        self.assertIn("D N 1 sum l(xi,)", out.parsed_content)
        self.assertNotIn("$$", out.parsed_content)

    def test_failure_is_marked_fallback_not_completed(self):
        """A formula that degraded to OCR noise must not read as parsed."""
        out = self._route(raising_backend(RuntimeError("boom")))
        self.assertEqual(out.status, "fallback")
        self.assertIsNone(out.metadata["latex"])
        self.assertIn("boom", out.metadata["latex_error"])

    def test_empty_output_falls_back(self):
        out = self._route(fixed_backend(""))
        self.assertEqual(out.status, "fallback")

    def test_disabled_extractor_falls_back(self):
        r = DualExtractionRouter(formula_extractor=False)
        out = r.route([blk("display_formula", texts=("x",))],
                      page_image=page(), output_dir=self.tmp)[0]
        self.assertEqual(out.status, "fallback")
        self.assertEqual(out.metadata["latex_error"], "formula extraction disabled")

    def test_no_page_image_falls_back(self):
        r = DualExtractionRouter(
            formula_extractor=FormulaExtractor(backend=fixed_backend(LATEX)))
        out = r.route([blk("display_formula", texts=("x",))])[0]
        self.assertEqual(out.status, "fallback")
        self.assertIsNone(out.crop_path)

    def test_ocr_text_is_kept_even_on_success(self):
        out = self._route(fixed_backend(LATEX), texts=("garbled ocr",))
        self.assertEqual(out.metadata["ocr_fallback_text"], "garbled ocr")

    def test_handler_reads_the_crop_not_the_whole_page(self):
        b = fixed_backend(LATEX)
        out = self._route(b)
        self.assertEqual(b.calls[0].size, (210, 70))  # 200x60 box + 5px padding
        self.assertNotEqual(b.calls[0].size, (PAGE_W, PAGE_H))
        self.assertIsNotNone(out.crop_path)

    def test_only_formula_blocks_reach_the_extractor(self):
        b = fixed_backend(LATEX)
        r = DualExtractionRouter(formula_extractor=FormulaExtractor(backend=b))
        r.route([blk("text", texts=("prose",)), blk("image"), blk("table")],
                page_image=page(), output_dir=self.tmp)
        self.assertEqual(len(b.calls), 0)


class CropReuse(unittest.TestCase):
    def test_context_keeps_the_crop_in_memory(self):
        tmp = tempfile.mkdtemp()
        try:
            r = DualExtractionRouter(formula_extractor=False)
            ctx = RouteContext(block_type=BlockType.FORMULA, page_image=page(),
                               output_dir=tmp)
            r._make_crop(blk("display_formula"), ctx)
            self.assertIsNotNone(ctx.crop_image)
            self.assertEqual(ctx.crop_image.size, (210, 70))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_crop_is_reread_from_disk_when_not_in_memory(self):
        tmp = tempfile.mkdtemp()
        try:
            r = DualExtractionRouter(formula_extractor=False)
            path = os.path.join(tmp, "c.png")
            Image.new("RGB", (60, 40), (255, 255, 255)).save(path)
            ctx = RouteContext(crop_path=path)
            self.assertEqual(r._crop_for(ctx).size, (60, 40))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_crop_for_returns_none_without_either(self):
        r = DualExtractionRouter(formula_extractor=False)
        self.assertIsNone(r._crop_for(RouteContext()))


if __name__ == "__main__":
    unittest.main()

"""Tests for the cropping engine and the vision handler (step 2).

Synthetic page images, so no paddle, no GPU and no model download.
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

from src.crop_engine import CropEngine  # noqa: E402
from src.dual_extractor import (BlockType, DualExtractionRouter,  # noqa: E402
                                RouteContext)
from src.ocr_engine import TextLine  # noqa: E402
from src.pipeline import OrderedBlock  # noqa: E402

PAGE_W, PAGE_H = 400, 600


def page(colour=(255, 255, 255)):
    return Image.new("RGB", (PAGE_W, PAGE_H), colour)


def marked_page():
    """White page with a red square at (100,100)-(200,200), to prove the crop
    takes the region actually asked for."""
    img = page()
    for x in range(100, 200):
        for y in range(100, 200):
            img.putpixel((x, y), (255, 0, 0))
    return img


def line(text, x0=100, y0=200, x1=300, y1=230, score=0.9):
    return TextLine(text=text, bbox=[x0, y0, x1, y1], score=score)


def blk(label, bbox, texts=(), order=0):
    return OrderedBlock(order=order, label=label, bbox=list(bbox),
                        lines=[line(t) for t in texts])


class Padding(unittest.TestCase):
    def setUp(self):
        self.e = CropEngine()

    def test_default_padding_is_five_px_on_every_side(self):
        crop = self.e.crop_block(page(), [100, 100, 200, 200])
        self.assertEqual(crop.size, (110, 110))

    def test_explicit_padding_overrides_the_default(self):
        self.assertEqual(self.e.crop_block(page(), [100, 100, 200, 200], 0).size,
                         (100, 100))
        self.assertEqual(self.e.crop_block(page(), [100, 100, 200, 200], 20).size,
                         (140, 140))

    def test_engine_level_padding_is_used_by_clamp(self):
        e = CropEngine(padding=12)
        self.assertEqual(e.clamp_bbox([100, 100, 200, 200], PAGE_W, PAGE_H),
                         (88, 88, 212, 212))

    def test_crop_takes_the_requested_region(self):
        crop = self.e.crop_block(marked_page(), [100, 100, 200, 200], padding=0)
        # Every pixel of the unpadded crop is the red square.
        self.assertEqual(crop.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(crop.getpixel((99, 99)), (255, 0, 0))
        self.assertEqual(sorted(crop.getcolors())[0][1], (255, 0, 0))

    def test_fractional_coordinates_expand_outward(self):
        # floor the min, ceil the max, so nothing is clipped off.
        self.assertEqual(self.e.clamp_bbox([10.7, 20.2, 30.1, 40.9], PAGE_W, PAGE_H, 0),
                         (10, 20, 31, 41))


class Clamping(unittest.TestCase):
    def setUp(self):
        self.e = CropEngine()

    def test_padding_never_runs_past_the_top_left(self):
        self.assertEqual(self.e.clamp_bbox([2, 3, 50, 60], PAGE_W, PAGE_H, 10),
                         (0, 0, 60, 70))

    def test_padding_never_runs_past_the_bottom_right(self):
        x1, y1, x2, y2 = self.e.clamp_bbox([350, 550, 399, 599], PAGE_W, PAGE_H, 20)
        self.assertEqual((x2, y2), (PAGE_W, PAGE_H))

    def test_box_larger_than_the_page_is_clamped_to_the_page(self):
        self.assertEqual(self.e.clamp_bbox([-50, -80, 900, 900], PAGE_W, PAGE_H),
                         (0, 0, PAGE_W, PAGE_H))

    def test_crop_of_an_oversized_box_matches_the_page(self):
        self.assertEqual(self.e.crop_block(page(), [-10, -10, 5000, 5000]).size,
                         (PAGE_W, PAGE_H))

    def test_inverted_corners_are_normalised(self):
        self.assertEqual(self.e.clamp_bbox([200, 200, 100, 100], PAGE_W, PAGE_H, 0),
                         (100, 100, 200, 200))

    def test_box_entirely_off_the_page_raises(self):
        with self.assertRaises(ValueError):
            self.e.crop_block(page(), [500, 700, 600, 800])

    def test_short_bbox_raises(self):
        with self.assertRaises(ValueError):
            self.e.crop_block(page(), [1, 2])

    def test_zero_area_box_still_yields_pixels_once_padded(self):
        crop = self.e.crop_block(page(), [50, 50, 50, 50])
        self.assertEqual(crop.size, (10, 10))


class Inputs(unittest.TestCase):
    def test_numpy_array_input(self):
        import numpy as np

        arr = np.zeros((PAGE_H, PAGE_W, 3), dtype=np.uint8)
        arr[100:200, 100:200] = (255, 0, 0)          # RGB red
        crop = CropEngine().crop_block(arr, [100, 100, 200, 200], padding=0)
        self.assertEqual(crop.size, (100, 100))
        self.assertEqual(crop.getpixel((0, 0)), (255, 0, 0))

    def test_bgr_arrays_are_channel_swapped_when_declared(self):
        import numpy as np

        arr = np.zeros((PAGE_H, PAGE_W, 3), dtype=np.uint8)
        arr[100:200, 100:200] = (0, 0, 255)          # red written BGR-style
        crop = CropEngine(assume_bgr=True).crop_block(arr, [100, 100, 200, 200], 0)
        self.assertEqual(crop.getpixel((0, 0)), (255, 0, 0))

    def test_unsupported_input_type_raises(self):
        with self.assertRaises(TypeError):
            CropEngine().crop_block("not-an-image", [0, 0, 10, 10])


class Saving(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_creates_missing_directories(self):
        out = os.path.join(self.tmp, "deep", "nested", "crop.png")
        path = CropEngine().save_crop(page(), out)
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(path, out)

    def test_saved_file_is_a_readable_png_of_the_right_size(self):
        e = CropEngine()
        out = os.path.join(self.tmp, "c.png")
        e.save_crop(e.crop_block(marked_page(), [100, 100, 200, 200], 0), out)
        with Image.open(out) as img:
            self.assertEqual(img.format, "PNG")
            self.assertEqual(img.size, (100, 100))
            self.assertEqual(img.convert("RGB").getpixel((0, 0)), (255, 0, 0))

    def test_crop_and_save_round_trip(self):
        out = os.path.join(self.tmp, "x", "y.png")
        path = CropEngine().crop_and_save(page(), [10, 10, 60, 60], out, padding=0)
        with Image.open(path) as img:
            self.assertEqual(img.size, (50, 50))


class VisionHandler(unittest.TestCase):
    """Requirement 3: figures become real assets plus a Markdown tag."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.r = DualExtractionRouter()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _route_figure(self, page_number=0, bbox=(100, 100, 200, 200)):
        return self.r.route([blk("image", bbox)], page_number=page_number,
                            page_image=marked_page(), output_dir=self.tmp)[0]

    def test_markdown_tag_uses_the_documented_path(self):
        out = self._route_figure(page_number=3)
        self.assertIn(f"![Figure](figures/fig_p3_{out.block_id}.png)",
                      out.parsed_content)

    def test_status_is_completed(self):
        self.assertEqual(self._route_figure().metadata["status"], "completed")

    def test_figure_file_is_written(self):
        out = self._route_figure(page_number=2)
        expected = os.path.join(self.tmp, "figures",
                                f"fig_p2_{out.block_id}.png")
        self.assertTrue(os.path.isfile(expected), f"missing {expected}")
        with Image.open(expected) as img:
            self.assertEqual(img.size, (110, 110))   # 100px box + 5px padding

    def test_crop_file_is_written_and_recorded(self):
        out = self._route_figure(page_number=1)
        expected = os.path.join(self.tmp, "crops",
                                f"page_1_block_{out.block_id}.png")
        self.assertEqual(out.crop_path, expected)
        self.assertTrue(os.path.isfile(expected))

    def test_figure_and_crop_have_the_same_pixels(self):
        out = self._route_figure()
        with Image.open(out.crop_path) as a, \
             Image.open(out.metadata["figure_path"]) as b:
            self.assertEqual(a.tobytes(), b.tobytes())

    def test_caption_text_follows_the_image_tag(self):
        out = self.r.route([blk("image", (100, 100, 200, 200), ("Figure 1. A plot",))],
                           page_image=marked_page(), output_dir=self.tmp)[0]
        self.assertTrue(out.parsed_content.startswith("![Figure]("))
        self.assertIn("Figure 1. A plot", out.parsed_content)
        self.assertEqual(out.metadata["caption_text"], "Figure 1. A plot")

    def test_offpage_figure_degrades_without_crashing(self):
        out = self.r.route([blk("image", (900, 900, 1000, 1000))],
                           page_image=page(), output_dir=self.tmp)[0]
        self.assertIsNone(out.crop_path)
        self.assertIn("crop_error", out.metadata)
        self.assertEqual(out.metadata["status"], "pending")


class CroppedTypes(unittest.TestCase):
    """Requirement 2: TABLE and FORMULA are cropped but stay pending."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # Extraction off: these tests are about cropping, and the real
        # extractors would try to construct PP-FormulaNet / PP-Structure.
        self.r = DualExtractionRouter(formula_extractor=False,
                                      table_extractor=False)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _route(self, label):
        return self.r.route([blk(label, (100, 100, 200, 200), ("content",))],
                            page_number=0, page_image=marked_page(),
                            output_dir=self.tmp)[0]

    def test_table_is_cropped_even_when_extraction_is_disabled(self):
        out = self._route("table")
        self.assertEqual(out.block_type, BlockType.TABLE)
        self.assertEqual(out.metadata["status"], "fallback")
        self.assertTrue(os.path.isfile(out.crop_path))

    def test_formula_is_cropped_even_when_extraction_is_disabled(self):
        out = self._route("display_formula")
        self.assertEqual(out.block_type, BlockType.FORMULA)
        self.assertEqual(out.metadata["status"], "fallback")
        self.assertTrue(os.path.isfile(out.crop_path))

    def test_pending_types_get_no_figure_asset(self):
        self._route("table")
        self.assertFalse(os.path.isdir(os.path.join(self.tmp, "figures")))

    def test_text_blocks_are_never_cropped(self):
        for label in ("text", "doc_title", "abstract", None):
            out = self.r.route([blk(label, (100, 100, 200, 200), ("prose",))],
                               page_image=marked_page(), output_dir=self.tmp)[0]
            self.assertIsNone(out.crop_path, f"{label} should not be cropped")
        self.assertFalse(os.path.isdir(os.path.join(self.tmp, "crops")))

    def test_crop_filenames_follow_the_documented_scheme(self):
        out = self.r.route([blk("table", (100, 100, 200, 200))],
                           page_number=7, page_image=marked_page(),
                           output_dir=self.tmp)[0]
        self.assertTrue(out.crop_path.endswith(
            os.path.join("crops", f"page_7_block_{out.block_id}.png")))


class RoutePagesWithImages(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pairs_of_page_and_image(self):
        from src.pipeline import OrderedPage

        pages = [
            (OrderedPage("d.pdf", 0, PAGE_W, PAGE_H,
                         [blk("image", (100, 100, 200, 200))]), marked_page()),
            (OrderedPage("d.pdf", 1, PAGE_W, PAGE_H,
                         [blk("text", (10, 10, 100, 60), ("hi",))]), marked_page()),
        ]
        out = DualExtractionRouter().route_pages(pages, output_dir=self.tmp)
        self.assertEqual([len(p) for p in out], [1, 1])
        self.assertIsNotNone(out[0][0].crop_path)
        self.assertIsNone(out[1][0].crop_path)
        self.assertIn("![Figure](figures/fig_p0_0.png)", out[0][0].parsed_content)

    def test_page_carrying_its_own_image_attribute(self):
        from src.pipeline import OrderedPage

        p = OrderedPage("d.pdf", 0, PAGE_W, PAGE_H,
                        [blk("image", (100, 100, 200, 200))])
        p.image = marked_page()
        out = DualExtractionRouter().route_pages([p], output_dir=self.tmp)
        self.assertIsNotNone(out[0][0].crop_path)

    def test_bare_pages_still_work_without_images(self):
        from src.pipeline import OrderedPage

        p = OrderedPage("d.pdf", 0, PAGE_W, PAGE_H, [blk("text", (10, 10, 90, 40),
                                                         ("hi",))])
        out = DualExtractionRouter().route_pages([p])
        self.assertEqual(out[0][0].parsed_content, "hi")

    def test_ids_and_filenames_stay_unique_across_pages(self):
        from src.pipeline import OrderedPage

        pages = [(OrderedPage("d.pdf", i, PAGE_W, PAGE_H,
                              [blk("image", (100, 100, 200, 200))]), marked_page())
                 for i in range(3)]
        out = DualExtractionRouter().route_pages(pages, output_dir=self.tmp)
        paths = [p[0].crop_path for p in out]
        self.assertEqual(len(set(paths)), 3)
        figures = sorted(os.listdir(os.path.join(self.tmp, "figures")))
        self.assertEqual(figures, ["fig_p0_0.png", "fig_p1_1.png", "fig_p2_2.png"])


class ContextContract(unittest.TestCase):
    def test_handlers_still_accept_a_bare_block(self):
        """Step-1 contract: handler(block) works without a context."""
        r = DualExtractionRouter()
        self.assertEqual(
            r._handle_text(blk("text", (0, 0, 10, 10), ("a", "b")))["content"],
            "a\nb")

    def test_route_context_defaults_are_inert(self):
        ctx = RouteContext()
        self.assertIsNone(ctx.crop_path)
        self.assertIsNone(ctx.page_image)
        self.assertEqual(ctx.block_id, 0)


if __name__ == "__main__":
    unittest.main()

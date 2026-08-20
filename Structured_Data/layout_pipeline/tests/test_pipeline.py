"""Pipeline assembly tests.

These exercise DocumentPipeline.assemble, which is deliberately pure geometry:
it takes already-detected lines and blocks, so it runs with no paddle, no model
download and no GPU.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ocr_engine import LayoutBlock, TextLine, quad_to_bbox  # noqa: E402
from src.pipeline import (DocumentPipeline, attach_lines,  # noqa: E402
                          group_orphans, sort_lines, to_markdown)
from src.xycut_plus import DEFAULT_CONFIG  # noqa: E402

PAGE_W, PAGE_H = 1000, 1400


def line(text, x0, y0, x1, y1, score=0.99):
    return TextLine(text=text, bbox=[x0, y0, x1, y1], score=score)


def block(label, x0, y0, x1, y1):
    return LayoutBlock(label=label, bbox=[x0, y0, x1, y1], score=0.9)


def assemble(lines, blocks):
    pipe = DocumentPipeline.__new__(DocumentPipeline)  # no engine construction
    pipe.config = DEFAULT_CONFIG
    return pipe.assemble("test.pdf", 0, PAGE_W, PAGE_H, lines, blocks)


class QuadConversion(unittest.TestCase):
    def test_quad_to_bbox(self):
        quad = [[10, 20], [110, 22], [110, 60], [10, 58]]
        self.assertEqual(quad_to_bbox(quad), [10, 20, 110, 60])

    def test_rotated_quad_uses_extremes(self):
        quad = [[50, 10], [90, 50], [50, 90], [10, 50]]
        self.assertEqual(quad_to_bbox(quad), [10, 10, 90, 90])


class Attachment(unittest.TestCase):
    def test_lines_attach_to_containing_block(self):
        lines = [line("a", 100, 210, 400, 240), line("b", 100, 250, 400, 280)]
        blocks = [block("text", 90, 200, 480, 300)]
        per_block, orphans = attach_lines(lines, blocks)
        self.assertEqual(len(per_block[0]), 2)
        self.assertEqual(orphans, [])

    def test_line_outside_every_block_is_an_orphan(self):
        lines = [line("stray", 600, 900, 900, 940)]
        blocks = [block("text", 90, 200, 480, 300)]
        per_block, orphans = attach_lines(lines, blocks)
        self.assertEqual(per_block[0], [])
        self.assertEqual(len(orphans), 1)

    def test_orphans_become_blocks_so_text_is_never_lost(self):
        """Layout recall gaps must not delete text -- the PP-StructureV3 failure."""
        lines = [line("in", 100, 210, 400, 240), line("stray", 600, 900, 900, 940)]
        blocks = [block("text", 90, 200, 480, 300)]
        page = assemble(lines, blocks)
        recovered = " ".join(b.text for b in page.blocks)
        self.assertIn("in", recovered)
        self.assertIn("stray", recovered)
        self.assertTrue(any(b.synthetic for b in page.blocks))

    def test_partial_overlap_below_threshold_is_orphaned(self):
        # Only a sliver of the line sits inside the block.
        lines = [line("edge", 400, 210, 800, 240)]
        blocks = [block("text", 90, 200, 450, 300)]
        _, orphans = attach_lines(lines, blocks)
        self.assertEqual(len(orphans), 1)


class OrphanGrouping(unittest.TestCase):
    def test_adjacent_orphans_merge(self):
        lines = [line("l1", 100, 200, 400, 230), line("l2", 100, 235, 400, 265)]
        groups = group_orphans(lines)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)

    def test_distant_orphans_stay_separate(self):
        lines = [line("l1", 100, 200, 400, 230), line("l2", 100, 900, 400, 930)]
        self.assertEqual(len(group_orphans(lines)), 2)

    def test_side_by_side_orphans_stay_separate(self):
        lines = [line("l", 100, 200, 400, 230), line("r", 600, 200, 900, 230)]
        self.assertEqual(len(group_orphans(lines)), 2)


class Assembly(unittest.TestCase):
    def test_two_column_page_orders_blocks_and_lines(self):
        blocks = [
            block("doc_title", 80, 80, 920, 150),
            block("text", 80, 200, 480, 400),
            block("text", 520, 200, 920, 400),
        ]
        lines = [
            line("The Title", 90, 90, 900, 140),
            line("left one", 90, 210, 470, 240),
            line("left two", 90, 250, 470, 280),
            line("right one", 530, 210, 910, 240),
            line("right two", 530, 250, 910, 280),
        ]
        page = assemble(lines, blocks)
        texts = [b.text for b in page.blocks]
        self.assertEqual(texts[0], "The Title")
        self.assertEqual(texts[1], "left one\nleft two")
        self.assertEqual(texts[2], "right one\nright two")
        self.assertEqual([b.order for b in page.blocks], [0, 1, 2])

    def test_empty_text_block_is_dropped_but_figure_kept(self):
        blocks = [block("text", 80, 200, 480, 400), block("image", 520, 200, 920, 400)]
        page = assemble([], blocks)
        labels = [b.label for b in page.blocks]
        self.assertNotIn("text", labels)
        self.assertIn("image", labels)

    def test_no_detections_gives_empty_page(self):
        page = assemble([], [])
        self.assertEqual(page.blocks, [])

    def test_page_dict_is_serialisable(self):
        import json
        page = assemble([line("x", 90, 210, 470, 240)],
                        [block("text", 80, 200, 480, 400)])
        json.dumps(page.to_dict())  # must not raise


class Markdown(unittest.TestCase):
    def test_headings_and_prose(self):
        blocks = [block("doc_title", 80, 80, 920, 150),
                  block("paragraph_title", 80, 200, 480, 240),
                  block("text", 80, 250, 480, 400)]
        lines = [line("Deep Nets", 90, 90, 900, 140),
                 line("1 Introduction", 90, 205, 470, 235),
                 line("Body text here.", 90, 260, 470, 290)]
        md = to_markdown([assemble(lines, blocks)])
        self.assertIn("# Deep Nets", md)
        self.assertIn("## 1 Introduction", md)
        self.assertIn("Body text here.", md)

    def test_figure_becomes_a_comment_not_silent_loss(self):
        md = to_markdown([assemble([], [block("image", 80, 200, 480, 400)])])
        self.assertIn("<!-- image", md)

    def test_markdown_follows_reading_order(self):
        blocks = [block("text", 520, 200, 920, 400), block("text", 80, 200, 480, 400)]
        lines = [line("right", 530, 210, 910, 240), line("left", 90, 210, 470, 240)]
        md = to_markdown([assemble(lines, blocks)])
        self.assertLess(md.index("left"), md.index("right"),
                        "markdown must follow recovered order, not input order")



class LineOrdering(unittest.TestCase):
    """Regression: lines inside a block are a stack, not columns.

    Using the recursive cut here let a short line's ragged right edge open a
    phantom column, which on the ViT page put the citation year "2017" ahead of
    the sentence containing it.
    """

    def test_stacked_lines_keep_document_order(self):
        lines = [line("first line of the paragraph", 100, 200, 700, 230),
                 line("second line also long here", 100, 235, 700, 265),
                 line("2017", 100, 270, 180, 300)]  # short trailing line
        out = sort_lines(lines)
        self.assertEqual([l.text for l in out],
                         ["first line of the paragraph",
                          "second line also long here", "2017"])

    def test_side_by_side_lines_read_left_to_right(self):
        lines = [line("right", 500, 200, 700, 230),
                 line("left", 100, 202, 300, 232)]
        self.assertEqual([l.text for l in sort_lines(lines)], ["left", "right"])

    def test_short_line_does_not_reorder_the_block(self):
        lines = [line("a" * 40, 100, 200, 700, 230),
                 line("b", 100, 235, 140, 265),
                 line("c" * 40, 100, 270, 700, 300)]
        self.assertEqual([l.text[0] for l in sort_lines(lines)], ["a", "b", "c"])

if __name__ == "__main__":
    unittest.main()

"""Reading-order tests for XY-Cut++.

Pure geometry: no paddle, no GPU, no model download. Run with

    python3 -m unittest discover -s layout_pipeline/tests -v

Layouts are built as synthetic pages so the expected order is unambiguous.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.xycut_plus import (XYCutConfig, compute_reading_order,  # noqa: E402
                            detect_cross_layout, order_indices,
                            regional_density)

PAGE_W, PAGE_H = 1000.0, 1400.0


def col(x0, x1, y0, y1):
    return [x0, y0, x1, y1]


class SingleColumn(unittest.TestCase):
    """A one-column page must come back in plain top-to-bottom order."""

    def test_paragraphs_in_document_order(self):
        boxes = [col(100, 900, y, y + 80) for y in (100, 200, 300, 400, 500)]
        labels = ["text"] * len(boxes)
        self.assertEqual(order_indices(boxes, labels, PAGE_W, PAGE_H),
                         [0, 1, 2, 3, 4])

    def test_shuffled_input_is_recovered(self):
        ys = (100, 200, 300, 400, 500)
        boxes = [col(100, 900, y, y + 80) for y in ys]
        perm = [3, 0, 4, 1, 2]
        shuffled = [boxes[i] for i in perm]
        order = order_indices(shuffled, ["text"] * 5, PAGE_W, PAGE_H)
        recovered_tops = [shuffled[i][1] for i in order]
        self.assertEqual(recovered_tops, sorted(recovered_tops))

    def test_title_precedes_body(self):
        boxes = [
            col(100, 900, 60, 130),    # 0 title
            col(100, 900, 200, 280),   # 1 body
            col(100, 900, 300, 380),   # 2 body
        ]
        labels = ["doc_title", "text", "text"]
        self.assertEqual(order_indices(boxes, labels, PAGE_W, PAGE_H), [0, 1, 2])


class TwoColumn(unittest.TestCase):
    """The case row-major extraction gets wrong: columns must not interleave."""

    def _page(self):
        # Left column x in [80, 480], right column x in [520, 920].
        left = [col(80, 480, y, y + 80) for y in (200, 300, 400, 500)]
        right = [col(520, 920, y, y + 80) for y in (200, 300, 400, 500)]
        return left, right

    def test_left_column_fully_precedes_right(self):
        left, right = self._page()
        # Interleaved input, exactly how a row-major reader would emit them.
        boxes, labels = [], []
        for a, b in zip(left, right):
            boxes += [a, b]
            labels += ["text", "text"]
        order = order_indices(boxes, labels, PAGE_W, PAGE_H)
        xs = [boxes[i][0] for i in order]
        # Every left-column box must appear before every right-column box.
        split = len(left)
        self.assertTrue(all(x < 500 for x in xs[:split]),
                        f"left column not first: {xs}")
        self.assertTrue(all(x > 500 for x in xs[split:]),
                        f"right column not second: {xs}")

    def test_column_internal_order_is_top_down(self):
        left, right = self._page()
        boxes = left + right
        order = order_indices(boxes, ["text"] * len(boxes), PAGE_W, PAGE_H)
        tops = [boxes[i][1] for i in order]
        self.assertEqual(tops[:4], sorted(tops[:4]))
        self.assertEqual(tops[4:], sorted(tops[4:]))

    def test_spanning_title_is_not_cut_into_an_L(self):
        """A full-width title above two columns must be read first.

        This is the "L-shape" bug Phase 1 exists to prevent: without masking,
        a spanning block bridges the gutter and blocks the vertical cut.
        """
        left, right = self._page()
        title = col(80, 920, 80, 150)
        boxes = [title] + left + right
        labels = ["doc_title"] + ["text"] * (len(left) + len(right))
        order = order_indices(boxes, labels, PAGE_W, PAGE_H)
        self.assertEqual(order[0], 0, "title should be read first")
        xs = [boxes[i][0] for i in order[1:]]
        self.assertTrue(all(x < 500 for x in xs[:4]), f"columns interleaved: {xs}")

    # A full-width element mid-page has two legitimate readings, and which one
    # is right depends on the document convention. The masking policy is what
    # selects between them, so both are pinned down here.
    #
    #   +----------------+----------------+
    #   |    above_L     |    above_R     |
    #   +----------------+----------------+
    #   |            spanning             |
    #   +----------------+----------------+
    #   |    below_L     |    below_R     |
    #   +----------------+----------------+

    def _spanning_page(self, span_label):
        return (
            [col(80, 480, 200, 280),    # 0 above_L
             col(520, 920, 200, 280),   # 1 above_R
             col(80, 920, 320, 560),    # 2 spanning
             col(80, 480, 620, 700),    # 3 below_L
             col(520, 920, 620, 700)],  # 4 below_R
            ["text", "text", span_label, "text", "text"],
        )

    def test_masked_figure_lets_the_column_flow_past_it(self):
        """A masked full-width figure does not break the column flow.

        This is the LaTeX `figure*` convention: the figure floats, and body text
        continues down the left column past it before moving to the right. Phase
        1 removes the figure precisely so it cannot force a row split.
        """
        boxes, labels = self._spanning_page("figure")
        order = order_indices(boxes, labels, PAGE_W, PAGE_H)
        pos = {b: i for i, b in enumerate(order)}
        self.assertLess(pos[0], pos[3], "left column should run top to bottom")
        self.assertLess(pos[3], pos[1], "whole left column precedes the right")
        self.assertLess(pos[1], pos[4], "right column should run top to bottom")
        # The figure re-anchors to the left-column block it sits beneath.
        self.assertLess(pos[0], pos[2])

    def test_unmasked_spanning_block_forces_a_band_split(self):
        """An unmasked full-width block splits the page into horizontal bands.

        The newspaper convention, and the L-shape fallback: the spanning box
        bridges the gutter, so no vertical cut exists and the recursive pass
        drops to the horizontal axis.
        """
        boxes, labels = self._spanning_page("text")
        cfg = XYCutConfig(mask_vision=False, enable_cross_mask=False)
        order = order_indices(boxes, labels, PAGE_W, PAGE_H, cfg)
        pos = {b: i for i, b in enumerate(order)}
        for above in (0, 1):
            for below in (3, 4):
                self.assertLess(pos[above], pos[below],
                                "content above the banner must precede content below")
        self.assertLess(pos[2], pos[3], "the banner itself precedes the lower band")
        self.assertLess(pos[0], pos[1], "left column first within the upper band")


class AcademicHeader(unittest.TestCase):
    """Regression: the centred title/author/email stack must lead the page.

    With Stage B cross-layout masking enabled (the paper's Eq. 1-2), these
    centred full-width lines are all masked and Phase 4 re-anchors them after
    the left column -- on the real BERT page the title landed at reading
    position 9 of 13. Hence ``enable_cross_mask`` defaults to False.
    """

    def _header_page(self):
        boxes = [
            col(200, 800, 80, 150),    # 0 doc_title, centred, full width
            col(280, 720, 180, 215),   # 1 authors, centred
            col(300, 700, 225, 255),   # 2 email, centred
            col(80, 480, 320, 700),    # 3 left column
            col(520, 920, 320, 700),   # 4 right column
        ]
        labels = ["doc_title", "text", "text", "text", "text"]
        return boxes, labels

    def test_default_config_reads_the_header_first(self):
        boxes, labels = self._header_page()
        order = order_indices(boxes, labels, PAGE_W, PAGE_H)
        self.assertEqual(order[:3], [0, 1, 2],
                         f"header stack must lead the page, got {order}")
        self.assertEqual(order[3:], [3, 4], "then left column, then right")

    def test_cross_mask_is_off_by_default(self):
        from src.xycut_plus import DEFAULT_CONFIG
        self.assertFalse(DEFAULT_CONFIG.enable_cross_mask)


class MaskingAndReinsertion(unittest.TestCase):
    def test_marginal_furniture_does_not_break_columns(self):
        """A rotated margin stamp must not be mistaken for a column."""
        stamp = col(20, 60, 300, 900)      # tall, narrow, far left
        left = [col(120, 480, y, y + 80) for y in (200, 300, 400)]
        right = [col(520, 900, y, y + 80) for y in (200, 300, 400)]
        boxes = [stamp] + left + right
        labels = ["aside_text"] + ["text"] * 6
        order = order_indices(boxes, labels, PAGE_W, PAGE_H)
        body = [i for i in order if i != 0]
        xs = [boxes[i][0] for i in body]
        self.assertTrue(all(x < 500 for x in xs[:3]),
                        f"margin stamp corrupted the column cut: {xs}")

    def test_figure_is_reinserted_near_its_anchor(self):
        boxes = [
            col(80, 480, 200, 280),   # 0 left text
            col(80, 480, 300, 380),   # 1 left text
            col(520, 920, 200, 280),  # 2 right text
            col(520, 920, 300, 380),  # 3 right text
            col(520, 920, 400, 600),  # 4 figure, in the right column
        ]
        labels = ["text", "text", "text", "text", "figure"]
        order = order_indices(boxes, labels, PAGE_W, PAGE_H)
        pos = {b: i for i, b in enumerate(order)}
        self.assertGreater(pos[4], pos[2],
                           "figure should follow the right-column text it sits under")

    def test_mask_titles_toggle_changes_nothing_illegal(self):
        """Both title policies must still produce a valid permutation."""
        boxes = [col(80, 920, 80, 150), col(80, 480, 200, 400),
                 col(520, 920, 200, 400)]
        labels = ["doc_title", "text", "text"]
        for mask in (True, False):
            cfg = XYCutConfig(mask_titles=mask)
            order = order_indices(boxes, labels, PAGE_W, PAGE_H, cfg)
            self.assertEqual(sorted(order), [0, 1, 2])
            self.assertEqual(order[0], 0)


class Hyperparameters(unittest.TestCase):
    def test_min_gap_suppresses_narrow_splits(self):
        """A gutter narrower than min_gap_px must not be treated as a column break."""
        left = [col(80, 490, y, y + 80) for y in (200, 300)]
        right = [col(500, 900, y, y + 80) for y in (200, 300)]  # 10px gutter
        boxes = left + right
        labels = ["text"] * 4

        tight = order_indices(boxes, labels, PAGE_W, PAGE_H,
                              XYCutConfig(min_gap_px=1.0))
        xs = [boxes[i][0] for i in tight]
        self.assertTrue(all(x < 500 for x in xs[:2]),
                        "10px gutter should split with min_gap_px=1")

        loose = order_indices(boxes, labels, PAGE_W, PAGE_H,
                              XYCutConfig(min_gap_px=50.0))
        tops = [boxes[i][1] for i in loose]
        self.assertEqual(tops, sorted(tops),
                         "with min_gap_px=50 the gutter is ignored, so order is "
                         "row-major by y")

    def test_beta_controls_cross_layout_detection(self):
        boxes = [col(80, 480, 200, 280), col(520, 920, 200, 280),
                 col(80, 920, 320, 400)]
        idxs = list(range(3))
        wide = detect_cross_layout(boxes, idxs, XYCutConfig(beta=1.3))
        self.assertIn(2, wide, "the full-width block should be cross-layout")
        never = detect_cross_layout(boxes, idxs, XYCutConfig(beta=10.0))
        self.assertEqual(never, set(), "beta=10 should mark nothing as spanning")

    def test_regional_density_ratio(self):
        boxes = [col(0, 100, 0, 100), col(0, 200, 200, 300)]
        # box1 area 20000, box0 area 10000 -> tau_d = 2.0 when box1 is cross
        self.assertAlmostEqual(regional_density(boxes, [0, 1], {1}), 2.0)
        self.assertAlmostEqual(regional_density(boxes, [0, 1], set()), 0.0)
        self.assertEqual(regional_density(boxes, [0, 1], {0, 1}), float("inf"))


class Contract(unittest.TestCase):
    def test_empty_and_singleton(self):
        self.assertEqual(order_indices([], [], PAGE_W, PAGE_H), [])
        self.assertEqual(order_indices([col(0, 10, 0, 10)], ["text"],
                                       PAGE_W, PAGE_H), [0])

    def test_output_is_always_a_permutation(self):
        boxes = [col(80, 480, 200, 280), col(520, 920, 200, 280),
                 col(80, 920, 80, 150), col(520, 920, 400, 600),
                 col(20, 60, 300, 900)]
        labels = ["text", "text", "doc_title", "table", "aside_text"]
        order = order_indices(boxes, labels, PAGE_W, PAGE_H)
        self.assertEqual(sorted(order), list(range(len(boxes))))

    def test_ranks_invert_the_ordering(self):
        boxes = [col(80, 480, 300, 380), col(80, 480, 200, 280)]
        labels = ["text", "text"]
        order = order_indices(boxes, labels, PAGE_W, PAGE_H)
        ranks = compute_reading_order(boxes, labels, PAGE_W, PAGE_H)
        for rank, idx in enumerate(order):
            self.assertEqual(ranks[idx], rank)

    def test_missing_labels_are_tolerated(self):
        boxes = [col(80, 480, 200, 280), col(80, 480, 300, 380)]
        self.assertEqual(order_indices(boxes, None, PAGE_W, PAGE_H), [0, 1])

    def test_all_masked_page_still_orders(self):
        """A page of nothing but figures has no backbone to anchor against."""
        boxes = [col(80, 480, 300, 500), col(80, 480, 100, 250)]
        labels = ["figure", "figure"]
        order = order_indices(boxes, labels, PAGE_W, PAGE_H)
        self.assertEqual(order, [1, 0], "should fall back to geometric order")


if __name__ == "__main__":
    unittest.main()

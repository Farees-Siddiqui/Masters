"""Tests for the XML projection engine (step 1).

Pure serialisation: no models, no GPU, no disk beyond one tempdir.
"""

import os
import shutil
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dual_extractor import BlockType, SemanticBlock  # noqa: E402
from src.xml_projector import (XMLProjector, format_bbox,  # noqa: E402
                               infer_title, xml_safe)


def sb(block_id, btype, content="", page=0, label=None, status="extracted",
       bbox=(10.0, 20.0, 110.0, 60.0), **meta):
    md = {"page": page, "status": status}
    if label:
        md["label"] = label
    md.update(meta)
    return SemanticBlock(block_id=block_id, block_type=btype, bbox=list(bbox),
                         parsed_content=content, metadata=md)


def parse(xml):
    return ET.fromstring(xml)


class Helpers(unittest.TestCase):
    def test_format_bbox(self):
        self.assertEqual(format_bbox([1, 2.25, 3, 4]), "1.0,2.2,3.0,4.0")
        self.assertEqual(format_bbox(None), "")
        self.assertEqual(format_bbox([1, 2]), "")

    def test_xml_safe_removes_only_illegal_chars(self):
        self.assertEqual(xml_safe("a\x00b\x08c"), "abc")
        self.assertEqual(xml_safe("tab\tnl\ncr\r"), "tab\tnl\ncr\r")
        self.assertEqual(xml_safe("Θ ∑ é"), "Θ ∑ é")
        self.assertEqual(xml_safe(None), "")

    def test_infer_title_picks_the_doc_title(self):
        blocks = [sb(0, BlockType.TEXT, "intro"),
                  sb(1, BlockType.TITLE, "Deep Residual Learning",
                     label="doc_title")]
        self.assertEqual(infer_title(blocks), "Deep Residual Learning")

    def test_infer_title_ignores_section_headings(self):
        blocks = [sb(0, BlockType.TITLE, "1 Introduction",
                     label="paragraph_title")]
        self.assertEqual(infer_title(blocks), "Document")


class ElementMapping(unittest.TestCase):
    """Requirement 1: one element per BlockType."""

    def setUp(self):
        self.p = XMLProjector()

    def test_title_becomes_heading_with_level(self):
        el = self.p.block_to_element(sb(1, BlockType.TITLE, "T", label="doc_title"))
        self.assertEqual(el.tag, "heading")
        self.assertEqual(el.get("level"), "1")
        self.assertEqual(el.text, "T")

    def test_section_heading_is_level_two(self):
        el = self.p.block_to_element(
            sb(1, BlockType.TITLE, "1 Intro", label="paragraph_title"))
        self.assertEqual(el.get("level"), "2")

    def test_unknown_label_heading_uses_default_level(self):
        el = self.p.block_to_element(sb(1, BlockType.TITLE, "T", label="weird"))
        self.assertEqual(el.get("level"), "2")

    def test_text_becomes_paragraph(self):
        el = self.p.block_to_element(sb(2, BlockType.TEXT, "Body text."))
        self.assertEqual(el.tag, "paragraph")
        self.assertEqual(el.text, "Body text.")

    def test_unknown_becomes_flagged_paragraph(self):
        el = self.p.block_to_element(sb(3, BlockType.UNKNOWN, "?"))
        self.assertEqual(el.tag, "paragraph")
        self.assertEqual(el.get("unmapped"), "true")

    def test_formula_element(self):
        el = self.p.block_to_element(
            sb(4, BlockType.FORMULA, "$$\nx=1\n$$", status="completed",
               latex="x=1"))
        self.assertEqual(el.tag, "formula")
        self.assertEqual(el.get("format"), "latex")
        self.assertEqual(el.get("status"), "completed")
        self.assertEqual(el.get("latex"), "x=1")
        self.assertIn("$$", el.text)

    def test_table_element_records_shape(self):
        el = self.p.block_to_element(
            sb(5, BlockType.TABLE, "| a |\n|---|", status="completed",
               table_format="markdown", table_rows=2, table_columns=1))
        self.assertEqual(el.tag, "table")
        self.assertEqual(el.get("format"), "markdown")
        self.assertEqual((el.get("rows"), el.get("columns")), ("2", "1"))
        self.assertIsNone(el.get("merged_cells"))

    def test_html_table_marks_merged_cells(self):
        el = self.p.block_to_element(
            sb(6, BlockType.TABLE, "<table><tr><td>x</td></tr></table>",
               table_format="html", merged_cells=True))
        self.assertEqual(el.get("format"), "html")
        self.assertEqual(el.get("merged_cells"), "true")

    def test_failed_table_has_no_format(self):
        el = self.p.block_to_element(
            sb(7, BlockType.TABLE, "<!-- WARNING -->", status="fallback"))
        self.assertEqual(el.get("format"), "none")
        self.assertEqual(el.get("status"), "fallback")

    def test_figure_element(self):
        el = self.p.block_to_element(
            sb(8, BlockType.VISION, "![Figure](figures/fig_p0_8.png)",
               status="completed", figure_rel_path="figures/fig_p0_8.png",
               caption_text="Figure 1. Training error"))
        self.assertEqual(el.tag, "figure")
        self.assertEqual(el.get("src"), "figures/fig_p0_8.png")
        self.assertEqual(el.get("alt"), "Figure 1. Training error")

    def test_figure_without_src_keeps_its_content(self):
        el = self.p.block_to_element(
            sb(9, BlockType.VISION, "<!-- VISION: pending -->", status="pending"))
        self.assertIsNone(el.get("src"))
        self.assertEqual(el.get("alt"), "Figure")
        self.assertIn("pending", el.text)


class Provenance(unittest.TestCase):
    """Requirement 3: id / bbox / status match the input blocks."""

    def test_attributes_are_present(self):
        el = XMLProjector().block_to_element(
            sb(42, BlockType.TEXT, "x", bbox=(1.5, 2.5, 3.5, 4.5),
               status="extracted", label="text"))
        self.assertEqual(el.get("id"), "42")
        self.assertEqual(el.get("bbox"), "1.5,2.5,3.5,4.5")
        self.assertEqual(el.get("status"), "extracted")
        self.assertEqual(el.get("label"), "text")

    def test_label_can_be_suppressed(self):
        el = XMLProjector(include_label=False).block_to_element(
            sb(1, BlockType.TEXT, "x", label="text"))
        self.assertIsNone(el.get("label"))

    def test_every_block_keeps_its_own_id_and_bbox(self):
        blocks = [sb(i, BlockType.TEXT, f"p{i}", bbox=(i, i, i + 5, i + 5))
                  for i in range(6)]
        root = parse(XMLProjector().project_to_xml(blocks))
        els = root.findall(".//paragraph")
        self.assertEqual([e.get("id") for e in els],
                         [str(b.block_id) for b in blocks])
        self.assertEqual([e.get("bbox") for e in els],
                         [format_bbox(b.bbox) for b in blocks])


class Document(unittest.TestCase):
    def test_root_and_title(self):
        root = parse(XMLProjector().project_to_xml(
            [sb(0, BlockType.TEXT, "x")], doc_title="My Paper"))
        self.assertEqual(root.tag, "document")
        self.assertEqual(root.get("title"), "My Paper")
        self.assertEqual(root.get("blocks"), "1")

    def test_blocks_are_grouped_into_pages(self):
        blocks = [sb(0, BlockType.TEXT, "a", page=0),
                  sb(1, BlockType.TEXT, "b", page=0),
                  sb(2, BlockType.TEXT, "c", page=1)]
        root = parse(XMLProjector().project_to_xml(blocks))
        pages = root.findall("page")
        self.assertEqual([p.get("number") for p in pages], ["0", "1"])
        self.assertEqual(len(pages[0]), 2)
        self.assertEqual(len(pages[1]), 1)
        self.assertEqual(root.get("pages"), "2")

    def test_document_order_is_preserved_exactly(self):
        blocks = [
            sb(0, BlockType.TITLE, "Title", label="doc_title"),
            sb(1, BlockType.TEXT, "para"),
            sb(2, BlockType.FORMULA, "$$x$$"),
            sb(3, BlockType.TABLE, "| a |"),
            sb(4, BlockType.VISION, "![Figure](f.png)", figure_rel_path="f.png"),
            sb(5, BlockType.TEXT, "after"),
        ]
        root = parse(XMLProjector().project_to_xml(blocks))
        emitted = [el for page in root for el in page]
        self.assertEqual([e.tag for e in emitted],
                         ["heading", "paragraph", "formula", "table",
                          "figure", "paragraph"])
        self.assertEqual([e.get("id") for e in emitted],
                         ["0", "1", "2", "3", "4", "5"])

    def test_revisited_page_number_does_not_reorder_blocks(self):
        """Order beats tidiness: a page that reappears opens a new element
        rather than having its blocks moved back."""
        blocks = [sb(0, BlockType.TEXT, "a", page=0),
                  sb(1, BlockType.TEXT, "b", page=1),
                  sb(2, BlockType.TEXT, "c", page=0)]
        root = parse(XMLProjector().project_to_xml(blocks))
        self.assertEqual([p.get("number") for p in root.findall("page")],
                         ["0", "1", "0"])
        ids = [el.get("id") for page in root for el in page]
        self.assertEqual(ids, ["0", "1", "2"])

    def test_empty_input(self):
        root = parse(XMLProjector().project_to_xml([]))
        self.assertEqual(root.tag, "document")
        self.assertEqual(root.get("blocks"), "0")
        self.assertEqual(len(root.findall("page")), 0)

    def test_project_pages_flattens(self):
        pages = [[sb(0, BlockType.TEXT, "a", page=0)],
                 [sb(1, BlockType.TEXT, "b", page=1)]]
        root = parse(XMLProjector().project_pages(pages))
        self.assertEqual(root.get("blocks"), "2")


class Escaping(unittest.TestCase):
    """Requirement 2: well-formedness and entity escaping."""

    def test_ampersands_and_angle_brackets_in_text(self):
        xml = XMLProjector().project_to_xml(
            [sb(0, BlockType.TEXT, "a < b & c > d")])
        self.assertIn("&lt;", xml)
        self.assertIn("&amp;", xml)
        self.assertEqual(parse(xml).find(".//paragraph").text, "a < b & c > d")

    def test_quotes_in_attributes(self):
        xml = XMLProjector().project_to_xml(
            [sb(0, BlockType.VISION, "", figure_rel_path='f"1".png',
                caption_text='He said "hi" & left')])
        root = parse(xml)
        self.assertEqual(root.find(".//figure").get("alt"), 'He said "hi" & left')

    def test_html_table_survives_a_round_trip(self):
        html = '<table><tr><td>a &amp; b</td><td colspan="2">c</td></tr></table>'
        xml = XMLProjector().project_to_xml(
            [sb(0, BlockType.TABLE, html, table_format="html")])
        self.assertNotIn("<td>", xml, "raw HTML must be escaped, not nested")
        self.assertEqual(parse(xml).find(".//table").text, html)

    def test_latex_backslashes_survive(self):
        latex = r"$$\Theta=\frac{1}{N}\sum_{i=1}^{N}\ell(x_i)$$"
        xml = XMLProjector().project_to_xml([sb(0, BlockType.FORMULA, latex)])
        self.assertEqual(parse(xml).find(".//formula").text, latex)

    def test_illegal_control_characters_are_dropped(self):
        xml = XMLProjector().project_to_xml(
            [sb(0, BlockType.TEXT, "before\x00after")])
        self.assertEqual(parse(xml).find(".//paragraph").text, "beforeafter")

    def test_declaration_and_wellformedness(self):
        xml = XMLProjector().project_to_xml([sb(0, BlockType.TEXT, "x")])
        self.assertTrue(xml.lstrip().startswith("<?xml"))
        parse(xml)  # must not raise

    def test_title_with_special_characters(self):
        xml = XMLProjector().project_to_xml([], doc_title='R&D <report> "v2"')
        self.assertEqual(parse(xml).get("title"), 'R&D <report> "v2"')


class Saving(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_creates_directories_and_writes(self):
        p = XMLProjector()
        out = os.path.join(self.tmp, "deep", "doc.xml")
        p.save_xml(p.project_to_xml([sb(0, BlockType.TEXT, "x")]), out)
        self.assertTrue(os.path.isfile(out))
        ET.parse(out)

    def test_saved_file_is_indented(self):
        p = XMLProjector()
        out = os.path.join(self.tmp, "doc.xml")
        p.save_xml(p.project_to_xml([sb(0, BlockType.TEXT, "x")]), out)
        text = open(out, encoding="utf-8").read()
        self.assertIn("\n  <page", text)
        self.assertIn("\n    <paragraph", text)

    def test_save_rejects_malformed_xml(self):
        with self.assertRaises(ET.ParseError):
            XMLProjector().save_xml("<document><page></document>",
                                    os.path.join(self.tmp, "bad.xml"))


class DictInput(unittest.TestCase):
    """A saved blocks.json can be re-projected without re-running inference."""

    def test_plain_dicts_project(self):
        payload = sb(3, BlockType.TABLE, "| a |", status="completed",
                     table_format="markdown").to_dict()
        root = parse(XMLProjector().project_to_xml([payload]))
        el = root.find(".//table")
        self.assertEqual(el.get("id"), "3")
        self.assertEqual(el.get("format"), "markdown")

    def test_semantic_block_round_trip(self):
        original = sb(7, BlockType.FORMULA, "$$x$$", status="completed",
                      latex="x")
        restored = SemanticBlock.from_dict(original.to_dict())
        self.assertEqual(restored.block_id, 7)
        self.assertIs(restored.block_type, BlockType.FORMULA)
        self.assertEqual(restored.status, "completed")
        self.assertEqual(restored.parsed_content, "$$x$$")

    def test_unknown_block_type_string_degrades(self):
        restored = SemanticBlock.from_dict(
            {"block_id": 1, "block_type": "NONSENSE", "bbox": [0, 0, 1, 1]})
        self.assertIs(restored.block_type, BlockType.UNKNOWN)

class DynamicProjection(unittest.TestCase):
    """Step 2: the discovered-schema tree serialises to XML.

    Unlike the structural projection this imposes no vocabulary at all — every
    tag here came from the model.
    """

    def setUp(self):
        from src.ie_engine.node_schema import DynamicElement
        self.E = DynamicElement
        self.p = XMLProjector()

    def _tree(self):
        E = self.E
        return E("report_card",
                 attributes={"institution": "Milton College", "year": "2025-2026"},
                 children=[
                     E("student", attributes={"last_name": "Abebe"},
                       children=[E("address",
                                   attributes={"city": "London",
                                               "street": "1180 Fanshawe Park Road"})]),
                     E("grades", attributes={"grade": "85"}),
                     E("note", text_content="issued for information only"),
                 ])

    def test_tags_attributes_text_and_children(self):
        root = parse(self.p.project_dynamic_xml(self._tree()))
        self.assertEqual(root.tag, "report_card")
        self.assertEqual(root.get("institution"), "Milton College")
        self.assertEqual([c.tag for c in root], ["student", "grades", "note"])
        self.assertEqual(root.find("student/address").get("city"), "London")
        self.assertEqual(root.find("note").text, "issued for information only")

    def test_output_is_declared_and_indented(self):
        xml = self.p.project_dynamic_xml(self._tree())
        self.assertTrue(xml.lstrip().startswith("<?xml"))
        self.assertIn("\n  <student", xml)
        self.assertIn("\n    <address", xml)

    def test_deep_nesting_is_preserved(self):
        E = self.E
        deep = E("a", children=[E("b", children=[E("c", children=[
            E("d", text_content="bottom")])])])
        root = parse(self.p.project_dynamic_xml(deep))
        self.assertEqual(root.find("b/c/d").text, "bottom")

    def test_non_string_attribute_values_are_cleaned(self):
        el = self.E("x", attributes={"n": 42, "f": 1.5, "yes": True,
                                     "no": False, "none": None,
                                     "obj": {"k": "v"}})
        root = parse(self.p.project_dynamic_xml(el))
        self.assertEqual(root.get("n"), "42")
        self.assertEqual(root.get("f"), "1.5")
        self.assertEqual(root.get("yes"), "true")   # not Python's "True"
        self.assertEqual(root.get("no"), "false")
        self.assertIsNone(root.get("none"))         # empty attrs are dropped
        self.assertEqual(root.get("obj"), '{"k": "v"}')

    def test_illegal_tags_and_keys_are_sanitised(self):
        el = self.E("Student Record", attributes={"Last Name": "Abebe"},
                    children=[self.E("2024 grades")])
        root = parse(self.p.project_dynamic_xml(el))
        self.assertEqual(root.tag, "student-record")
        self.assertEqual(root.get("last-name"), "Abebe")
        self.assertEqual(root[0].tag, "n2024-grades")

    def test_entities_are_escaped(self):
        el = self.E("x", attributes={"q": 'He said "hi" & left'},
                    text_content="a < b & c > d")
        xml = self.p.project_dynamic_xml(el)
        self.assertIn("&amp;", xml)
        root = parse(xml)
        self.assertEqual(root.get("q"), 'He said "hi" & left')
        self.assertEqual(root.text, "a < b & c > d")

    def test_illegal_control_characters_removed(self):
        root = parse(self.p.project_dynamic_xml(
            self.E("x", text_content="before\x00after")))
        self.assertEqual(root.text, "beforeafter")

    def test_title_and_source_are_attached_to_the_root(self):
        root = parse(self.p.project_dynamic_xml(
            self.E("doc"), doc_title="My Doc", source="doc01.pdf"))
        self.assertEqual(root.get("title"), "My Doc")
        self.assertEqual(root.get("source"), "doc01.pdf")

    def test_empty_tree_is_still_well_formed(self):
        root = parse(self.p.project_dynamic_xml(self.E("empty")))
        self.assertEqual(root.tag, "empty")
        self.assertEqual(len(root), 0)

    def test_repeated_children_keep_their_order(self):
        E = self.E
        el = E("cargo", children=[E("item", attributes={"id": str(i)})
                                  for i in range(5)])
        root = parse(self.p.project_dynamic_xml(el))
        self.assertEqual([c.get("id") for c in root], ["0", "1", "2", "3", "4"])

    def test_projects_a_whole_dynamic_document(self):
        from src.ie_engine.node_schema import DynamicDocument
        doc = DynamicDocument(root=self._tree(), source="doc08.pdf")
        root = parse(self.p.project_dynamic_document(doc))
        self.assertEqual(root.get("source"), "doc08.pdf")
        self.assertEqual(root.tag, "report_card")

    def test_end_to_end_from_raw_model_json(self):
        """Arbitrary LLM JSON -> tree -> XML, with no schema anywhere."""
        from src.ie_engine import DynamicInformationExtractor
        doc = DynamicInformationExtractor.parse_response(
            '{"invoice": {"total": "9.99", "lines": ['
            '{"sku": "A1", "qty": 2}, {"sku": "B2", "qty": 1}]}}')
        root = parse(self.p.project_dynamic_xml(doc.root))
        self.assertEqual(root.tag, "invoice")
        self.assertEqual(root.get("total"), "9.99")
        self.assertEqual([c.get("sku") for c in root.findall("lines")],
                         ["A1", "B2"])


if __name__ == "__main__":
    unittest.main()

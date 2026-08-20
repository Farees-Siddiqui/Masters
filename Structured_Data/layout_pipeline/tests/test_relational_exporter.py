"""Tests for the relational exporter.

No LLM, no GPU: the trees here are built by hand or parsed from XML strings, so
the whole file is pure data manipulation. The fixtures use tags the exporter has
never seen, because a discovered schema is by definition unknown to it.
"""

import csv
import os
import sqlite3
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ie_engine.node_schema import DynamicElement  # noqa: E402
from src.ie_engine.relational_exporter import (  # noqa: E402
    BASE_COLUMNS, RelationalExporter, coerce_element, sql_ident)

#: student -> address -> city, the deep-nesting case from the corpus.
DEEP_XML = """<?xml version='1.0' encoding='utf-8'?>
<email date="12 March 2026" subject="year-end file">
  <sender email_address="r.delacroix@milton-college.ca" />
  <recipients email_address="academic.advising@milton-college.ca" />
  <body>
    <student name="Farees Siddiqui">
      <mailing_address street="14 Fake Street">
        <city name="Milton" province="ON" />
      </mailing_address>
      <grade value="90" evaluation="Satisfactory" />
    </student>
  </body>
</email>"""

#: Two sibling records under one parent: the array case.
REGISTER_XML = """<?xml version='1.0' encoding='utf-8'?>
<student_register title="Consolidated Student Register">
  <students last_name="Chen" first_name="Wei">
    <address street="704 Highland Avenue" city="Hamilton" />
    <evaluation grade="97" remark="Excellent" />
  </students>
  <students last_name="Okonkwo" first_name="Chidi">
    <address street="2201 Lakeshore Road" city="Burlington" />
    <evaluation grade="94" remark="Excellent" />
  </students>
</student_register>"""


class SqlIdentTests(unittest.TestCase):
    def test_hyphens_and_dots_become_underscores(self):
        self.assertEqual(sql_ident("student-record"), "student_record")
        self.assertEqual(sql_ident("a.b"), "a_b")

    def test_leading_digit_is_prefixed(self):
        self.assertEqual(sql_ident("2024-grades"), "n2024_grades")

    def test_empty_falls_back(self):
        self.assertEqual(sql_ident(""), "value")
        self.assertEqual(sql_ident(None, default="col"), "col")


class FlatteningTests(unittest.TestCase):
    """1. Deeply nested trees flatten to one table per tag."""

    def setUp(self):
        self.ex = RelationalExporter(DEEP_XML)

    def test_every_tag_becomes_a_table(self):
        self.assertEqual(
            set(self.ex.table_names()),
            {"email", "sender", "recipients", "body", "student",
             "mailing_address", "city", "grade"})

    def test_root_table_is_the_root_tag(self):
        self.assertEqual(self.ex.root_table, "email")

    def test_parents_are_created_before_children(self):
        # Creation order is what lets SQLite enforce the keys at insert time.
        order = self.ex.table_names()
        for child, parent in [("student", "body"), ("city", "mailing_address"),
                              ("mailing_address", "student")]:
            self.assertLess(order.index(parent), order.index(child),
                            f"{parent} must be created before {child}")

    def test_attributes_become_columns(self):
        grade = self.ex.tables["grade"]
        self.assertEqual(grade.columns,
                         list(BASE_COLUMNS) + ["value", "evaluation"])
        self.assertEqual(grade.rows[0]["value"], "90")

    def test_deep_chain_survives_four_levels(self):
        # email -> body -> student -> mailing_address -> city
        tables = self.ex.to_dict()
        city = tables["city"][0]
        addr = tables["mailing_address"][0]
        student = tables["student"][0]
        body = tables["body"][0]
        self.assertEqual(city["name"], "Milton")
        self.assertEqual(city["parent_table"], "mailing_address")
        self.assertEqual(city["parent_id"], addr["id"])
        self.assertEqual(addr["parent_id"], student["id"])
        self.assertEqual(student["parent_id"], body["id"])
        self.assertEqual(body["parent_id"], tables["email"][0]["id"])

    def test_root_row_has_no_parent(self):
        root = self.ex.to_dict()["email"][0]
        self.assertIsNone(root["parent_id"])
        self.assertIsNone(root["parent_table"])

    def test_text_content_becomes_a_column(self):
        ex = RelationalExporter("<note><line>hello</line></note>")
        self.assertIn("text_content", ex.tables["line"].columns)
        self.assertEqual(ex.tables["line"].rows[0]["text_content"], "hello")

    def test_indentation_whitespace_is_not_text(self):
        # The pretty-printed XML puts newlines in .text on every parent.
        self.assertFalse(self.ex.tables["email"].has_text)
        self.assertNotIn("text_content", self.ex.tables["body"].columns)

    def test_attribute_named_id_is_not_clobbered(self):
        ex = RelationalExporter('<r><x id="A7" parent_id="Z" /></r>')
        row = ex.tables["x"].rows[0]
        self.assertEqual(row["id"], 1)          # ours, the surrogate key
        self.assertEqual(row["attr_id"], "A7")  # theirs, preserved
        self.assertEqual(row["attr_parent_id"], "Z")


class ArrayTests(unittest.TestCase):
    """Repeated children become rows, not comma-joined attributes."""

    def setUp(self):
        self.ex = RelationalExporter(REGISTER_XML)

    def test_repeated_children_are_separate_rows(self):
        students = self.ex.to_dict()["students"]
        self.assertEqual(len(students), 2)
        self.assertEqual([s["last_name"] for s in students], ["Chen", "Okonkwo"])

    def test_each_row_gets_its_own_grandchildren(self):
        tables = self.ex.to_dict()
        by_parent = {a["parent_id"]: a["city"] for a in tables["address"]}
        self.assertEqual(by_parent, {1: "Hamilton", 2: "Burlington"})

    def test_ids_are_unique_and_sequential_per_table(self):
        for name, rows in self.ex.to_dict().items():
            ids = [r["id"] for r in rows]
            self.assertEqual(ids, list(range(1, len(rows) + 1)), name)


class KeyIntegrityTests(unittest.TestCase):
    """2. Primary and foreign keys line up across every linked table."""

    def test_every_parent_id_resolves_to_a_real_row(self):
        for xml in (DEEP_XML, REGISTER_XML):
            ex = RelationalExporter(xml)
            tables = ex.to_dict()
            ids = {name: {r["id"] for r in rows} for name, rows in tables.items()}
            for name, rows in tables.items():
                for row in rows:
                    if row["parent_table"] is None:
                        continue
                    self.assertIn(row["parent_table"], ids,
                                  f"{name} points at an unknown table")
                    self.assertIn(row["parent_id"], ids[row["parent_table"]],
                                  f"{name}.parent_id is dangling")

    def test_exactly_one_root_row_has_a_null_parent(self):
        ex = RelationalExporter(REGISTER_XML)
        orphans = [(n, r) for n, rows in ex.to_dict().items()
                   for r in rows if r["parent_id"] is None]
        self.assertEqual(len(orphans), 1)
        self.assertEqual(orphans[0][0], "student_register")

    def test_single_parent_gets_a_declared_foreign_key(self):
        ex = RelationalExporter(DEEP_XML)
        stmt = ex.create_statement(ex.tables["city"])
        self.assertIn('FOREIGN KEY ("parent_id") REFERENCES "mailing_address"',
                      stmt)

    def test_polymorphic_parent_keeps_the_column_but_drops_the_constraint(self):
        # <address> hangs off two different tags, so no single table can be
        # named in a REFERENCES clause.
        xml = ('<doc><issuer><address city="Milton"/></issuer>'
               '<student><address city="Guelph"/></student></doc>')
        ex = RelationalExporter(xml)
        self.assertEqual(ex.tables["address"].parent_tables,
                         {"issuer", "student"})
        stmt = ex.create_statement(ex.tables["address"])
        self.assertNotIn("FOREIGN KEY", stmt)
        self.assertIn('"parent_table" TEXT', stmt)
        rows = ex.to_dict()["address"]
        self.assertEqual([r["parent_table"] for r in rows],
                         ["issuer", "student"])


class SqliteRoundTripTests(unittest.TestCase):
    """3. The database is created, populated, and queryable with joins."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = os.path.join(self.tmp.name, "doc.db")

    def test_tables_are_created(self):
        RelationalExporter(DEEP_XML).to_sqlite(self.db)
        self.assertTrue(os.path.exists(self.db))
        conn = sqlite3.connect(self.db)
        self.addCleanup(conn.close)
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertLessEqual(
            {"email", "sender", "body", "student", "mailing_address", "city"},
            names)

    def test_join_recovers_the_nesting(self):
        RelationalExporter(REGISTER_XML).to_sqlite(self.db)
        conn = sqlite3.connect(self.db)
        self.addCleanup(conn.close)
        rows = conn.execute("""
            SELECT s.last_name, a.city, e.grade
            FROM students s
            JOIN address a ON a.parent_id = s.id AND a.parent_table = 'students'
            JOIN evaluation e ON e.parent_id = s.id
            ORDER BY s.id
        """).fetchall()
        self.assertEqual(rows, [("Chen", "Hamilton", "97"),
                                ("Okonkwo", "Burlington", "94")])

    def test_four_level_join_reaches_the_leaf(self):
        RelationalExporter(DEEP_XML).to_sqlite(self.db)
        conn = sqlite3.connect(self.db)
        self.addCleanup(conn.close)
        row = conn.execute("""
            SELECT st.name, ma.street, c.name
            FROM email em
            JOIN body b  ON b.parent_id  = em.id
            JOIN student st ON st.parent_id = b.id
            JOIN mailing_address ma ON ma.parent_id = st.id
            JOIN city c ON c.parent_id = ma.id
        """).fetchone()
        self.assertEqual(row, ("Farees Siddiqui", "14 Fake Street", "Milton"))

    def test_foreign_keys_are_enforced_on_the_loaded_database(self):
        RelationalExporter(DEEP_XML).to_sqlite(self.db)
        conn = sqlite3.connect(self.db)
        self.addCleanup(conn.close)
        conn.execute("PRAGMA foreign_keys = ON")
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute('INSERT INTO city (parent_id, parent_table, name) '
                         "VALUES (999, 'mailing_address', 'Nowhere')")

    def test_id_autoincrements_on_further_inserts(self):
        RelationalExporter(REGISTER_XML).to_sqlite(self.db)
        conn = sqlite3.connect(self.db)
        self.addCleanup(conn.close)
        conn.execute('INSERT INTO students (parent_id, parent_table, last_name) '
                     "VALUES (1, 'student_register', 'Nguyen')")
        self.assertEqual(
            conn.execute("SELECT id FROM students WHERE last_name='Nguyen'")
                .fetchone()[0], 3)

    def test_rerunning_refreshes_rather_than_doubles(self):
        for _ in range(2):
            RelationalExporter(REGISTER_XML).to_sqlite(self.db)
        conn = sqlite3.connect(self.db)
        self.addCleanup(conn.close)
        self.assertEqual(
            conn.execute("SELECT COUNT(*) FROM students").fetchone()[0], 2)


class CsvBundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_one_csv_per_table_with_headers(self):
        ex = RelationalExporter(REGISTER_XML)
        written = ex.to_csv_bundle(self.tmp.name)
        self.assertEqual(len(written), len(ex.tables))
        self.assertEqual({os.path.basename(p) for p in written},
                         {f"{n}.csv" for n in ex.table_names()})
        with open(os.path.join(self.tmp.name, "students.csv"),
                  encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        self.assertEqual(rows[0],
                         ["id", "parent_id", "parent_table",
                          "last_name", "first_name"])
        self.assertEqual(rows[1], ["1", "1", "student_register", "Chen", "Wei"])
        self.assertEqual(len(rows), 3)

    def test_missing_cells_are_blank_not_none(self):
        # Second <x> lacks the attribute the first one introduced.
        ex = RelationalExporter('<r><x a="1" /><x b="2" /></r>')
        ex.to_csv_bundle(self.tmp.name)
        with open(os.path.join(self.tmp.name, "x.csv"), encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        self.assertEqual(rows[0], ["id", "parent_id", "parent_table", "a", "b"])
        self.assertEqual(rows[1], ["1", "1", "r", "1", ""])
        self.assertEqual(rows[2], ["2", "1", "r", "", "2"])

    def test_creates_the_directory(self):
        target = os.path.join(self.tmp.name, "nested", "tables")
        RelationalExporter(DEEP_XML).to_csv_bundle(target)
        self.assertTrue(os.path.isdir(target))


class InputCoercionTests(unittest.TestCase):
    """Every form the IE engine hands out is accepted."""

    def test_dynamic_element(self):
        root = DynamicElement("student", {"name": "Wei"})
        root.add(DynamicElement("address", {"city": "Hamilton"}))
        ex = RelationalExporter(root)
        self.assertEqual(ex.to_dict()["address"][0]["city"], "Hamilton")

    def test_etree_element_and_tree(self):
        tree = ET.ElementTree(ET.fromstring(REGISTER_XML))
        self.assertEqual(len(RelationalExporter(tree).to_dict()["students"]), 2)
        self.assertEqual(
            len(RelationalExporter(tree.getroot()).to_dict()["students"]), 2)

    def test_xml_file_path(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "doc.semantic.xml")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(DEEP_XML)
            ex = RelationalExporter(path)
            self.assertEqual(ex.root_table, "email")

    def test_extractor_json_payload(self):
        payload = {"student_record": {
            "name": {"lastname": "Chen", "firstname": "Wei"},
            "courses": [{"code": "MATH101"}, {"code": "PHYS201"}],
        }}
        ex = RelationalExporter(payload)
        self.assertEqual(ex.root_table, "student_record")
        self.assertEqual([c["code"] for c in ex.to_dict()["courses"]],
                         ["MATH101", "PHYS201"])

    def test_dynamic_element_to_dict_round_trip(self):
        root = DynamicElement("doc", {"a": "1"})
        root.add(DynamicElement("kid", {"b": "2"}))
        ex = RelationalExporter(root.to_dict())
        self.assertEqual(ex.to_dict()["kid"][0]["b"], "2")

    def test_unsupported_type_is_rejected(self):
        with self.assertRaises(TypeError):
            RelationalExporter(object())

    def test_bare_string_that_is_not_xml_or_a_path_is_rejected(self):
        with self.assertRaises(ValueError):
            coerce_element("just some words")


if __name__ == "__main__":
    unittest.main(verbosity=2)

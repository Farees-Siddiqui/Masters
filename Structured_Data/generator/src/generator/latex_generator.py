"""Stage 5a: ask local Llama 3 to write the LaTeX for one document.

    List[Record] + layout_style -> LLMLaTeXGenerator -> LaTeX source

This replaces the fixed Jinja templates that used to render Stage 5. The trade
is deliberate and worth stating, because it changes what can go wrong:

* A template *could not* alter a value. It received escaped strings and
  interpolated them, so whatever the manifest claimed was on the page was on the
  page. What it could not do was vary — three templates produced three layouts,
  and a corpus of a thousand documents had three shapes.
* A model *can* vary without limit, which is the point. But it can also reword
  "Green Earth Landscaping", round "$1,240.50", drop a field, or invent one, and
  every one of those silently makes the ground truth wrong.

So the value fidelity that used to be structural is now *checked*:
:func:`missing_values` reads the generated source back and reports any recorded
value that is not in it, and the renderer feeds that back to the model as a
repair. Escaping is likewise no longer guaranteed by construction — it is
instructed in the system prompt and enforced by the compiler, which is why the
repair loop exists at all.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .instance_types import Record
from .schema_types import SchemaGraph

log = logging.getLogger(__name__)

#: Layouts the generator can be asked for. ``auto`` is resolved by the caller.
LAYOUT_STYLES = ("auto", "table", "form", "letter")

#: Packages the model may use. Anything else is a compile failure on a machine
#: with a minimal TeX install, and the repair loop cannot install packages.
ALLOWED_PACKAGES = ("geometry", "array", "booktabs", "longtable", "parskip",
                    "tabularx", "enumitem", "fontenc", "inputenc")

#: Kept in the preamble of every document whatever the model wrote. LaTeX
#: hyphenating a data value ("Othertown" -> "Other-town") inserts a character
#: that was never in the data, so the value can no longer be read back off the
#: page while the manifest still claims it. Measured on a real llama3.3:70b run.
HYPHENATION_GUARD = r"""% Inserted by the generator: a hyphen LaTeX adds to a data value cannot be
% recovered by an extractor, and would make the manifest wrong about this page.
\hyphenpenalty=10000
\exhyphenpenalty=10000
\emergencystretch=3em
\sloppy"""

#: How a null is presented to the model, and what it should print.
NULL_PRESENTATION = "(blank -- print an em dash)"


class LatexText(str):
    """A string already escaped for LaTeX. See :func:`escape_latex`."""

    __slots__ = ()


#: The ten characters LaTeX reserves. Three replacements contain braces of their
#: own, which is why the substitution is a single pass: replacing sequentially,
#: backslash first, would leave the "{}" it inserts to be escaped again by the
#: brace rule and "\\" would come out as "\textbackslash\{\}" -- a literal
#: backslash followed by two braces on the page.
_ESCAPES: Dict[str, str] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_ESCAPE_RE = re.compile("[" + re.escape("".join(_ESCAPES)) + "]")

#: Inverse of the table above, for reading a generated source back. Longest
#: first so "\textbackslash{}" is consumed before "\{".
_UNESCAPES: Tuple[Tuple[str, str], ...] = (
    (r"\textbackslash{}", "\\"),
    (r"\textasciitilde{}", "~"),
    (r"\textasciicircum{}", "^"),
    (r"\textbackslash", "\\"),
    (r"\textasciitilde", "~"),
    (r"\textasciicircum", "^"),
    (r"\&", "&"), (r"\%", "%"), (r"\$", "$"), (r"\#", "#"),
    (r"\_", "_"), (r"\{", "{"), (r"\}", "}"),
)

NULL_GLYPH = "---"


def escape_latex(value: Any) -> LatexText:
    """Make ``value`` safe to typeset. Idempotent on :class:`LatexText`.

    Still used even though the model does its own escaping: the fidelity check
    compares a recorded value against the generated source, and needs to know
    what the escaped form of that value looks like.
    """
    if isinstance(value, LatexText):
        return value
    if value is None:
        return LatexText(NULL_GLYPH)
    if isinstance(value, bool):
        return LatexText("Yes" if value else "No")
    return LatexText(_ESCAPE_RE.sub(lambda m: _ESCAPES[m.group()], str(value)))


def unescape_latex(source: str) -> str:
    """Undo LaTeX escaping, for reading a value back out of a source file."""
    text = str(source)
    for escaped, plain in _UNESCAPES:
        text = text.replace(escaped, plain)
    return text


_WHITESPACE = re.compile(r"\s+")
#: Formatting commands a model wraps values in. Stripped before comparison so
#: "\textbf{Green Earth}" still matches the recorded "Green Earth".
_WRAPPERS = re.compile(
    r"\\(?:textbf|textit|texttt|emph|textsc|underline|mbox|textsl|textrm)\s*\{")
#: Commands that put a break on the page. Replaced with a space *before*
#: unescaping, so a literal backslash in the data (which arrives as
#: "\textbackslash{}" and only becomes "\" afterwards) cannot be mistaken
#: for one of them.
_BREAKS = re.compile(r"\\\\\*?|\\(?:newline|linebreak|par|hfill|hspace|"
                     r"vspace|quad|qquad)\b")


def normalize_for_comparison(source: str) -> str:
    """Flatten a LaTeX source into something a raw value can be searched in.

    Turns page breaks into spaces, de-escapes, drops the formatting commands a
    value might be wrapped in, turns ``~`` and ``\\,`` into spaces, and collapses
    whitespace — because a value the model broke across a line, bolded, or
    spaced with a tie is still that value on the page. What survives all that
    and still does not match is a value that was actually reworded, rounded,
    truncated or dropped.
    """
    text = _BREAKS.sub(" ", str(source))
    text = unescape_latex(text)
    text = _WRAPPERS.sub("", text)
    text = text.replace("~", " ").replace(r"\,", " ").replace(r"\ ", " ")
    text = text.replace("}", " ").replace("{", " ")
    return _WHITESPACE.sub(" ", text)


def humanize(name: Any) -> str:
    """``company_name`` -> ``Company Name``, for a field label in a prompt."""
    words = re.split(r"[^0-9a-zA-Z]+", str(name or ""))
    return " ".join(w[:1].upper() + w[1:] for w in words if w)


# --------------------------------------------------------------------------- #
# Extracting the source from a model response
# --------------------------------------------------------------------------- #
_FENCE = re.compile(r"```(?:la)?tex\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_ANY_FENCE = re.compile(r"```\s*\n?(.*?)```", re.DOTALL)


def extract_latex(text: Optional[str]) -> Optional[str]:
    """Pull a complete LaTeX document out of a model response.

    More forgiving than taking the response as-is, because a model asked for
    bare source still fences it, prefixes "Here is the document:", or appends a
    note afterwards. The document is whatever runs from the first
    ``\\documentclass`` to the last ``\\end{document}``; anything outside that is
    commentary and is dropped.
    """
    if not text or not text.strip():
        return None
    body = text.strip()

    fenced = _FENCE.search(body) or _ANY_FENCE.search(body)
    if fenced and "\\documentclass" in fenced.group(1):
        body = fenced.group(1)

    start = body.find("\\documentclass")
    if start < 0:
        return None
    end = body.rfind("\\end{document}")
    if end < 0:
        return None
    return body[start:end + len("\\end{document}")].strip() + "\n"


def harden_source(source: str) -> str:
    """Guarantee the preamble settings the corpus depends on.

    Only the hyphenation guard, and only when the model did not already set it.
    Everything else the model wrote is left alone: repairing its LaTeX by hand
    would hide the failures the repair loop exists to surface.
    """
    if "hyphenpenalty" in source:
        return source
    marker = "\\begin{document}"
    index = source.find(marker)
    if index < 0:
        return source
    return source[:index] + HYPHENATION_GUARD + "\n\n" + source[index:]


# --------------------------------------------------------------------------- #
# Fidelity
# --------------------------------------------------------------------------- #
def recorded_values(records: Sequence[Record]) -> List[Tuple[str, str, Any]]:
    """``(record_id, field, value)`` for everything that must be on the page.

    Nulls are excluded — a blank field has no value to look for. Foreign keys
    are excluded too, and deliberately: the join is expressed by *containment*,
    the child appearing inside the parent's document, and printing the parent id
    in every row would hand an extractor the answer. Record identifiers are
    included, because a record the page never names cannot be recovered from it.
    """
    out: List[Tuple[str, str, Any]] = []
    for record in records:
        out.append((record.id, "id", record.id))
        for name, value in record.attributes.items():
            if value is None or value == record.id:
                continue
            out.append((record.id, name, value))
    return out


def missing_values(records: Sequence[Record], source: str
                   ) -> List[Tuple[str, str, Any]]:
    """Recorded values that do not appear in ``source``.

    The check runs against a de-escaped, whitespace-collapsed copy of the
    source, so a value the model bolded, split across a line or escaped
    differently still counts as present. What it catches is a value that was
    reworded, rounded, truncated or dropped.
    """
    haystack = normalize_for_comparison(source)
    missing = []
    for record_id, field, value in recorded_values(records):
        needle = _WHITESPACE.sub(" ", str(value)).strip()
        if needle and needle not in haystack:
            missing.append((record_id, field, value))
    return missing


#: Literals that appear in the prompts below as illustrations of form. A page
#: carrying one of these is carrying a fact that came from the instructions
#: rather than from the records — measured on a real llama3.1:8b run, which
#: copied "$1,240.50" out of the letter layout's example sentence and stated it
#: as this document's account balance.
#:
#: This is the mirror of :func:`missing_values` and the more dangerous half. A
#: dropped value leaves the manifest claiming something the page lacks, which is
#: detectable by reading the page. An invented one puts a fact on the page that
#: no record holds and that the manifest never mentions, so nothing downstream
#: can tell it from data.
PROMPT_EXAMPLE_VALUES = (
    "$1,240.50", "1,240.50", "USD 1240.50",
    "Bell & Co.", "50% off_now",
    "Early Learning",
)


def leaked_examples(records: Sequence[Record], source: str) -> List[str]:
    """Prompt example literals that reached the page without backing a record.

    A value is only a leak when no record holds it or contains it: an education
    corpus may legitimately contain a course called "Early Learning", and a page
    that prints it because a record says so is correct. A value the model
    *altered* rather than invented is :func:`missing_values`'s business, not
    this one.
    """
    haystack = normalize_for_comparison(source)
    held = {_WHITESPACE.sub(" ", str(v)).strip()
            for _, _, v in recorded_values(records)}
    found: List[str] = []
    # Longest first, then drop any shorter example contained in one already
    # found: "$1,240.50" and "1,240.50" are both listed, because either form
    # alone is a leak, but one sentence carrying both is one leak, not two.
    for example in sorted(PROMPT_EXAMPLE_VALUES, key=len, reverse=True):
        if example not in haystack:
            continue
        # A substring of a value a record legitimately holds is part of that
        # value, not a second fact: a page showing "$1,240.50" because a record
        # says so also shows "1,240.50". If the model dropped the "$", that is
        # a *missing* value and missing_values is what reports it.
        if any(example in value for value in held):
            continue
        if any(example in longer for longer in found):
            continue
        found.append(example)
    return [e for e in PROMPT_EXAMPLE_VALUES if e in found]


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
LATEX_GENERATION_SYSTEM_PROMPT = r"""\
You write LaTeX. You are given the records that belong on one document and the \
layout it should take, and you return the complete source for that document.

Output rules:
- Return ONLY LaTeX source. No prose before or after it, no markdown fences.
- Start at \documentclass and end at \end{document}. Include the preamble.
- Use only these packages: geometry, array, booktabs, longtable, tabularx, \
parskip, enumitem, fontenc, inputenc. Nothing else is installed.
- Use \documentclass{article}. Do not use tikz, minted, hyperref, fancyhdr, \
tcolorbox, xcolor or any package not listed above.
- Do not use \write18, \input or \include.

Mistakes that make the document fail to build. Each of these was a real failure:
- \begin{letter}, \opening, \closing, \address, \signature and \makelabels belong to \documentclass{letter}, which is not available. Write a letter as ordinary paragraphs in an article document. Using them gives "Environment letter undefined".
- \\ ends a line *inside* a paragraph or a table row. It is not a blank line \
and not a paragraph separator: a \\ on a line of its own, or straight after \
\par, \vspace or \end{tabular}, gives "There's no line here to end" and the \
document does not build. Separate paragraphs with a blank line.
- Every row of a table, including the last one, ends with \\. A rule command (\toprule, \midrule, \bottomrule) goes on a line of its own, directly after a row's \\ or directly after \begin{tabular}. A row without its \\ before a rule gives "Misplaced \noalign".
- & separates columns and only means that inside a tabular, array or align. Outside one it is an ordinary character and must be written \&, which puts a literal "&" on the page. So do NOT use \& to separate a label from its value: write "Title: Early Learning", or put the two in a two-column tabular. A stray "&" on the page is text that was never in the data.
- The number of & separators in a row must be one less than the number of columns the tabular was declared with.

Escaping. Every one of these characters is reserved by LaTeX and must be \
written as shown when it appears in a value:
  &  ->  \&        %  ->  \%        $  ->  \$
  #  ->  \#        _  ->  \_        {  ->  \{        }  ->  \}
  ~  ->  \textasciitilde{}          ^  ->  \textasciicircum{}
  a single backslash  ->  \textbackslash{}
An unescaped one of these will not compile, or will silently typeset the wrong \
thing. "Bell & Co. 50% off_now" must be written "Bell \& Co. 50\% off\_now".

Fidelity. This is the part that matters most:
- Reproduce every value EXACTLY as given, character for character. Do not \
reword, abbreviate, expand, round, reformat, translate or correct anything. \
"$1,240.50" stays "$1,240.50" -- escaped as "\$1,240.50" -- and does not \
become "1240.50" or "$1,240.5" or "USD 1240.50".
- Include every record and every field you are given. Print each record's \
identifier as well as its fields.
- Do not invent a value, a field or a record that you were not given. Do not \
add totals, subtotals, counts or dates of your own.
- The examples in these instructions show form only. No amount, name, wording \
or value from them may appear in your document -- they are not data. Every fact \
on the page comes from the records you were given and from nowhere else.
- A field marked blank must be printed as an em dash (---), not omitted and not \
filled in.
- Where a record is marked as linked to another, show it inside or beneath that \
record's section. Do not print the linking id itself.

The document must be a plausible real-world document in the stated domain -- \
give it a heading and whatever caption, label or wording a real one would carry \
-- but every fact on it must come from the records you were given.
"""

_LAYOUT_INSTRUCTIONS = {
    "table": """\
Layout: a tabular document, of the kind an invoice or statement uses.
- Put the root record's fields in a header block at the top: a two-column \
tabular of label and value, or a simple run of labelled lines.
- Put the linked records in a table, one row per record, one column per field, \
with a header row. Use booktabs rules (\\toprule, \\midrule, \\bottomrule).
- Right-align columns holding money, percentages or counts.
- If a table would need more than five columns, break it into one small block \
per record instead of a wide table, so nothing runs off the page.""",
    "form": """\
Layout: a completed form, of the kind an office keeps on file.
- One section per record, each with a heading naming the record and its \
identifier.
- Inside a section, one labelled field per line or per table row: the label in \
bold, the value beside it.
- Draw a frame or a rule around each section so the sections read as boxes. Use \
\\fbox with a minipage, or \\hrule between sections.
- No prose. Labels and values only.""",
    "letter": """\
Layout: a formal letter, of the kind an organisation sends out.
- A letterhead, a date line and a reference line at the top, then a salutation, \
then body paragraphs, then a sign-off.
- State the values in running prose, in complete sentences. Each sentence names \
a field in ordinary words and gives that field's exact value. Compose the \
sentences yourself from the records you were given; do not copy a sentence out \
of these instructions and do not write a list of labelled fields.
- Linked records are described in the body, one short paragraph or one sentence \
each, naming each record's identifier as you go.
- Do not use a tabular. The whole point of this layout is that the same facts \
are in sentences.""",
}

LATEX_REPAIR_SYSTEM_PROMPT = r"""\
You fix LaTeX that failed to compile. You are given the source and the errors \
pdflatex reported, and you return the corrected source.

Output rules:
- Return ONLY the complete corrected LaTeX source, from \documentclass to \
\end{document}. No prose, no markdown fences, no explanation.
- Return the WHOLE document, not a patch or a fragment.

How to fix it:
- Read the error and the line it points at. Fix that.
- The commonest cause by far is an unescaped reserved character in a value: \
&, %, $, #, _, { or }. Escape it: \&, \%, \$, \#, \_, \{, \}.
- The next commonest is a package that is not installed. Only geometry, array, \
booktabs, longtable, tabularx, parskip, enumitem, fontenc and inputenc are \
available -- remove any other \usepackage line and whatever needed it.
- Then: a tabular whose row has the wrong number of & separators for its column \
specification, a missing \\ at the end of a row, an \end that does not match \
its \begin, or a command that does not exist.
- "Environment letter undefined" means \begin{letter}, \opening or \closing \
was used in an article document. Rewrite that part as ordinary paragraphs.
- "Misplaced \noalign" means a rule (\toprule, \midrule, \bottomrule) follows \
a table row that is missing its trailing \\. Add the \\.
- "There's no line here to end" means a \\ was used where there is no line to \
break -- on a line of its own, straight after \par, \vspace or \end{tabular}, \
or before the first paragraph. Delete that \\; separate the paragraphs with a \
blank line instead.
- "Missing $ inserted" or "Missing number" after a value usually means a \
reserved character in that value was not escaped.

What NOT to change:
- Do not change, drop, reword or reformat any value in the document. The values \
are the point of the document; only the LaTeX around them is broken.
- Do not remove content to make it compile. Every record and field in the \
broken source must still be in the fixed one.
"""

LATEX_RESTORE_SYSTEM_PROMPT = r"""\
You correct LaTeX documents whose data does not match its source records.

You are given a LaTeX document, and a list of values that are supposed to appear \
in it and do not. Put them back.

Output rules:
- Return ONLY the complete corrected LaTeX source, from \documentclass to \
\end{document}. No prose, no markdown fences, no explanation.

How to fix it:
- Each missing value belongs to the record named beside it. Find that record's \
section, row or paragraph and put the value there under its field name.
- If the value is present but altered -- reworded, rounded, abbreviated, \
reformatted -- replace what is there with the exact value given.
- If the record is missing from the document altogether, add it in the same \
style as the records that are there.
- Write the value character for character as given, escaping only LaTeX's \
reserved characters (& % $ # _ { } as \& \% \$ \# \_ \{ \}).
- Change nothing else. Every other value in the document must survive untouched.
"""


def layout_instruction(layout_style: str) -> str:
    """The layout half of the prompt, for one style."""
    try:
        return _LAYOUT_INSTRUCTIONS[layout_style]
    except KeyError:
        raise ValueError(
            f"unknown layout style {layout_style!r}; expected one of "
            f"{', '.join(sorted(_LAYOUT_INSTRUCTIONS))}") from None


class LaTeXGenerationError(RuntimeError):
    """The model never returned a usable LaTeX document."""


class LLMLaTeXGenerator:
    """Writes the LaTeX for one document scope, and repairs it when asked.

    ``schema`` is optional and only sharpens the prompt: knowing that ``total``
    is a currency and ``status`` an enum lets the model be told, which is what
    stops "$1,240.50" arriving as "1240.5".
    """

    def __init__(self, client: Any = None, *,
                 schema: Optional[SchemaGraph] = None,
                 seed: Optional[int] = None,
                 max_attempts: int = 3) -> None:
        if client is None:
            from .llm_bridge import build_client
            # json_mode off: ollama's "format": "json" makes the model emit a
            # JSON document, which is the one thing a LaTeX prompt must not get.
            client = build_client(seed=seed, json_mode=False, max_tokens=6144)
        self.client = client
        self.schema = schema
        self.seed = seed
        self.max_attempts = max(1, int(max_attempts))
        #: Raw text of the last response, for diagnosing a bad generation.
        self.last_response: Optional[str] = None

    # -- prompt construction ------------------------------------------------ #
    def _attribute_type(self, entity_name: str, attribute: str) -> str:
        if self.schema is None:
            return ""
        entity = self.schema.entity(entity_name)
        if entity is None:
            return ""
        attr = entity.attribute(attribute)
        return attr.type if attr is not None else ""

    def describe_records(self, document_records: Sequence[Record]) -> str:
        """The records, laid out for a prompt.

        One line per field, ``name (type) = value``, with the root record first
        and every other record marked as linked to its parent. Values are given
        raw, not escaped: the model is told how to escape, and giving it
        pre-escaped values would have it escape them twice.
        """
        if not document_records:
            raise ValueError("a document needs at least one record")
        root, children = document_records[0], document_records[1:]
        parent_of = {r.id: v for r in document_records
                     for v in r.foreign_keys.values()}

        lines: List[str] = []
        for position, record in enumerate([root] + list(children)):
            if position == 0:
                header = (f"MAIN RECORD -- {humanize(record.entity_name)} "
                          f"[identifier: {record.id}]")
            else:
                parent = parent_of.get(record.id)
                linked = (f", linked to {parent}"
                          if parent and parent in {r.id for r in document_records}
                          else "")
                header = (f"LINKED RECORD -- {humanize(record.entity_name)} "
                          f"[identifier: {record.id}{linked}]")
            lines.append(header)
            for name, value in record.attributes.items():
                if value == record.id:
                    continue  # already stated as the identifier
                attr_type = self._attribute_type(record.entity_name, name)
                label = f"{humanize(name)} ({attr_type})" if attr_type \
                    else humanize(name)
                shown = NULL_PRESENTATION if value is None else value
                lines.append(f"  {label} = {shown}")
            for column in record.orphaned_keys:
                value = record.foreign_keys.get(column)
                lines.append(f"  {humanize(column)} = {value} "
                             f"(a reference that matches no record -- print it "
                             f"and note that it could not be matched)")
            lines.append("")
        return "\n".join(lines).rstrip()

    def build_user_prompt(self, document_records: Sequence[Record],
                          layout_style: str, domain: str) -> str:
        count = len(document_records)
        return (
            f"Domain: {domain}\n"
            f"{layout_instruction(layout_style)}\n\n"
            f"Records for this document ({count} in total):\n\n"
            f"{self.describe_records(document_records)}\n\n"
            f"Write the complete LaTeX source for one document containing every "
            f"record above. Return the source only."
        )

    # -- public api --------------------------------------------------------- #
    def generate_latex_source(self, document_records: List[Record],
                              layout_style: str, domain: str) -> str:
        """Complete, compilable LaTeX for one document.

        Raises :class:`LaTeXGenerationError` when no attempt produced something
        that even looks like a document — a response with no ``\\documentclass``
        is not worth handing to pdflatex.
        """
        layout_instruction(layout_style)  # validates the style
        system = LATEX_GENERATION_SYSTEM_PROMPT
        user = self.build_user_prompt(document_records, layout_style, domain)
        reasons: List[str] = []

        for attempt in range(1, self.max_attempts + 1):
            prompt = user if attempt == 1 else (
                user + f"\n\nYour previous response could not be used: "
                       f"{reasons[-1]}. Return only the LaTeX source, starting "
                       f"with \\documentclass and ending with "
                       f"\\end{{document}}.")
            source = self._ask(system, prompt)
            if source is not None:
                return harden_source(source)
            reasons.append(self._failure_reason())
            log.warning("latex generation attempt %d/%d unusable: %s",
                        attempt, self.max_attempts, reasons[-1])

        raise LaTeXGenerationError(
            f"no usable LaTeX for {document_records[0].id} after "
            f"{self.max_attempts} attempt(s): " + "; ".join(reasons))

    def repair_latex_source(self, source: str, error_log: str,
                            document_records: Sequence[Record] = (),
                            layout_style: str = "", domain: str = "") -> str:
        """A corrected version of ``source``, given what pdflatex complained of.

        Returns the original source unchanged if the model produced nothing
        usable: handing the compiler the same input again wastes a pass, but it
        keeps the failure attributable to the document rather than to the repair.
        """
        prompt = (
            f"pdflatex reported these errors:\n\n{error_log.strip()}\n\n"
            f"Here is the source that produced them:\n\n{source}\n\n"
            f"Return the corrected complete source."
        )
        if document_records:
            prompt += (f"\n\nFor reference, the values that must survive "
                       f"unchanged:\n\n"
                       f"{self.describe_records(document_records)}")
        repaired = self._ask(LATEX_REPAIR_SYSTEM_PROMPT, prompt)
        if repaired is None:
            log.warning("repair produced no usable source: %s",
                        self._failure_reason())
            return source
        return harden_source(repaired)

    def restore_values(self, source: str,
                       missing: Sequence[Tuple[str, str, Any]],
                       document_records: Sequence[Record] = ()) -> str:
        """A version of ``source`` with dropped or altered values put back."""
        if not missing:
            return source
        listing = "\n".join(
            f"  record {record_id}, field {field}: {value}"
            for record_id, field, value in missing)
        prompt = (
            f"These values belong in the document and are missing from it, or "
            f"appear in an altered form:\n\n{listing}\n\n"
            f"Here is the document:\n\n{source}\n\n"
            f"Return the corrected complete source."
        )
        if document_records:
            prompt += (f"\n\nThe full record set for this document:\n\n"
                       f"{self.describe_records(document_records)}")
        restored = self._ask(LATEX_RESTORE_SYSTEM_PROMPT, prompt)
        if restored is None:
            log.warning("value restoration produced no usable source: %s",
                        self._failure_reason())
            return source
        return harden_source(restored)

    # -- transport ---------------------------------------------------------- #
    def _ask(self, system: str, user: str) -> Optional[str]:
        """One call, returning extracted LaTeX or ``None``."""
        self.last_response = None
        text = self.client.complete(system, user)
        self.last_response = text
        return extract_latex(text)

    def _failure_reason(self) -> str:
        if not self.last_response or not self.last_response.strip():
            return str(getattr(self.client, "last_error", None)
                       or "the model returned nothing")
        if "\\documentclass" not in self.last_response:
            return "the response contains no \\documentclass"
        return "the response has no \\end{document}"


__all__ = [
    "LLMLaTeXGenerator",
    "LaTeXGenerationError",
    "LATEX_GENERATION_SYSTEM_PROMPT",
    "LATEX_REPAIR_SYSTEM_PROMPT",
    "LATEX_RESTORE_SYSTEM_PROMPT",
    "LAYOUT_STYLES",
    "ALLOWED_PACKAGES",
    "HYPHENATION_GUARD",
    "NULL_GLYPH",
    "LatexText",
    "escape_latex",
    "unescape_latex",
    "normalize_for_comparison",
    "extract_latex",
    "harden_source",
    "layout_instruction",
    "recorded_values",
    "missing_values",
    "leaked_examples",
    "PROMPT_EXAMPLE_VALUES",
    "humanize",
]

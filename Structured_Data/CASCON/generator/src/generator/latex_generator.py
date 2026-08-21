"""Stage 5a: ask local Llama 3 to write the LaTeX for one document.

    List[Record] + layout_hint -> LLMLaTeXGenerator -> LaTeX source

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

There is no list of layouts. There used to be — ``table``, ``form``, ``letter``,
three paragraphs of hand-written instruction each — and a corpus generated from
it had three shapes however many documents it held, which is the same failure the
Jinja templates had with an extra step. The layout is now *invented per
document*: :func:`layout_directive` tells the model to read the subgraph it has
been given and work out what document those records would really live on in that
domain, then design that page. A caller who wants a particular look passes the
look as a sentence — ``"1990s technical spec sheet with dense grid lines"`` — and
it is handed to the model as a stylistic brief, not matched against anything.

Two consequences are dealt with here rather than left to the caller:

* **Variety has to be provoked.** ``--seed`` pins the decoder and forces greedy
  decoding, so the same prompt returns the same page every time. The directive
  therefore varies per document — a rotating lead-off device and a rotating axis
  to push on — which is what makes a seeded run reproducible *and* varied
  instead of reproducible and uniform.
* **What the model invented has to be recoverable.** The page is asked to
  declare its own layout in a ``% LAYOUT:`` comment, which :func:`layout_declaration`
  reads back so the manifest can state, per PDF, the layout that was actually
  produced rather than the hint that was requested. Comments are stripped before
  the fidelity check for the same reason a value must be *on the page*: text in a
  comment is not typeset, so it cannot stand in for a value that is missing.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .instance_types import Record
from .schema_types import SchemaGraph

log = logging.getLogger(__name__)

#: The one hint that is not a stylistic brief: it asks the model to invent the
#: layout from the records themselves. Every other value of ``layout_hint`` is
#: freeform text and is passed through to the model verbatim.
AUTO_LAYOUT = "auto"

#: Packages the model may use. Anything else is a compile failure on a machine
#: with a minimal TeX install, and the repair loop cannot install packages.
#:
#: Wider than it was, and deliberately: a model told to invent a page and then
#: denied columns, rules and shading can only invent variations on a single
#: column of text. ``multicol``, ``xcolor``, ``colortbl``, ``ragged2e`` and
#: ``setspace`` are what "a shaded inspection box beside a two-column body" needs
#: to exist at all. All of them ship with a standard TeX Live installation.
ALLOWED_PACKAGES = ("geometry", "array", "booktabs", "longtable", "parskip",
                    "tabularx", "enumitem", "multicol", "xcolor", "colortbl",
                    "ragged2e", "setspace", "fontenc", "inputenc")

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
#: A LaTeX comment: an unescaped ``%`` to the end of its line. The capture group
#: is the run of backslash-pairs before it, which are real backslashes rather
#: than an escape of the ``%``, and so must survive the removal. ``\%`` is an
#: escaped percent sign in a value and is not a comment, which is why the
#: lookbehind is there.
#:
#: Stripped before anything else, and before unescaping, because a value only
#: counts as present when it is *typeset*: the generator asks each page to
#: declare its invented layout in a ``% LAYOUT:`` comment, and a comment naming a
#: record would otherwise satisfy a fidelity check for a value that is nowhere on
#: the page.
_COMMENT = re.compile(r"(?<!\\)((?:\\\\)*)%[^\n]*")


def strip_comments(source: str) -> str:
    """Drop every LaTeX comment from ``source``, keeping escaped ``\\%``."""
    return _COMMENT.sub(lambda m: m.group(1), str(source))


def normalize_for_comparison(source: str) -> str:
    """Flatten a LaTeX source into something a raw value can be searched in.

    Drops comments, turns page breaks into spaces, de-escapes, drops the
    formatting commands a value might be wrapped in, turns ``~`` and ``\\,`` into
    spaces, and collapses whitespace — because a value the model broke across a
    line, bolded, or spaced with a tie is still that value on the page. What
    survives all that and still does not match is a value that was actually
    reworded, rounded, truncated or dropped.
    """
    text = _BREAKS.sub(" ", strip_comments(source))
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
#: A line in a prompt below that ends in a single backslash is soft-wrapped
#: source, not content. See :func:`_unfold`.
_FOLD = re.compile(r"[ \t]*(?<!\\)\\\n[ \t]*")


def _unfold(text: str) -> str:
    r"""Join the lines a raw string left broken by a trailing backslash.

    The prompts are raw strings because they are largely LaTeX: ``\&`` and
    ``\documentclass`` have to reach the model exactly as written, and in an
    ordinary string every one of those backslashes would have to be doubled.
    The cost is that a raw string does *not* honour a backslash-newline as a
    line continuation, so source wrapped at a readable width keeps a literal
    backslash and a newline in the middle of each sentence. That is worse than
    untidy: the prompt tells a model to write LaTeX, and a stray ``\`` in it
    reads as the start of a command.

    Unfolding here keeps the source wrapped and the prompt in whole sentences.
    A *single* trailing backslash is always a wrap; ``\\`` is LaTeX's own
    line break and is left alone, which is what the lookbehind is for.
    """
    return _FOLD.sub(" ", text).lstrip()


LATEX_GENERATION_SYSTEM_PROMPT = _unfold(r"""\
You are a document designer who writes LaTeX. You are given the records that \
belong on one document and a brief for how it should look, and you design that \
document and return the complete source for it.

You are not filling in a template. Nothing you are given names a layout for you \
to reproduce; the shape of the page is yours to decide, and deciding it well -- \
so that the page looks like a real document of its kind rather than a dump of \
the data it carries -- is the job.

Output rules:
- Return ONLY LaTeX source. No prose before or after it, no markdown fences.
- Start at \documentclass and end at \end{document}. Include the preamble.
- The line straight after \documentclass is a single comment beginning \
"% LAYOUT:" and then one plain sentence, written by you, naming the kind of \
document you decided to write and the visual structure you gave it. It is a \
description of what you did, not a slot to fill in, and it is the only comment \
that has to be there.
- Use only these packages: geometry, array, booktabs, longtable, tabularx, \
multicol, xcolor, colortbl, ragged2e, setspace, parskip, enumitem, fontenc, \
inputenc. Nothing else is installed.
- Use \documentclass{article}. Do not use tikz, minted, hyperref, fancyhdr, \
tcolorbox, graphicx or any package not listed above.
- Do not use \write18, \input or \include. Do not reference an external file, \
an image or a logo: nothing outside this source exists.

What you may build the page out of. This is what is installed, not a menu of \
layouts -- combine these however the document you are designing needs:
- multicol for a two- or three-column body, and minipages side by side for \
panels that sit next to each other.
- xcolor and colortbl for a shaded box, a tinted table row, a rule in a colour, \
a reversed heading bar. \colorbox, \fcolorbox, \rowcolors and >{\columncolor{...}} \
all work. Keep any shade light enough for black text to stay readable.
- booktabs, array and tabularx for rules and column widths; \hline and vertical \
bars in the column spec for a dense ruled grid where that is the look you want.
- geometry for the page size and margins, setspace for line spacing, ragged2e \
for justification, and font-size and font-shape commands for typographic \
contrast.
- \fbox, \framebox, \parbox, \rule and \hrule for boxes, stamps and separators.

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

Fidelity. This is the part that matters most, and it does not bend for the \
design. Whatever page you invent, 100% of what you were given has to be on it:
- EVERY record you are given appears on the page. Every field of every record \
appears on the page, under a label a reader can tell it by. Every record's \
identifier is printed as well as its fields. A linked child record does not lose \
fields for being a child, and the last child is as complete as the first.
- Reproduce every value EXACTLY as given, character for character. Do not \
reword, abbreviate, expand, round, reformat, translate or correct anything. \
"$1,240.50" stays "$1,240.50" -- escaped as "\$1,240.50" -- and does not \
become "1240.50" or "$1,240.5" or "USD 1240.50".
- If the layout you designed has no room for a field, the layout is wrong. \
Change the design; do not drop the field, summarise it away, replace it with \
"et al.", "and others", "(see attached)" or an ellipsis, or cut a table short.
- Do not invent a value, a field or a record that you were not given. Do not \
add totals, subtotals, counts, dates, addresses, reference numbers, signatures, \
prices, contact details or legal wording of your own. Headings, captions, column \
titles and connecting words are yours to write; facts are not. If a real \
document of this kind would carry a fact you were not given, leave it out -- an \
absent field is a gap, an invented one is a lie the ground truth cannot see.
- The examples in these instructions show form only. No amount, name, wording \
or value from them may appear in your document -- they are not data. Every fact \
on the page comes from the records you were given and from nowhere else.
- A field marked blank must be printed as an em dash (---), not omitted and not \
filled in.
- Where a record is marked as linked to another, show it inside or beneath that \
record's section. Do not print the linking id itself.
- A value written in a LaTeX comment is not on the page. Comments are not \
typeset, so a field mentioned only in one has been dropped.

The document must be a plausible real-world document in the stated domain -- \
give it a heading and whatever caption, label or wording a real one would carry \
-- but every fact on it must come from the records you were given.
""")

#: Devices a real document might be built out of, named as *devices* and not as
#: documents. The list is shown to the model as evidence of how wide the range
#: is, with a standing instruction that it is not a menu — naming three finished
#: layouts is what produced a corpus with three shapes, and naming ten would
#: produce one with ten. Nothing here carries a value, a name or an amount, so
#: nothing here can be copied onto a page as a fact (see PROMPT_EXAMPLE_VALUES).
DESIGN_DEVICES = (
    "a body set in two or three columns",
    "a shaded inspection or verification box",
    "a line-item financial table that rules off between groups",
    "a run of prose -- a log entry, a narrative note, a summary paragraph",
    "a formal masthead with a reference block ruled off beneath it",
    "metric panels sitting side by side across the page",
    "correspondence, written to somebody, signed off at the end",
    "a stamped or boxed status flag set against the heading",
    "a dense grid of small labelled cells",
    "a signature and countersignature block",
    "a tinted header row over a plain body, or the reverse",
    "a narrow margin column carrying labels beside a wide one carrying values",
)

#: One axis to push on per document. Paired with the rotation over
#: DESIGN_DEVICES, this is what makes a *seeded* run varied: --seed forces
#: greedy decoding, so an identical prompt returns an identical page, and the
#: only way to get twelve different documents out of a deterministic decoder is
#: to send it twelve different prompts.
DESIGN_AXES = (
    "the page geometry -- margins, page shape, how much white space there is",
    "the number of columns and how the page divides horizontally",
    "typographic contrast -- size, weight, small caps, letter spacing",
    "how rules, frames and shading separate one part of the page from another",
    "the balance between prose and tabular material",
    "the order things are presented in, and what the eye is meant to reach first",
)

LAYOUT_INVENTION_DIRECTIVE = """\
Layout: yours to invent, for this document, now.

Read the records below before you decide anything: the domain, the entity names, \
the field names, and the kind of value each field holds. Work out what document \
these particular records would really live on in that domain -- what it would be \
called, who would issue it, who would read it, what a filed copy of it looks \
like -- and then design that document. Give it the heading, the sections, the \
captions and the page shape a real one would have.

Do not reach for the obvious shape. If the records suggest one arrangement \
immediately, that is the arrangement every generic rendering of this data would \
use; find the one an organisation in this domain would actually print. Two \
documents built from similar records should not come out looking like each \
other."""

_DEVICE_PREAMBLE = """\
For a sense of the range available -- these are devices, not documents, and not \
a menu to choose from. Mix them, ignore them, or build something that is not \
here:"""

_FREEFORM_DIRECTIVE = """\
Layout: follow the brief below. It was written by the person requesting this \
document and is quoted to you exactly as they wrote it.

{brief}

Read it as a description of how the page should look and feel, and invent the \
LaTeX structure that realises it -- the sections, the rules, the columns, the \
typography. The brief says what the document should be like; it does not say \
where anything goes, and that is your decision.

The brief describes form only. Nothing in it is a fact about these records: no \
wording, name, number or date from it may appear on the page unless a record \
below supplies it."""

#: Added when one subgraph is being rendered more than once. Deliberately names
#: no layout for any variant: handing variant 1 "an invoice" and variant 2 "a
#: memo" would be the fixed enum this stage was rewritten to remove, reappearing
#: one level down and capped at however many names the list held. What makes the
#: variants differ instead is the same thing that makes neighbouring documents
#: differ -- a rotated lead-off device and axis, driven by a distinct
#: ``variation`` per variant -- plus the standing instruction below that
#: resembling a sibling is a failure.
_VARIANT_DIRECTIVE = """\
This is layout {index} of {total} for one single set of records.

The other {others} in this set carry exactly the same records as this one, field \
for field and value for value, and differ only in how they are laid out. They \
exist so that a corpus can tell an extractor that has understood the data apart \
from one that has memorised a shape, which only works if the shapes really are \
different.

So this page must not resemble its siblings, and the difference has to be \
structural rather than cosmetic. Change what kind of document this is, how the \
page divides, what the eye reaches first, and whether the material is set as \
prose, as panels or as a table. Two variants that differ only in their heading, \
their wording or their choice of rule are one variant and a wasted page.

None of this touches the data. Every record and every field is on this page as \
completely as on any other variant -- a different shape is not licence to carry \
less."""


def variant_directive(variant: int, variant_count: int) -> str:
    """The "this is one of several layouts" clause, or ``""`` for a lone page.

    ``variant`` is zero-based, matching the loop that renders it; the wording
    counts from one, matching what the filenames say.
    """
    total = max(1, int(variant_count))
    if total < 2:
        return ""
    index = min(max(0, int(variant)), total - 1) + 1
    others = (f"{total - 1} document" if total == 2
              else f"{total - 1} documents")
    return _VARIANT_DIRECTIVE.format(index=index, total=total, others=others)


def normalize_layout_hint(layout_hint: Any) -> str:
    """The hint as the rest of the module wants it: stripped, never empty.

    ``None``, ``""`` and whitespace all mean "no preference stated", which is
    :data:`AUTO_LAYOUT` — the same thing the default asks for. There is nothing
    to validate beyond that: any other string is a brief, and a brief cannot be
    wrong.
    """
    text = "" if layout_hint is None else str(layout_hint).strip()
    return text or AUTO_LAYOUT


def is_auto(layout_hint: Any) -> bool:
    """Whether ``layout_hint`` asks the model to invent the layout itself."""
    return normalize_layout_hint(layout_hint).lower() == AUTO_LAYOUT


def layout_directive(layout_hint: str = AUTO_LAYOUT, *,
                     variation: int = 0, variant: int = 0,
                     variant_count: int = 1) -> str:
    """The layout half of the prompt, built for one document.

    ``variation`` is the document's index in the run. It rotates which device
    leads the list and which axis is called out, so consecutive documents in one
    corpus are asked for visibly different pages even under a pinned seed. It
    changes the *prompt*, never the records, so the ground truth is untouched by
    it and a rerun at the same seed reproduces the same corpus.

    ``variant`` and ``variant_count`` describe this page's place among the
    layouts of *one* subgraph, under ``--layouts-per-graph``. They add the
    clause that says so; the visible difference between variants comes from
    ``variation``, which the caller varies per variant for exactly that reason.
    A ``variant_count`` of 1 adds nothing, so a single-layout run sends the same
    prompt it always did.
    """
    hint = normalize_layout_hint(layout_hint)
    parts = [LAYOUT_INVENTION_DIRECTIVE if is_auto(hint)
             else _FREEFORM_DIRECTIVE.format(brief=_quote_brief(hint))]

    siblings = variant_directive(variant, variant_count)
    if siblings:
        parts.append(siblings)

    offset = max(0, int(variation)) % len(DESIGN_DEVICES)
    devices = DESIGN_DEVICES[offset:] + DESIGN_DEVICES[:offset]
    parts.append(_DEVICE_PREAMBLE + "\n"
                 + "\n".join(f"  - {device}" for device in devices))

    axis = DESIGN_AXES[max(0, int(variation)) % len(DESIGN_AXES)]
    parts.append(f"For this document in particular, push hardest on {axis}. "
                 f"Whatever you settle on, the page has to stay legible and "
                 f"has to fit the paper: no column so narrow that a value "
                 f"breaks up inside it, no table wider than the page.")
    return "\n\n".join(parts)


def _quote_brief(hint: str) -> str:
    """A freeform brief, indented so the model can see where it ends."""
    return "\n".join(f"    {line.strip()}"
                     for line in hint.splitlines() if line.strip())


#: How a page states the layout it turned out to be. Asked for in the system
#: prompt, read back by :func:`layout_declaration`, and recorded per PDF in the
#: manifest — the hint says what was requested, this says what was produced.
LAYOUT_DECLARATION_PREFIX = "% LAYOUT:"
_DECLARATION = re.compile(r"^[ \t]*%+[ \t]*LAYOUT[ \t]*:[ \t]*(\S[^\n]*?)[ \t]*$",
                          re.MULTILINE | re.IGNORECASE)
#: Long enough for a sentence, short enough that a model which ignored the
#: instruction and wrote an essay cannot bloat every manifest entry.
MAX_DECLARATION_LENGTH = 300


def layout_declaration(source: Optional[str]) -> Optional[str]:
    """The layout the page says it is, or ``None`` if it never said.

    ``None`` is a normal outcome, not an error: the declaration is a comment the
    model was asked to write, and a page that compiles and carries its data is
    correct whether or not it introduced itself. The caller falls back to the
    hint that was sent.
    """
    if not source:
        return None
    match = _DECLARATION.search(str(source))
    if match is None:
        return None
    declared = _WHITESPACE.sub(" ", match.group(1)).strip()
    if len(declared) > MAX_DECLARATION_LENGTH:
        declared = declared[:MAX_DECLARATION_LENGTH].rstrip() + "..."
    return declared or None

LATEX_REPAIR_SYSTEM_PROMPT = _unfold(r"""\
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
booktabs, longtable, tabularx, multicol, xcolor, colortbl, ragged2e, setspace, \
parskip, enumitem, fontenc and inputenc are available -- remove any other \
\usepackage line and whatever needed it.
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
- Do not redesign the page. The layout is this document's own -- keep its \
columns, boxes, shading, headings and section order, and keep the "% LAYOUT:" \
comment line exactly as it is. Simplifying the design is not a fix; it is a \
different document.
""")

LATEX_RESTORE_SYSTEM_PROMPT = _unfold(r"""\
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
- A value written into a LaTeX comment has not been put back: comments are not \
typeset. Put it where it will print.
- Keep the page as it is designed. Add the missing value in the style of the \
section it belongs to; do not restructure the document around it.
- Change nothing else. Every other value in the document must survive untouched.
""")


class LaTeXGenerationError(RuntimeError):
    """The model never returned a usable LaTeX document."""


class LLMLaTeXGenerator:
    """Writes the LaTeX for one document scope, and repairs it when asked.

    ``schema`` is optional and only sharpens the prompt: knowing that ``total``
    is a currency and ``status`` an enum lets the model be told, which is what
    stops "$1,240.50" arriving as "1240.5".

    ``layout_hint`` is freeform and is the default for every document this
    generator writes; a per-call hint overrides it. ``"auto"`` — the default —
    asks the model to invent a layout from the records themselves. Anything else
    is a stylistic brief ("1990s technical spec sheet with dense grid lines")
    and is passed through verbatim. There is no set of accepted values, so
    nothing here rejects a hint.
    """

    def __init__(self, client: Any = None, *,
                 schema: Optional[SchemaGraph] = None,
                 layout_hint: str = AUTO_LAYOUT,
                 seed: Optional[int] = None,
                 max_attempts: int = 3) -> None:
        if client is None:
            from .llm_bridge import build_client
            # json_mode off: ollama's "format": "json" makes the model emit a
            # JSON document, which is the one thing a LaTeX prompt must not get.
            client = build_client(seed=seed, json_mode=False, max_tokens=6144)
        self.client = client
        self.schema = schema
        self.layout_hint = normalize_layout_hint(layout_hint)
        self.seed = seed
        self.max_attempts = max(1, int(max_attempts))
        #: Raw text of the last response, for diagnosing a bad generation.
        self.last_response: Optional[str] = None
        #: The layout half of the last prompt sent, verbatim. Read by the
        #: renderer so the manifest can record what each PDF was actually asked
        #: for, which under a rotating directive differs document to document.
        self.last_layout_directive: Optional[str] = None

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

    def resolve_layout_hint(self, layout_hint: Any = None) -> str:
        """The hint for one call: the one passed, else this generator's own."""
        if layout_hint is None:
            return self.layout_hint
        return normalize_layout_hint(layout_hint)

    def build_layout_directive(self, layout_hint: Any = None, *,
                               variation: int = 0, variant: int = 0,
                               variant_count: int = 1) -> str:
        """The exact layout wording one document will be sent."""
        return layout_directive(self.resolve_layout_hint(layout_hint),
                                variation=variation, variant=variant,
                                variant_count=variant_count)

    def build_user_prompt(self, document_records: Sequence[Record],
                          layout_hint: Any = None, domain: str = "", *,
                          variation: int = 0, variant: int = 0,
                          variant_count: int = 1) -> str:
        count = len(document_records)
        directive = self.build_layout_directive(
            layout_hint, variation=variation, variant=variant,
            variant_count=variant_count)
        return (
            f"Domain: {domain}\n"
            f"{directive}\n\n"
            f"Records for this document ({count} in total):\n\n"
            f"{self.describe_records(document_records)}\n\n"
            f"Design one document that carries every record above, and write "
            f"its complete LaTeX source. Return the source only."
        )

    # -- public api --------------------------------------------------------- #
    def generate_latex_source(self, document_records: List[Record],
                              layout_hint: Any = None, domain: str = "", *,
                              variation: int = 0, variant: int = 0,
                              variant_count: int = 1) -> str:
        """Complete, compilable LaTeX for one document.

        ``layout_hint`` is freeform and unvalidated — ``"auto"`` to have the
        model invent the page, any other text as a stylistic brief.
        ``variation`` distinguishes this document from its neighbours in the
        same run, which is what stops a seeded corpus coming out uniform.

        ``variant`` and ``variant_count`` say that this page is one of several
        layouts of the *same* records, under ``--layouts-per-graph``. They only
        change the wording; the records are handed over unchanged, because a
        variant that carried different data would not be a layout variant.

        Raises :class:`LaTeXGenerationError` when no attempt produced something
        that even looks like a document — a response with no ``\\documentclass``
        is not worth handing to pdflatex.
        """
        system = LATEX_GENERATION_SYSTEM_PROMPT
        hint = self.resolve_layout_hint(layout_hint)
        self.last_layout_directive = self.build_layout_directive(
            hint, variation=variation, variant=variant,
            variant_count=variant_count)
        user = self.build_user_prompt(document_records, hint, domain,
                                      variation=variation, variant=variant,
                                      variant_count=variant_count)
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
                            layout_hint: Any = None, domain: str = "") -> str:
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
    "AUTO_LAYOUT",
    "ALLOWED_PACKAGES",
    "DESIGN_DEVICES",
    "DESIGN_AXES",
    "LAYOUT_INVENTION_DIRECTIVE",
    "LAYOUT_DECLARATION_PREFIX",
    "HYPHENATION_GUARD",
    "NULL_GLYPH",
    "LatexText",
    "escape_latex",
    "unescape_latex",
    "strip_comments",
    "normalize_for_comparison",
    "extract_latex",
    "harden_source",
    "layout_directive",
    "variant_directive",
    "layout_declaration",
    "normalize_layout_hint",
    "is_auto",
    "recorded_values",
    "missing_values",
    "leaked_examples",
    "PROMPT_EXAMPLE_VALUES",
    "humanize",
]

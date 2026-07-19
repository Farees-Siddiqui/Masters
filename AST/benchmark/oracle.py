"""Real ground truth for AST<->PDF alignment, derived from the PDF text layer.

The synthetic benchmark (:mod:`benchmark.synthetic`) generates its boxes *from*
the AST's own text, so both sides of the match descend from one source. That
measures robustness to character noise, not the real task: reconciling two
*independent* engines (Mistral OCR on the Doc side, PP-OCRv5 on the PDF side)
whose disagreements are structured — hyphenation, ligatures, dropped page
furniture, LaTeX vs. garbled formulas — rather than IID noise.

This module supplies a **referee** neither engine can influence: a born-digital
PDF's embedded text layer, which records exactly what glyph sits at exactly what
coordinate. Gold is derived in two independent steps:

1. **Geometry** — each PDF box owns the text-layer tokens whose centres fall
   inside it. Pure coordinate containment; no OCR is trusted.
2. **Text** — each AST node is placed into the text-layer token stream, giving
   ``node -> token span``. This is a *strictly easier* matching problem than the
   one under test (Mistral's transcription vs. the true text, near-identical)
   and, unlike the aligners, it may use the text layer itself.

Composing the two gives ``token -> node`` gold. Tokens no node claims (page
numbers, figure interiors, running heads) fall out as the **NULL class** for
free — the case the synthetic gold structurally cannot express, and the one that
matters most when alignment is used for provenance.

**Honesty constraints.** The oracle grades the benchmark, so it is itself
reported on, never assumed: :func:`derive_gold` returns an
:class:`OracleReport` carrying the node placement rate, token coverage, and
conflict count. Nodes it cannot place (typically formulas, where Mistral emits
LaTeX and no literal glyph run exists) are *excluded* from gold rather than
guessed — which makes the benchmark easier than reality, so the exclusion rate
is reported as a headline number rather than buried.

Only born-digital PDFs have a text layer; this is a benchmark instrument, not a
production path.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from alignment.naive_aligner import _iter_segments

_WS = re.compile(r"\s+")

# A node must match its located window this well to be trusted into gold. Below
# this we drop the node rather than invent a placement.
_MIN_PLACE_RATIO = 0.75

# Nodes shorter than this (normalized chars) are too weak to place unambiguously
# ("5", "•", a bare section number) — they'd anchor anywhere.
_MIN_PLACE_CHARS = 4


@dataclass(frozen=True)
class Token:
    """One whitespace-delimited word from the PDF text layer.

    ``bbox`` is in **rendered-pixel space** (top-left origin, y down) to match
    the layout engine's boxes — see :func:`_charbox_to_px` for the transform off
    the PDF's native bottom-left/points coordinate system.

    ``fragments`` holds one rect per *line* the word occupies: an ordinary word
    has a single fragment equal to ``bbox``, while a word hyphenated across a
    line break (pdfium's U+FFFE marker) carries one rect per half. ``bbox``
    stays the union of all fragments, so single-rect consumers keep working —
    but for a hyphenated word that union spans two lines and most of the
    column, so anything geometric (containment, drawing) should prefer
    ``fragments``.
    """

    page: int
    index: int  # index within the page
    text: str
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) pixels
    stream_start: int  # offset into the normalized document stream
    stream_end: int
    # Per-line sub-rects; defaults to [bbox] for ordinary single-line words.
    fragments: list[tuple[float, float, float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.fragments:
            object.__setattr__(self, "fragments", [self.bbox])

    @property
    def key(self) -> str:
        return f"{self.page}:{self.index}"

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2, (y0 + y1) / 2)


@dataclass
class OracleReport:
    """Diagnostics that decide whether the gold is trustworthy at all."""

    n_tokens: int = 0
    n_nodes: int = 0
    n_placed: int = 0
    n_unplaced: int = 0
    unplaced_samples: list[str] = field(default_factory=list)
    unplaced_nodes: list[str] = field(default_factory=list)  # every unplaced id
    n_conflicts: int = 0
    n_gold_tokens: int = 0  # tokens owned by some node
    n_null_tokens: int = 0  # tokens owned by no node (page furniture, figures)
    # Per-placement provenance for the human-audit tool: node_id -> {"start",
    # "end", "ratio", "exact"}. "exact" means a literal substring hit; fuzzy
    # placements carry the SequenceMatcher ratio that admitted them.
    placement_meta: dict[str, dict] = field(default_factory=dict)

    @property
    def placed_node_ids(self) -> set[str]:
        """Ids the oracle actually placed — the set gold can vouch for."""
        return set(self.placement_meta)

    @property
    def placement_rate(self) -> float:
        return self.n_placed / self.n_nodes if self.n_nodes else 0.0

    @property
    def token_coverage(self) -> float:
        return self.n_gold_tokens / self.n_tokens if self.n_tokens else 0.0

    def to_dict(self) -> dict:
        return {
            "n_tokens": self.n_tokens,
            "n_nodes": self.n_nodes,
            "n_placed": self.n_placed,
            "n_unplaced": self.n_unplaced,
            "placement_rate": round(self.placement_rate, 4),
            "token_coverage": round(self.token_coverage, 4),
            "n_conflicts": self.n_conflicts,
            "n_gold_tokens": self.n_gold_tokens,
            "n_null_tokens": self.n_null_tokens,
            "unplaced_samples": self.unplaced_samples[:10],
            "unplaced_nodes": list(self.unplaced_nodes),
        }


def _norm(s: str) -> str:
    """Normalize for matching: NFKC (folds ligatures), lowercase, collapse space.

    NFKC maps the ligature 'ﬁ' to 'fi', so a text layer that stores the glyph
    and an AST that stores the letters compare equal.

    Line-break hyphens are dropped. pdfium encodes a *soft* (layout-induced)
    hyphen as U+FFFE and leaves it inside the word, so 'ex￾tremely' heals
    to 'extremely' while a real ASCII hyphen in 'low-level' — a hyphen that
    belongs to the word — survives untouched. That distinction is why we key off
    the codepoint instead of guessing from line geometry.
    """
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("￾", "")  # pdfium's soft/line-break hyphen
    s = s.replace("­", "")  # soft hyphen
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-")
    return _WS.sub(" ", s).strip().lower()


# Punctuation that never takes a leading space, and openers that take no
# trailing one. The text layer positions a superscript marker or a comma as its
# own run, so naively joining tokens with spaces invents 'competitions1 ,' where
# both the real text and the AST read 'competitions1,'.
_NO_SPACE_BEFORE = set(",.;:!?)]}%")
_NO_SPACE_AFTER = set("([{$")


def _needs_space(prev: str, nxt: str) -> bool:
    if not prev or not nxt:
        return False
    if nxt[0] in _NO_SPACE_BEFORE:
        return False
    if prev[-1] in _NO_SPACE_AFTER:
        return False
    return True


def _charbox_to_px(
    charbox: tuple[float, float, float, float], page_h_pt: float, scale: float
) -> tuple[float, float, float, float]:
    """PDF points (bottom-left origin, y up) -> rendered pixels (top-left, y down).

    Getting this wrong yields a gold map that is plausible, self-consistent, and
    describes nothing — so it is verified against known layout boxes in the
    oracle's self-check rather than trusted.
    """
    left, bottom, right, top = charbox
    return (
        left * scale,
        (page_h_pt - top) * scale,  # top edge
        right * scale,
        (page_h_pt - bottom) * scale,  # bottom edge
    )


# Characters that mark a layout-induced break *inside* a word: pdfium's
# U+FFFE line-break hyphen and the Unicode soft hyphen. Both are stripped from
# the text stream by _norm; geometrically they split a token into fragments.
_FRAG_BREAKS = {"￾", "­"}


def _union_rects(
    rects: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    return (
        min(r[0] for r in rects),
        min(r[1] for r in rects),
        max(r[2] for r in rects),
        max(r[3] for r in rects),
    )


def extract_tokens(pdf_path: str | Path, dpi: int = 200) -> list[Token]:
    """Read every page's text layer into pixel-space tokens.

    Tokens are whitespace-delimited, so a word hyphenated across a line break
    is **one token**: pdfium marks the break with U+FFFE *inside* the word
    ('compet\\ufffeitive'), which is not whitespace. The marker is later
    stripped by :func:`_norm`, healing the stream text to 'competitive'.
    Geometrically the marker splits the token into per-line ``fragments`` —
    one rect per half — while ``bbox`` remains the union of all fragments
    (which for a hyphenated word spans two lines and most of the column, so
    prefer ``fragments`` for anything geometric).
    """
    import pypdfium2 as pdfium

    scale = dpi / 72.0
    doc = pdfium.PdfDocument(str(pdf_path))
    tokens: list[Token] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        _, h_pt = page.get_size()
        textpage = page.get_textpage()
        n_chars = textpage.count_chars()
        raw = textpage.get_text_range()

        # Group chars into words, tracking each word's char indices for its bbox.
        groups: list[list[int]] = []
        cur: list[int] = []
        for i in range(min(n_chars, len(raw))):
            if raw[i].isspace():
                if cur:
                    groups.append(cur)
                    cur = []
            else:
                cur.append(i)
        if cur:
            groups.append(cur)

        page_tokens: list[Token] = []
        for idx, char_idx in enumerate(groups):
            text = "".join(raw[i] for i in char_idx)
            boxes_px: dict[int, tuple[float, float, float, float]] = {}
            for i in char_idx:
                try:
                    box = textpage.get_charbox(i)
                except Exception:
                    continue
                boxes_px[i] = _charbox_to_px(box, h_pt, scale)
            if not boxes_px:
                continue

            # Split at the line-break/soft-hyphen markers: each side of the
            # break becomes its own fragment, so a hyphenated word carries one
            # rect per line instead of a giant two-line union.
            runs: list[list[int]] = []
            run: list[int] = []
            for i in char_idx:
                if raw[i] in _FRAG_BREAKS:
                    if run:
                        runs.append(run)
                        run = []
                else:
                    run.append(i)
            if run:
                runs.append(run)

            fragments: list[tuple[float, float, float, float]] = []
            for run in runs:
                pxs = [boxes_px[i] for i in run if i in boxes_px]
                if pxs:
                    fragments.append(_union_rects(pxs))
            if not fragments:
                # Only break chars carried boxes — fall back to their union so
                # the token survives exactly as it did before fragments.
                fragments = [_union_rects(list(boxes_px.values()))]

            page_tokens.append(
                Token(
                    page=page_index + 1,
                    index=idx,
                    text=text,
                    bbox=_union_rects(fragments),
                    stream_start=-1,  # assigned in _build_stream
                    stream_end=-1,
                    fragments=fragments,
                )
            )
        tokens.extend(page_tokens)

    return tokens


def build_stream(tokens: list[Token]) -> tuple[str, list[Token]]:
    """Join tokens into one normalized document stream, recording each span.

    Returns ``(stream, tokens)`` where every token carries the ``[stream_start,
    stream_end)`` slice of ``stream`` it contributed. Hyphenated line breaks are
    healed here (see :func:`extract_tokens`).
    """
    # Normalize first so the separator decision sees final text.
    normed = [(tok, _norm(tok.text)) for tok in tokens]
    normed = [(tok, piece) for tok, piece in normed if piece]

    parts: list[str] = []
    out: list[Token] = []
    pos = 0

    for i, (tok, piece) in enumerate(normed):
        start = pos
        parts.append(piece)
        pos += len(piece)
        end = pos

        out.append(
            Token(
                page=tok.page,
                index=tok.index,
                text=tok.text,
                bbox=tok.bbox,
                stream_start=start,
                stream_end=end,
                fragments=tok.fragments,
            )
        )

        if i + 1 < len(normed) and _needs_space(piece, normed[i + 1][1]):
            parts.append(" ")
            pos += 1

    return "".join(parts), out


def _locate(
    stream: str, needle: str, cursor: int
) -> tuple[int, int, float, bool] | None:
    """Find ``needle`` in ``stream``, preferring the first hit at/after ``cursor``.

    The cursor makes placement **monotonic**, which is what disambiguates
    repeated text: boilerplate that appears five times resolves to the
    occurrence that follows the previous node, matching reading order, instead
    of always collapsing onto the first.

    Falls back to a fuzzy window anchored on the needle's rarest word when no
    literal run exists (reflowed whitespace, an OCR'd character the text layer
    disagrees on). Returns ``(start, end, ratio, exact)`` or ``None`` — ``exact``
    is True only for a literal substring hit, never for the fuzzy path (a fuzzy
    window can still score ratio 1.0 when the needle matches split across gaps).
    """
    if not needle:
        return None

    hit = stream.find(needle, cursor)
    if hit == -1:
        hit = stream.find(needle)  # wrap: AST order may not follow stream order
    if hit != -1:
        return hit, hit + len(needle), 1.0, True

    # Fuzzy: anchor on the needle's rarest word, then score a window around it.
    # Only words that actually occur are eligible: ranking by raw count would
    # preferentially pick a *zero*-count word (one the text layer spells
    # differently — the very reason the exact match failed) and then bail
    # because the anchor can't be found.
    # `sorted` + a lexical tiebreak keep anchor choice deterministic: iterating a
    # set of strings varies with per-process hash randomization, which would make
    # tied candidates resolve differently from run to run and hand back different
    # gold each time.
    counts = {w: stream.count(w) for w in sorted(set(needle.split())) if len(w) >= 4}
    present = {w: c for w, c in counts.items() if c > 0}
    if not present:
        return None
    anchor = min(present, key=lambda w: (present[w], w))

    best: tuple[int, int, float, bool] | None = None
    span = len(needle)
    search_from = 0
    while True:
        at = stream.find(anchor, search_from)
        if at == -1:
            break
        search_from = at + 1
        # The anchor may sit anywhere inside the needle; scan a padded window.
        lo = max(0, at - span)
        hi = min(len(stream), at + span + len(anchor))
        window = stream[lo:hi]
        sm = SequenceMatcher(None, needle, window, autojunk=False)
        blocks = sm.get_matching_blocks()
        matched = sum(b.size for b in blocks)
        ratio = matched / len(needle) if needle else 0.0
        if ratio >= _MIN_PLACE_RATIO:
            # Tighten to the matched extent inside the window.
            real = [b for b in blocks if b.size]
            if not real:
                continue
            s = lo + real[0].b
            e = lo + real[-1].b + real[-1].size
            if best is None or ratio > best[2]:
                best = (s, e, ratio, False)
    return best


def place_nodes(
    ast_root: dict, stream: str
) -> tuple[dict[str, tuple[int, int]], OracleReport]:
    """Locate each AST text segment's character span in the text-layer stream.

    Segments come from :func:`alignment.naive_aligner._iter_segments` — the same
    set the aligners consume — so gold and prediction share a label space.
    Unplaceable segments are dropped and counted, never guessed.
    """
    report = OracleReport()
    placements: dict[str, tuple[int, int]] = {}
    cursor = 0

    for node_id, text in _iter_segments(ast_root):
        needle = _norm(text)
        if not needle:
            continue
        report.n_nodes += 1
        if len(needle) < _MIN_PLACE_CHARS:
            report.n_unplaced += 1
            report.unplaced_samples.append(f"{node_id}: {needle[:60]!r} (too short)")
            report.unplaced_nodes.append(node_id)
            continue

        found = _locate(stream, needle, cursor)
        if found is None:
            report.n_unplaced += 1
            report.unplaced_samples.append(f"{node_id}: {needle[:60]!r}")
            report.unplaced_nodes.append(node_id)
            continue

        start, end, ratio, exact = found
        placements[node_id] = (start, end)
        report.placement_meta[node_id] = {
            "start": start,
            "end": end,
            "ratio": round(ratio, 4),
            "exact": exact,
        }
        report.n_placed += 1
        cursor = end

    return placements, report


def gold_token_owner(
    tokens: list[Token], placements: dict[str, tuple[int, int]], report: OracleReport
) -> dict[str, str | None]:
    """Compose geometry + text placement into ``token_key -> node_id | None``.

    A token belongs to the node whose placed span overlaps it. Tokens no node
    claims map to ``None`` — the NULL class (page numbers, figure interiors).
    Overlapping placements are resolved first-come and counted as conflicts.
    """
    owner: dict[str, str | None] = {t.key: None for t in tokens}
    claimed: set[str] = set()

    for node_id, (start, end) in placements.items():
        for tok in tokens:
            if tok.stream_start < end and tok.stream_end > start:
                if tok.key in claimed:
                    report.n_conflicts += 1
                    continue
                owner[tok.key] = node_id
                claimed.add(tok.key)

    report.n_tokens = len(tokens)
    report.n_gold_tokens = sum(1 for v in owner.values() if v is not None)
    report.n_null_tokens = sum(1 for v in owner.values() if v is None)
    return owner


def boxes_to_tokens(
    pages: list[dict], tokens: list[Token]
) -> dict[str, list[str]]:
    """Map ``box_key -> [token_key]`` by centre containment.

    Purely geometric: a token belongs to the box containing the most of its
    fragment centres (for an ordinary word that is its single bbox centre; a
    hyphenated word votes with one centre per line half, so it lands in the box
    holding its halves rather than wherever the centre of its two-line union
    bbox happens to fall). Ties — including the nested-boxes case, where a cell
    inside a table region contains the same centres — go to the **smallest**
    box by area, so the most specific region owns the token.
    """
    by_page: dict[int, list[Token]] = {}
    for t in tokens:
        by_page.setdefault(t.page, []).append(t)

    out: dict[str, list[str]] = {}
    for page in pages:
        page_no = page["page"]
        page_tokens = by_page.get(page_no, [])
        boxes = page.get("boxes", [])

        for tok in page_tokens:
            votes: dict[int, int] = {}
            for fx0, fy0, fx1, fy1 in tok.fragments:
                cx, cy = (fx0 + fx1) / 2, (fy0 + fy1) / 2
                for i, b in enumerate(boxes):
                    x0, y0, x1, y1 = b["bbox"]
                    if x0 <= cx <= x1 and y0 <= cy <= y1:
                        votes[i] = votes.get(i, 0) + 1
            best_i, best_key = None, None
            for i, count in votes.items():
                x0, y0, x1, y1 = boxes[i]["bbox"]
                area = max(x1 - x0, 0) * max(y1 - y0, 0)
                key = (-count, area)  # most centres, then smallest box
                if best_key is None or key < best_key:
                    best_i, best_key = i, key
            if best_i is not None:
                out.setdefault(f"{page_no}:{best_i}", []).append(tok.key)

    return out


def derive_gold(
    pdf_path: str | Path, ast_root: dict, pages: list[dict], dpi: int = 200
) -> tuple[list[Token], dict[str, str | None], dict[str, list[str]], OracleReport]:
    """Full derivation: ``(tokens, token_gold, box_tokens, report)``."""
    tokens = extract_tokens(pdf_path, dpi=dpi)
    stream, tokens = build_stream(tokens)
    placements, report = place_nodes(ast_root, stream)
    token_gold = gold_token_owner(tokens, placements, report)
    box_tokens = boxes_to_tokens(pages, tokens)
    return tokens, token_gold, box_tokens, report

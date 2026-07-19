"""Token-level, stratified scoring for the real-document benchmark.

Why tokens rather than boxes
----------------------------
:mod:`benchmark.metrics` scores one decision per *box*, which makes the number
move when the layout engine changes its mind about splitting a paragraph: the
same aligner on the same text scores F1 0.99 at region granularity and 0.40 at
line granularity, because the boxes got smaller, not because it got worse. That
conflates aligner quality with box granularity and makes granularities
incomparable.

Scoring "which node owns this **word token**" removes granularity from the
denominator — a box covering ten tokens contributes ten decisions — so region,
line and word land on one scale. It is also the only unit with a defined answer
when a box straddles two nodes, or when the layout merges a heading into the
paragraph beneath it.

The NULL class
--------------
Gold comes from :mod:`benchmark.oracle`, where a token no AST node claims is
``None``. That is a real label, not a gap: page numbers, running heads and text
inside figures *should* map to nothing, and an aligner that invents an owner for
them is committing a **false attribution** — the error that matters most when
alignment is used for provenance. :attr:`Scores.false_attribution_rate` reports
it directly.

What is deliberately not scored
-------------------------------
Gold-NULL means "no AST node was placed here", which is not always the same as
"no node owns this". For tables and display formulas the AST *does* carry a node
— Mistral emits a markdown table or a LaTeX run — but it has no literal glyph
sequence for the oracle to place, so gold records NULL. Grading those would
penalise an aligner for being right. Tokens in those regions are **excluded**
and counted (:attr:`Scores.n_excluded`) rather than quietly scored, so the
blind spot is a reported number instead of a free win.

The same failure mode exists outside those labels: whenever the oracle failed
to place a node (LaTeX inline math, image refs, sub-minimum text, fuzzy
misses), that node's tokens fall out as gold-NULL, and an aligner that
*correctly* matches them would be charged a false attribution. When the caller
supplies the set of placed node ids, a gold-NULL token whose *predicted* node
is unplaced is likewise excluded and counted
(:attr:`Scores.n_excluded_unverifiable`) — the oracle has no standing to grade
a claim about a node it could not itself locate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Regions whose AST counterpart exists but is non-literal (markdown table, LaTeX
# formula), so the oracle cannot place it. Gold-NULL here is an oracle limit,
# not ground truth — exclude from scoring and report the count.
ORACLE_BLIND_LABELS = frozenset({"table", "display_formula", "formula_number"})


@dataclass
class Scores:
    """Token-level scores for one method, over one stratum."""

    n_tokens: int = 0
    n_excluded: int = 0  # total excluded (blind labels + unverifiable)
    # Subset of n_excluded: gold-NULL tokens whose predicted node the oracle
    # failed to place, so the "false attribution" verdict is unverifiable.
    n_excluded_unverifiable: int = 0
    n_gold_owned: int = 0  # gold is a node
    n_gold_null: int = 0  # gold is None (true null)
    n_pred_owned: int = 0

    tp: int = 0  # gold node, predicted correctly
    fp: int = 0  # predicted a node that is not gold (incl. gold-null)
    fn: int = 0  # gold node, predicted wrong or abstained
    tn: int = 0  # gold null, correctly abstained
    false_attrib: int = 0  # gold null, but a node was predicted
    tp_ambiguous: int = 0  # wrong node, but its text is identical to gold's

    # Undefined metrics return None rather than 0.0. A stratum of pure page
    # furniture (every token gold-NULL) has no recall to measure, and printing
    # 0.000 there reads as "failed completely" when the correct reading is "not
    # applicable — score this stratum by abstention instead".

    @property
    def precision(self) -> float | None:
        d = self.tp + self.fp
        return self.tp / d if d else None

    @property
    def recall(self) -> float | None:
        d = self.tp + self.fn
        return self.tp / d if d else None

    @property
    def f1(self) -> float | None:
        r = self.recall
        if r is None:
            return None  # no gold owner exists here — nothing to retrieve
        p = self.precision
        if p is None:
            return 0.0  # gold existed and we predicted nothing: a miss, not N/A
        if (p + r) == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def accuracy(self) -> float:
        """Fraction of scored tokens labelled correctly, NULL included."""
        d = self.n_tokens - self.n_excluded
        return (self.tp + self.tn) / d if d else 0.0

    @property
    def accuracy_up_to_ambiguity(self) -> float:
        """Accuracy crediting a match to any node whose text is identical.

        Repeated text is not identifiable from text alone, so the gap between
        this and :attr:`accuracy` is the ambiguity tax rather than a defect.
        """
        d = self.n_tokens - self.n_excluded
        return (self.tp + self.tp_ambiguous + self.tn) / d if d else 0.0

    @property
    def null_accuracy(self) -> float | None:
        """Of tokens that truly own no node, how many did we correctly abstain on."""
        return self.tn / self.n_gold_null if self.n_gold_null else None

    @property
    def false_attribution_rate(self) -> float | None:
        """Of tokens that truly own no node, how many were given one anyway.

        The provenance-critical error: the system pointing confidently at a
        source region for something no source supports.
        """
        return self.false_attrib / self.n_gold_null if self.n_gold_null else None

    def to_dict(self) -> dict:
        def r4(v: float | None) -> float | None:
            return round(v, 4) if v is not None else None

        return {
            "n_tokens": self.n_tokens,
            "n_excluded": self.n_excluded,
            "n_excluded_unverifiable": self.n_excluded_unverifiable,
            "n_gold_owned": self.n_gold_owned,
            "n_gold_null": self.n_gold_null,
            "precision": r4(self.precision),
            "recall": r4(self.recall),
            "f1": r4(self.f1),
            "accuracy": r4(self.accuracy),
            "accuracy_up_to_ambiguity": r4(self.accuracy_up_to_ambiguity),
            "null_accuracy": r4(self.null_accuracy),
            "false_attribution_rate": r4(self.false_attribution_rate),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "false_attrib": self.false_attrib,
        }


def merge_scores(parts: list[Scores]) -> Scores:
    """Pool raw counts across documents — a **micro** average.

    Summing the confusion counts weights every *token* equally, so long
    documents pull harder. That is the right default for "how does this behave
    on a page", but it hides per-document variance, which is why
    :func:`macro_f1` is reported next to it.
    """
    out = Scores()
    for p in parts:
        out.n_tokens += p.n_tokens
        out.n_excluded += p.n_excluded
        out.n_excluded_unverifiable += p.n_excluded_unverifiable
        out.n_gold_owned += p.n_gold_owned
        out.n_gold_null += p.n_gold_null
        out.n_pred_owned += p.n_pred_owned
        out.tp += p.tp
        out.fp += p.fp
        out.fn += p.fn
        out.tn += p.tn
        out.false_attrib += p.false_attrib
        out.tp_ambiguous += p.tp_ambiguous
    return out


def macro_f1(parts: list[Scores]) -> float | None:
    """Mean of per-document F1 — every **document** weighted equally.

    Diverges from the micro F1 when documents differ in size or difficulty; the
    gap between the two is itself the signal that per-doc variance matters.
    """
    vals = [p.f1 for p in parts if p.f1 is not None]
    return sum(vals) / len(vals) if vals else None


@dataclass
class StratifiedScores:
    overall: Scores = field(default_factory=Scores)
    by_label: dict[str, Scores] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "overall": self.overall.to_dict(),
            "by_label": {k: v.to_dict() for k, v in sorted(self.by_label.items())},
        }


def predict_token_owner(
    reverse: dict[str, list[str]], box_tokens: dict[str, list[str]]
) -> dict[str, str | None]:
    """Turn an aligner's ``box -> [node]`` map into ``token -> node | None``.

    Every token inside a box inherits that box's **primary** (highest-scoring)
    predicted owner. Tokens in a box the aligner made no claim about — and
    tokens in no box at all — are predicted NULL, i.e. an abstention.
    """
    pred: dict[str, str | None] = {}
    for box_key, token_keys in box_tokens.items():
        owners = reverse.get(box_key) or []
        owner = owners[0] if owners else None
        for tk in token_keys:
            pred[tk] = owner
    return pred


def score_tokens(
    token_gold: dict[str, str | None],
    token_pred: dict[str, str | None],
    token_label: dict[str, str],
    node_text: dict[str, str],
    placed_nodes: set[str] | None = None,
) -> StratifiedScores:
    """Score ``token -> node`` predictions against oracle gold, per stratum.

    ``token_label`` maps a token to the layout label of its containing box (the
    stratum). ``node_text`` maps node id to normalized text, used only to decide
    whether a wrong answer is wrong-but-unidentifiable.

    ``placed_nodes`` is the set of node ids the oracle actually placed (from
    :attr:`benchmark.oracle.OracleReport.placed_node_ids`). When given, a
    gold-NULL token whose *predicted* node is not in the set is excluded rather
    than scored false_attrib/fp: that gold-NULL is an oracle placement failure,
    not evidence the token owns nothing, so the prediction is unverifiable —
    possibly exactly right. When ``None`` (legacy callers), the old behaviour
    is kept and such tokens score as false attributions.

    Known asymmetry: contamination is only detectable when the aligner *points
    at* an unplaced node. A gold-NULL token whose true owner is unplaced but
    where the aligner abstained still scores tn — we cannot tell "correctly
    ignored page furniture" from "missed a node the oracle also missed" without
    ground truth the oracle can't supply. That residual bias (null_accuracy and
    tn are flattered by an unknown amount) is accepted until human annotation
    of the unplaced nodes resolves it.
    """
    out = StratifiedScores()

    for tk, gold in token_gold.items():
        label = token_label.get(tk, "none")
        pred = token_pred.get(tk)
        buckets = [out.overall, out.by_label.setdefault(label, Scores())]

        excluded = gold is None and label in ORACLE_BLIND_LABELS
        # Oracle-contamination guard (see docstring): gold-NULL + prediction of
        # a node the oracle failed to place cannot be adjudicated.
        unverifiable = (
            not excluded
            and gold is None
            and pred is not None
            and placed_nodes is not None
            and pred not in placed_nodes
        )
        for s in buckets:
            s.n_tokens += 1
            if excluded or unverifiable:
                s.n_excluded += 1
            if unverifiable:
                s.n_excluded_unverifiable += 1
        if excluded or unverifiable:
            continue

        for s in buckets:
            if pred is not None:
                s.n_pred_owned += 1

            if gold is None:
                s.n_gold_null += 1
                if pred is None:
                    s.tn += 1
                else:
                    s.false_attrib += 1
                    s.fp += 1
                continue

            s.n_gold_owned += 1
            if pred == gold:
                s.tp += 1
            else:
                s.fn += 1
                if pred is not None:
                    s.fp += 1
                    # Wrong node, but identical text: not identifiable from text.
                    if (
                        pred in node_text
                        and gold in node_text
                        and node_text[pred] == node_text[gold]
                    ):
                        s.tp_ambiguous += 1

    return out

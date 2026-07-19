"""benchmark_v2-0: node-level localization benchmark against human-verified gold.

The v1 benchmark (:mod:`benchmark`) scores token-level attribution against gold
derived automatically from the PDF text layer. It is statistically dense but
illegible, and its answer key is an inference with measured failure modes
(~24% of nodes unplaceable; false-attribution contamination).

v2 flips the hierarchy. The unit is the **AST node** — the thing a user
actually clicks — and the question is the one the product asks: *does the
aligner light up the right region of the page for this node?* Gold is a plain
asserted relation, ``node -> {layout boxes} | ABSENT``, sourced from the
annotation tool (``/annotate``) with the v1 oracle demoted to what it honestly
is: a proposal generator whose guesses humans confirm.

Two headline metrics, each one sentence:

- **localization accuracy** — fraction of nodes whose predicted region matches
  gold (pixel-area IoU of box unions >= 0.5, so a paragraph the layout engine
  split into two boxes still scores as found);
- **false claim rate** — fraction of furniture boxes (owned by nobody) that
  some node wrongly claimed.

v1 remains untouched alongside as the fine-grained diagnostic tier.
"""

from .gold import build_gold, GoldDoc
from .score import score_doc, ScoreCard

__all__ = ["build_gold", "GoldDoc", "score_doc", "ScoreCard"]

# Referee Report

**Manuscript:** Aligning Logical Document Structure with Physical Page Layout
**Journal:** Informatica (Slovenia) — Research/Regular article
**Material reviewed:** title block, abstract, Introduction (5 paragraphs), Figure 1, references
**Recommendation:** **Major revision.** As submitted the manuscript would not survive
editorial screening (4 pages against a stated 8–20 page range, and only the introduction
is present), so this report assesses the framing, the claims made in it, and compliance.

---

## 1. Summary of the submission

The paper argues that recovering document organisation from PDF has split into two
lines of work — structure-oriented OCR, which yields a hierarchy without coordinates,
and layout detection, which yields coordinates without a hierarchy — and that linking
the two should be treated as a matching problem between independently produced views
rather than as a byproduct of a single model. The authors build both views from the
same PDF, recover reading order over detected regions by recursive XY cutting, and
compare two aligners resting on opposite assumptions about order (a positional
character-stream aligner and an order-free embedding-similarity aligner), both able to
abstain. They evaluate at word level against an oracle over the PDF text layer and at
node level against human annotation with recorded provenance. Two findings are
claimed: automatically produced ground truth can fabricate the errors it is meant to
detect, and the two aligners fail on complementary material.

## 2. Assessment

**Originality and significance.** The framing is genuine and the paper is well
positioned to make a contribution. The insistence that "no match" is a valid outcome,
and the tiered-provenance ground truth, are the kind of methodological care this area
often lacks. The contamination finding is the most interesting thing here and is, in
my view, publishable on its own terms. However, novelty **cannot currently be assessed**
(see Major 1).

**Soundness.** The reasoning is careful and the paper is admirably candid about its
limits. Two specific claims are, however, stated more strongly than the evidence
supports (Major 3 and 4).

**Presentation.** Well above average. The prose is clear, the argument builds in a
sensible order, and Figure 1 is genuinely informative and does real work. Editorial
issues are minor and listed in §5.

---

## 3. Major points

**M1. The paper has one reference. This is disqualifying in its present state.**
The central premise — that the literature has split into two lines of work — is
asserted across the whole first paragraph without a single citation. A Related Work
section is required, and it must cover at minimum: LLM/VLM-based document conversion;
layout detection and its datasets; reading-order recovery; and, most importantly,
**prior work that already aligns a logical representation to page regions.** PubLayNet
and DocBank both construct their ground truth by aligning a structured source (XML,
LaTeX) to PDF tokens or regions, which is close enough to this paper's problem that
the manuscript must state explicitly what is different. My reading is that the
distinction is real and defensible — those works align a *known* source to its own
PDF, whereas here two *independent* recognisers must be reconciled with no shared
provenance — but the paper does not currently make that argument, and a reader who
knows those datasets will assume the problem is solved. This is the single most
important revision.

**M2. Misattributed citation for XY cutting.** Line 69 credits "recursive XY cutting"
to reference [1] (XY-Cut++, 2025). Recursive XY cut is due to Nagy and Seth (1984);
reference [1] is a recent refinement. Cite the original alongside it, and state
precisely which variant is implemented. As written this reads as crediting a 2025
paper with a 1984 algorithm.

**M3. "Fell to less than half" is true of one aligner, not both.** The abstract and
Introduction both state without qualification that separating out the unplaceable
nodes "cut the measured misattribution rate to less than half." From the authors' own
results this holds for the positional aligner (0.634 → 0.300, a 53% reduction) but not
for the similarity aligner (0.655 → 0.440, a 33% reduction). Qualify the claim or
report both figures. Reviewers will check this.

**M4. The headline number is anecdotal as presented.** "On one paper, 458 of the 477
misattributions..." does not say which paper, which aligner, or whether the document is
representative. A single document cannot carry a claim that the abstract elevates to a
general methodological lesson. Report the corpus-wide contamination rate with the
per-document distribution, and keep the single-document figure only as an illustration.

**M5. Results are claimed in an introduction that has no results section behind them.**
The complementary-regimes finding, and specifically "most visibly in tables," needs its
sample size stated. If it rests on a small number of table nodes in a ten-document
corpus, it cannot support an unhedged claim in the abstract. Either report n and a
measure of variability, or soften to a hypothesis the paper investigates.

**M6. Internal tension between the contributions and the stated limitation.**
The fourth contribution claims two findings that "generalise beyond this system,"
and the very next sentence concedes the corpus does not support general estimates.
Both can be true — a methodological warning generalises where an accuracy number does
not — but the paper must say so explicitly rather than leaving the contradiction
sitting in one paragraph.

**M7. Missing specification and reproducibility.** The similarity threshold τ governs
all abstention behaviour and is never given, nor is any sensitivity analysis offered.
The corpus is characterised only as "born-digital research papers" (how many? which?).
No model identities or versions, and no code or data availability statement. For a
paper whose contribution is partly a benchmark, releasing the annotations and the
annotation tool is close to mandatory.

**M8. Were other formulations considered?** The paper presents exactly two aligners as
though they exhaust the design space. A reader will immediately ask about global
assignment formulations (optimal bipartite matching / ILP), which are the obvious third
option for a problem framed as matching. If such variants were tried, report them; if
they were excluded, say why.

## 4. Minor points

- **M9.** "Oracle" is used in the abstract before it is defined, where it will read as
  a claim of ground-truth authority rather than a heuristic over the PDF text layer.
- **M10.** The abstract runs to roughly 330 words. It is ten sentences as the template
  suggests, but they are long ones; ~250 words is the usual ceiling.
- **M11.** "OCR models built on large language models" is imprecise — these are
  vision-language models. Consider "multimodal models".
- **M12.** Introduction paragraph 4 is ~340 words and carries three distinct arguments
  (protocol, contamination, findings). Split it.
- **M13.** Terminology drifts between "regions" (body text) and "boxes" (figures and
  parts of the text). Fix on one and use it everywhere, including figure labels.
- **M14.** No roadmap sentence at the end of the introduction.
- **M15.** Figure 1 is referenced on page 1 but placed on page 2. Acceptable, but
  consider whether it can be pulled forward.

## 5. Editorial and language

- L71: "Measuring how well **either** aligner works turned out to be as difficult as
  building **them**" — number disagreement. Use "both aligners ... them".
- L71: "458 of the 477 misattributions the benchmark reported pointed at a node" is a
  garden-path sentence. Recast: "of the 477 misattributions the benchmark reported, 458
  pointed at a node..."
- L71: "and we think this is worth stating plainly because the contaminated numbers
  looked entirely reasonable" — register is too conversational for the venue. Recast as
  a statement about why the failure is hard to detect.
- L69: "a sequence that can be compared with the AST **at all**" — colloquial; delete
  "at all" or rephrase.
- L21–22: "agree on nothing exactly, neither their tokenisation, nor..." — the adverb
  is misplaced and the correlative list does not agree with "agree on". Recast as
  "share neither a tokenisation, nor a treatment of mathematics, nor an emission order".
- L73: "implements that **view** end to end" — the antecedent is "matching problem";
  "framing" would be correct.
- **Spelling consistency:** the manuscript uses British -ise forms throughout
  (organised, recognised, tokenisation, penalised, generalise) but "locali**z**ation"
  at L73. Choose one convention.
- Tense drifts between the descriptive present ("We describe a system") and the
  narrative past ("Building this benchmark exposed"). Acceptable, but be deliberate.

## 6. Compliance with journal requirements

| Requirement | Status |
|---|---|
| 8–20 pages including references | **Not met** — currently 4 |
| Slovenian abstract (`\abstractSi`, *Povzetek*) | **Empty** — required by template |
| DOIs in references | **Not met** — add `10.48550/arXiv.2504.10258` |
| Complete author metadata | **Not met** — institution and received date are TODO |
| Figures legible in black and white | **Met** — verified via the greyscale build |
| Template followed (title, headings, captions) | Met |
| Cover letter stating novelty and fit | Not seen; required at submission |

## 7. Summary for the editor

The framing is strong and the methodological contribution around contaminated ground
truth is worth publishing. The manuscript is not yet assessable as a research article:
it lacks a literature review entirely, which makes its novelty claim unverifiable, and
two quantitative claims are stated more broadly than the authors' own data supports.
I would be glad to review a revised and completed version.

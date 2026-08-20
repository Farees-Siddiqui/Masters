"""Key canonicalization by *induction*, not by hand.

The extractor is open-schema on purpose: for a document type nobody has seen
before, the keys have to come from the document. But left alone the model
invents synonyms for keys it already has — a stage-1 run over one paper
produced 44 distinct keys across three passes, of which only 15 appeared in all
three. `layers`, `layer`, `number_of_layers` and `depth` are one key.
`journal` and `publication` are one key. That drift, not the extraction itself,
is what made record counts swing between runs.

The fix is to *learn* the vocabulary from what the model actually emits:

    (key, values) pairs  ->  embed  ->  cluster  ->  one label per cluster

No key is written down by a human, so an invoice corpus induces invoice keys
and a clinical corpus induces clinical keys.

**Why keys are embedded with their values.** Clustering key *names* alone does
not work, and the failure is not a tuning problem. Measured over three runs on
one paper, no threshold merges `depth` with `layers` before it also merges
`author` with `journal` and `result` with `task` — the names simply are not
similar enough in the right places. Clustering by value distribution alone
over-merges instead, collapsing `dataset`, `organization` and `venue` because
they all hold short proper nouns. Concatenating the two — the key name's
embedding beside the centroid of its values' embeddings — is what separates
them: `depth` and `layers` both hold bare integers, while `author` and
`journal` hold completely different value distributions. At threshold 0.45 that
representation produced only correct merges on the sample corpus.

Two properties keep it honest for unseen data:

- A key that is not close to any existing cluster is **kept as itself**, not
  snapped to the nearest label. New document types grow the vocabulary instead
  of being force-fit into an old one.
- `raw_key` is preserved on every record, so canonicalization is reversible.

Embeddings come from MiniLM via `transformers`. If the model can't be fetched
(offline box), we fall back to character-n-gram TF-IDF and warn — that fallback
is lexical only and will miss `depth`/`layers`.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.45  # cosine distance in the joint name+value space

_MODEL_CACHE: dict = {}


def _phrase(key: str) -> str:
    """`number_of_layers` -> `number of layers`, so the encoder sees words."""
    return key.replace("_", " ").replace("-", " ").strip().lower()


def _embed_minilm(texts: list[str]) -> np.ndarray | None:
    """Mean-pooled MiniLM embeddings, L2-normalized. None if unavailable."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception:
        return None
    try:
        if EMBED_MODEL not in _MODEL_CACHE:
            tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
            mod = AutoModel.from_pretrained(EMBED_MODEL).eval()
            _MODEL_CACHE[EMBED_MODEL] = (tok, mod)
        tok, mod = _MODEL_CACHE[EMBED_MODEL]
        out_chunks = []
        with torch.no_grad():
            for i in range(0, len(texts), 256):
                batch = tok(texts[i : i + 256], padding=True, truncation=True,
                            max_length=64, return_tensors="pt")
                hidden = mod(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).float()
                vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                out_chunks.append(torch.nn.functional.normalize(vec, p=2, dim=1))
            return torch.cat(out_chunks).cpu().numpy()
    except Exception as exc:  # network down, model missing, etc.
        print(f"[vocab] MiniLM unavailable ({exc}); using TF-IDF fallback", file=sys.stderr)
        return None


def _embed_tfidf(texts: list[str]) -> np.ndarray:
    """Character-n-gram fallback. Lexical only — merges morphology, not meaning."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    mat = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(texts).toarray()
    return mat / np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9, None)


def embed(texts: list[str]) -> tuple[np.ndarray, str]:
    """Embed arbitrary short strings. Returns (matrix, backend_name)."""
    vecs = _embed_minilm(texts)
    if vecs is not None:
        return vecs, "minilm"
    return _embed_tfidf(texts), "tfidf"


def _unit(mat: np.ndarray) -> np.ndarray:
    return mat / np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9, None)


def joint_vectors(
    key_values: dict[str, list[str]], keys: list[str]
) -> tuple[np.ndarray, str]:
    """Represent each key as [name embedding | centroid of its value embeddings].

    This is the representation the module docstring argues for: the name half
    keeps `dataset` away from `organization`, the value half pulls `depth`
    toward `layers`. Keys with no values fall back to duplicating the name half
    so the vector stays the right shape and the key can still be placed.
    """
    values = sorted({v for k in keys for v in key_values.get(k, [])})
    name_vecs, backend = embed([_phrase(k) for k in keys])
    if not values:
        return _unit(np.hstack([name_vecs, name_vecs])), backend

    val_vecs, _ = embed(values)
    index = {v: i for i, v in enumerate(values)}
    centroids = []
    for i, k in enumerate(keys):
        vs = key_values.get(k, [])
        if vs:
            centroids.append(val_vecs[[index[v] for v in vs]].mean(axis=0))
        else:
            centroids.append(name_vecs[i])
    cent = _unit(np.vstack(centroids))
    return _unit(np.hstack([name_vecs, cent])), backend


@dataclass
class Vocabulary:
    """A learned key vocabulary: cluster labels plus their joint-space centroids."""

    labels: list[str] = field(default_factory=list)
    centroids: np.ndarray | None = None
    members: dict[str, list[str]] = field(default_factory=dict)
    threshold: float = DEFAULT_THRESHOLD
    backend: str = "minilm"

    @classmethod
    def induce(
        cls,
        key_values: dict[str, list[str]],
        key_counts: Counter | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> "Vocabulary":
        """Cluster emitted keys; the most frequent member names each cluster.

        Frequency picks the label because the common surface form is the one
        the model reaches for by default — `venue` over `competition`, `depth`
        over `number_of_layers`. Nobody chooses it.
        """
        counts = key_counts or Counter({k: len(v) for k, v in key_values.items()})
        keys = [k for k, _ in counts.most_common()]
        if not keys:
            return cls(threshold=threshold)

        vecs, backend = joint_vectors(key_values, keys)

        if len(keys) == 1:
            return cls(labels=list(keys), centroids=vecs, members={keys[0]: [keys[0]]},
                       threshold=threshold, backend=backend)

        from sklearn.cluster import AgglomerativeClustering

        assign = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="cosine",
            linkage="average",
        ).fit_predict(vecs)

        groups: dict[int, list[str]] = defaultdict(list)
        for key, cid in zip(keys, assign):
            groups[int(cid)].append(key)

        labels, centroids, named = [], [], {}
        for _, group in sorted(groups.items()):
            label = max(group, key=lambda k: (counts[k], -len(k)))
            idx = [keys.index(k) for k in group]
            labels.append(label)
            centroids.append(vecs[idx].mean(axis=0))
            named[label] = sorted(group)

        return cls(labels=labels, centroids=_unit(np.vstack(centroids)),
                   members=named, threshold=threshold, backend=backend)

    def assign(self, key_values: dict[str, list[str]]) -> dict[str, str]:
        """Map each raw key onto the vocabulary, or keep it if genuinely new.

        Batch by design: canonicalization always has the document's values to
        hand, so it can use the same joint representation induction used.
        """
        if self.centroids is None or not self.labels:
            return {k: k for k in key_values}

        out: dict[str, str] = {}
        pending = []
        for raw in key_values:
            key = raw.strip().lower().replace(" ", "_").replace("-", "_")
            hit = next((l for l, g in self.members.items() if key == l or key in g), None)
            if hit:
                out[raw] = hit
            else:
                pending.append((raw, key))

        if pending:
            sub = {k: key_values[raw] for raw, k in pending}
            keys = [k for _, k in pending]
            vecs, _ = joint_vectors(sub, keys)
            sims = vecs @ self.centroids.T
            for (raw, key), row in zip(pending, sims):
                best = int(np.argmax(row))
                # The `else key` branch is the point: a distant key is never
                # snapped onto the nearest label just because it is nearest.
                out[raw] = self.labels[best] if (1.0 - float(row[best])) <= self.threshold else key
        return out

    def canonical(self, raw_key: str, values: list[str] | None = None) -> str:
        return self.assign({raw_key: values or []})[raw_key]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "labels": self.labels,
            "centroids": self.centroids.tolist() if self.centroids is not None else [],
            "members": self.members,
            "threshold": self.threshold,
            "backend": self.backend,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        d = json.loads(path.read_text(encoding="utf-8"))
        cents = np.array(d["centroids"], dtype=float) if d["centroids"] else None
        return cls(labels=d["labels"], centroids=cents, members=d.get("members", {}),
                   threshold=d.get("threshold", DEFAULT_THRESHOLD),
                   backend=d.get("backend", "minilm"))


def collect(paths: list[Path]) -> tuple[dict[str, list[str]], Counter]:
    """Gather key -> values and key counts from previous run outputs."""
    key_values: dict[str, list[str]] = defaultdict(list)
    counts: Counter = Counter()
    for p in paths:
        for rec in json.loads(p.read_text(encoding="utf-8"))["records"]:
            key = (rec.get("raw_key") or rec["key"]).strip().lower()
            key_values[key].append(rec["value"])
            counts[key] += 1
    return dict(key_values), counts


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Induce a key vocabulary from run outputs.")
    ap.add_argument("runs", nargs="+", type=Path, help="records*.json from previous runs")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--out", type=Path, default=Path("vocab.json"))
    args = ap.parse_args()

    key_values, counts = collect(args.runs)
    v = Vocabulary.induce(key_values, counts, threshold=args.threshold)
    v.save(args.out)

    merged = {l: g for l, g in v.members.items() if len(g) > 1}
    print(f"Induced {len(v.labels)} keys from {len(key_values)} distinct "
          f"({v.backend}, t={args.threshold}) -> {args.out}")
    print(f"{len(merged)} clusters merged:\n")
    for label, group in sorted(merged.items(), key=lambda x: -len(x[1])):
        print(f"  {label:22s} <- {', '.join(g for g in group if g != label)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

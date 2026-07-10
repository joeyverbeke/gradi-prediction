"""Embedding-prototype scenario scorer (Change 3).

Scores the *meaning* of a short string against per-scenario exemplar sentences in
embedding space, so DAF can fire on paraphrase the keyword stems miss. Pure CPU
via fastembed (all-MiniLM-L6-v2 int8 ONNX), no VRAM, no external services.

Score (margin form): ``score(text) = max_cos(text, exemplars) - max_cos(text, contrast)``.
Exemplars span what a participant might actually say on-topic (for ``ai_future``,
both valences); contrast covers hedging/meta filler. The margin fires on any
committed statement and stays quiet on stalling.

The class is pure/stateless after construction (no globals) so it can run inside a
dedicated worker thread. Embeddings are L2-normalized, so cosine == dot product.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ScenarioPrototypes:
    """Precomputed exemplar/contrast embedding matrices for one scenario."""

    scenario_id: str
    exemplars: np.ndarray  # shape (n_exemplars, dim), L2-normalized
    contrast: np.ndarray  # shape (n_contrast, dim), L2-normalized (may be empty)


class SemanticDetector:
    """Cosine-margin scenario scorer over a small sentence-embedding model."""

    def __init__(
        self,
        scenarios: Sequence[dict],
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
    ) -> None:
        """Build the detector.

        Args:
            scenarios: iterable of dicts, each ``{"id": str, "exemplars": [...],
                "contrast": [...]}``. Scenarios with no exemplars are skipped
                (they can never produce a semantic hit).
            model_name: fastembed model id.
            cache_dir: where fastembed caches the model. Point this at a
                co-located, persistent dir (``models/embed``) so the installation
                loads offline from cache.
        """
        # Imported lazily so a bad/missing fastembed install only breaks semantic
        # scoring (caller degrades to stems), never the whole app at import time.
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self._model_name = model_name

        self._prototypes: Dict[str, ScenarioPrototypes] = {}
        for scenario in scenarios:
            sid = str(scenario.get("id", "")).strip()
            if not sid:
                continue
            exemplars = [str(x).strip() for x in scenario.get("exemplars", []) if str(x).strip()]
            contrast = [str(x).strip() for x in scenario.get("contrast", []) if str(x).strip()]
            if not exemplars:
                # No exemplars -> nothing to score against; skip silently.
                continue
            self._prototypes[sid] = ScenarioPrototypes(
                scenario_id=sid,
                exemplars=self._embed(exemplars),
                contrast=self._embed(contrast) if contrast else np.empty((0, self._dim), dtype=np.float32),
            )

    @property
    def _dim(self) -> int:
        return getattr(self, "_dim_cache", 384)

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of strings into an (n, dim) L2-normalized float32 matrix."""
        vecs = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        # fastembed already normalizes these models, but enforce it so cosine==dot.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        self._dim_cache = vecs.shape[1]
        return vecs

    @property
    def scenario_ids(self) -> List[str]:
        return list(self._prototypes.keys())

    def score(self, text: str) -> Dict[str, float]:
        """Return the cosine-margin score of ``text`` for each known scenario.

        Empty/whitespace text yields an empty dict (no scores). Scenarios without a
        contrast set fall back to plain ``max_cos``.
        """
        text = (text or "").strip()
        if not text or not self._prototypes:
            return {}

        vec = self._embed([text])[0]  # (dim,), normalized
        scores: Dict[str, float] = {}
        for sid, proto in self._prototypes.items():
            pos = float(np.max(proto.exemplars @ vec)) if proto.exemplars.size else 0.0
            neg = float(np.max(proto.contrast @ vec)) if proto.contrast.size else 0.0
            scores[sid] = pos - neg
        return scores

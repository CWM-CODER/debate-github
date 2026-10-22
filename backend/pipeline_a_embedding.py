"""
Pipeline A — Semantic Embedding Model (PDF Section 7)
Uses sentence-transformers for meaning-aware retrieval.
IMPORTANT (PDF): Cosine similarity is for retrieval/ranking ONLY.
It is NOT used to infer contradiction, agreement, or debate quality.
That is the job of Pipeline D (NLI cross-encoder).
"""
from __future__ import annotations
import logging
import numpy as np
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Model loaded ONCE at startup — never reloaded per match (PDF Section 14 + 22)
# all-mpnet-base-v2 chosen per PDF Section 18 recommendation
_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
_model: Optional[SentenceTransformer] = None


def load_model() -> None:
    """Load embedding model once at application startup."""
    global _model
    if _model is None:
        logger.info(f"[Pipeline A] Loading embedding model: {_MODEL_NAME}")
        _model = SentenceTransformer(_MODEL_NAME)
        logger.info("[Pipeline A] Model loaded.")


def get_model() -> SentenceTransformer:
    if _model is None:
        load_model()
    return _model


def encode(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Encode a list of texts into L2-normalized embedding vectors.
    Returns shape (N, D).
    """
    if not texts:
        return np.array([])
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,  # unit vectors → cosine similarity = dot product
        show_progress_bar=False
    )
    return embeddings


def encode_single(text: str) -> np.ndarray:
    """Encode a single text. Returns shape (D,)."""
    return encode([text])[0]


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarities between rows of a (shape N×D) and b (shape M×D).
    Returns shape (N, M).
    Because embeddings are L2-normalized, this is just a dot product.
    """
    return np.dot(a, b.T)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two single vectors."""
    return float(np.dot(a, b))


def compute_topic_relevance(claim_embedding: np.ndarray, topic_embedding: np.ndarray) -> float:
    """
    Topic relevance signal for scoring (PDF Section 11 — 15% weight).
    Returns 0-1 similarity of claim to topic.
    """
    return max(0.0, cosine_similarity(claim_embedding, topic_embedding))


def detect_repetition_type(
    new_embedding: np.ndarray,
    own_history_embeddings: List[np.ndarray],
    HIGH_THRESHOLD: float = 0.90,
    MED_THRESHOLD: float = 0.75
) -> Tuple[str, float]:
    """
    Classify new claim against participant's own claim history.
    Returns (repetition_class, max_similarity_score).

    PDF Section 7.4:
    - repeated: same idea, no new evidence → penalizable
    - rephrased: same idea with different vocab (not penalized if no evidence check)
    - elaborated: extends a prior claim → NOT penalized
    - new: genuinely new claim

    NOTE: This classification is semantic only.
    The scoring engine decides the actual penalty (PDF Section 12, Rule 4):
    repetition WITH new evidence is NOT penalized.
    """
    from models import RepetitionClass

    if not own_history_embeddings:
        return RepetitionClass.NEW, 0.0

    sims = [cosine_similarity(new_embedding, h) for h in own_history_embeddings]
    max_sim = max(sims)

    if max_sim >= HIGH_THRESHOLD:
        return RepetitionClass.REPEATED, max_sim
    elif max_sim >= MED_THRESHOLD:
        return RepetitionClass.REPHRASED, max_sim
    else:
        return RepetitionClass.NEW, max_sim


def retrieve_top_k_candidates(
    query_embedding: np.ndarray,
    candidate_embeddings: List[np.ndarray],
    candidate_ids: List[str],
    k: int = 5,
    unresolved_boost: Optional[List[bool]] = None
) -> List[Tuple[str, float]]:
    """
    Retrieve top-k most semantically relevant candidates from a list.
    Used by Pipeline C to get candidate opponent claims for NLI analysis.

    unresolved_boost: parallel list; if True, boost that candidate's score
    (PDF Section 8.2: unresolved claims weighted higher).

    Returns list of (candidate_id, similarity_score) sorted descending.
    """
    if not candidate_embeddings or not candidate_ids:
        return []

    sims = [cosine_similarity(query_embedding, c) for c in candidate_embeddings]

    # Boost unanswered/partially-answered claims (PDF Section 8.2)
    UNRESOLVED_BOOST = 0.08
    if unresolved_boost and len(unresolved_boost) == len(sims):
        sims = [
            s + (UNRESOLVED_BOOST if boost else 0.0)
            for s, boost in zip(sims, unresolved_boost)
        ]

    ranked = sorted(
        zip(candidate_ids, sims),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked[:k]

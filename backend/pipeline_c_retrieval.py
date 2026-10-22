"""
Pipeline C — Relevant Point Retrieval (PDF Section 8)
Retrieves the opponent's most relevant previous claims for the current response.
Tracks unanswered/partially_addressed/resolved status across all turns.

PDF Section 8: "The system should not assume that the current participant
is responding to the immediately previous sentence."

Uses two-stage retrieval:
1. Fast: Embedding similarity (Pipeline A) for candidate retrieval
2. Precise: Cross-encoder reranking of top-k candidates
"""
from __future__ import annotations
import logging
from typing import List, Tuple, Optional
import numpy as np
from models import ClaimRecord, ClaimStatus, ArgumentRelation, RelationRecord

logger = logging.getLogger(__name__)

# Cross-encoder loaded once (PDF Section 14 + 22)
_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder = None


def load_cross_encoder() -> None:
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            logger.info(f"[Pipeline C] Loading cross-encoder: {_CROSS_ENCODER_MODEL}")
            _cross_encoder = CrossEncoder(_CROSS_ENCODER_MODEL)
            logger.info("[Pipeline C] Cross-encoder loaded.")
        except Exception as e:
            logger.error(f"[Pipeline C] Failed to load cross-encoder: {e}")
            _cross_encoder = None


def get_cross_encoder():
    if _cross_encoder is None:
        load_cross_encoder()
    return _cross_encoder


def retrieve_relevant_opponent_claims(
    current_text: str,
    current_claim_embedding: np.ndarray,
    opponent_claims: List[ClaimRecord],
    opponent_embeddings: List[np.ndarray],
    top_k: int = 5,
    final_k: int = 3
) -> List[Tuple[ClaimRecord, float]]:
    """
    Two-stage retrieval of opponent claims most relevant to the current claim.
    
    Stage 1: Fast embedding retrieval with unresolved-claim boost (PDF 8.2)
    Stage 2: Cross-encoder reranking for precise relevance scoring (PDF 8.1)
    
    Returns list of (ClaimRecord, score) sorted descending.
    """
    if not opponent_claims or not opponent_embeddings:
        return []

    from pipeline_a_embedding import retrieve_top_k_candidates

    # Build IDs and unresolved flags for retrieval
    candidate_ids = [c.claim_id for c in opponent_claims]
    unresolved_boost = [
        c.status in (ClaimStatus.UNANSWERED, ClaimStatus.PARTIALLY_ADDRESSED)
        for c in opponent_claims
    ]

    # Stage 1: Embedding-based top-k retrieval
    top_k_results = retrieve_top_k_candidates(
        query_embedding=current_claim_embedding,
        candidate_embeddings=opponent_embeddings,
        candidate_ids=candidate_ids,
        k=top_k,
        unresolved_boost=unresolved_boost
    )

    if not top_k_results:
        return []

    # Build ID→claim lookup
    id_to_claim = {c.claim_id: c for c in opponent_claims}

    # Stage 2: Cross-encoder reranking (PDF Section 8.1)
    cross_enc = get_cross_encoder()
    if cross_enc is not None:
        try:
            pairs = [
                (current_text, id_to_claim[cid].text)
                for cid, _ in top_k_results
                if cid in id_to_claim
            ]
            if pairs:
                ce_scores = cross_enc.predict(pairs)
                reranked = sorted(
                    zip([cid for cid, _ in top_k_results], ce_scores),
                    key=lambda x: x[1],
                    reverse=True
                )
                return [
                    (id_to_claim[cid], float(score))
                    for cid, score in reranked[:final_k]
                    if cid in id_to_claim
                ]
        except Exception as e:
            logger.warning(f"[Pipeline C] Cross-encoder reranking failed: {e}. Falling back to embedding order.")

    # Fallback: use embedding order if cross-encoder fails
    return [
        (id_to_claim[cid], float(score))
        for cid, score in top_k_results[:final_k]
        if cid in id_to_claim
    ]


def check_does_response_address_opponent(
    relevant_candidates: List[Tuple[ClaimRecord, float]],
    min_relevance_threshold: float = 0.30
) -> bool:
    """
    PDF Section 8.1: Separate "same topic" from "directly addresses opponent point".
    Returns True if at least one retrieved candidate has sufficient relevance.
    """
    if not relevant_candidates:
        return False
    return any(score >= min_relevance_threshold for _, score in relevant_candidates)


def build_retrieval_context(
    current_text: str,
    relevant_opponent_claims: List[Tuple[ClaimRecord, float]],
    recent_turn_window: List[str],
    max_context_chars: int = 1200
) -> str:
    """
    Build contextual text combining retrieved opponent claims and recent turns.
    Used to resolve pronoun references and contextual phrases (PDF Section 8.4).
    Bounded to prevent excessive inference cost.
    """
    context_parts = []

    # Recent turn window for pronoun resolution (PDF Section 8.4)
    if recent_turn_window:
        context_parts.append("Recent context:\n" + "\n".join(recent_turn_window[-2:]))

    # Retrieved opponent claims
    if relevant_opponent_claims:
        context_parts.append("Relevant opponent claims:")
        for claim, score in relevant_opponent_claims[:3]:
            context_parts.append(f"  [{claim.status.value}] {claim.text}")

    context = "\n".join(context_parts)
    return context[:max_context_chars]

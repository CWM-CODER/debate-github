"""
Pipeline D — NLI / Argument Relation Analysis (PDF Section 3, 7.2, 8.5)
Uses a cross-encoder NLI model to classify the logical relationship between
the current claim and retrieved opponent claims.

PDF Section 7.2: "Do not infer agreement or contradiction from embedding similarity.
Use a cross-encoder/NLI model and a stance signal after retrieval."

PDF Section 8.5: Relation labels: support, contradict, qualify, elaborate, neutral, unrelated.
PDF Section 12, Rule 1: "A contradiction is not automatically a strong rebuttal. 
'That is wrong' contains disagreement but little reasoning."

Model: cross-encoder/nli-deberta-v3-small (recommended in PDF Section 18)
"""
from __future__ import annotations
import logging
from typing import List, Tuple, Optional
from models import ArgumentRelation, RelationRecord, ClaimRecord

logger = logging.getLogger(__name__)

_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
_nli_model = None


def load_nli_model() -> None:
    """Load NLI model once at startup. Never reloaded per match."""
    global _nli_model
    if _nli_model is None:
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            logger.info(f"[Pipeline D] Loading NLI model: {_NLI_MODEL_NAME}")
            _nli_model = CrossEncoder(_NLI_MODEL_NAME)
            logger.info("[Pipeline D] NLI model loaded.")
        except Exception as e:
            logger.error(f"[Pipeline D] Failed to load NLI model: {e}")
            _nli_model = None


def get_nli_model():
    if _nli_model is None:
        load_nli_model()
    return _nli_model


def _nli_scores_to_relation(
    contradiction_score: float,
    entailment_score: float,
    neutral_score: float,
    similarity_score: float,
    current_text: str,
    opponent_text: str
) -> Tuple[ArgumentRelation, float, str]:
    """
    Map NLI output scores to debate-specific relation labels.
    
    PDF Section 8.5: support/contradict/qualify/elaborate/neutral/unrelated
    
    NLI labels from deberta: CONTRADICTION, ENTAILMENT, NEUTRAL
    We extend to 6 debate-relevant relations with heuristic rules.
    """
    # Qualify detection: moderate entailment + hedge language
    import re
    HEDGE = re.compile(
        r'\b(however|but|although|while|though|except|unless|'
        r'only if|in some cases|partly|partially|to some extent)\b',
        re.IGNORECASE
    )
    has_hedge = bool(HEDGE.search(current_text))

    max_score = max(contradiction_score, entailment_score, neutral_score)

    if contradiction_score > 0.55:
        if has_hedge:
            return ArgumentRelation.QUALIFY, contradiction_score, "Partially contests with qualification"
        return ArgumentRelation.CONTRADICT, contradiction_score, "Directly contradicts opponent claim"

    elif entailment_score > 0.55:
        # Distinguish support (agreeing with opponent) from elaboration (building on own)
        # In debate context: if same side claim → elaboration; cross-side → support (concession)
        if similarity_score > 0.80:
            return ArgumentRelation.ELABORATE, entailment_score, "Elaborates on a related point"
        return ArgumentRelation.SUPPORT, entailment_score, "Supports / concedes opponent's point"

    elif neutral_score > 0.55:
        if similarity_score > 0.50:
            return ArgumentRelation.NEUTRAL, neutral_score, "Same topic but no direct engagement"
        return ArgumentRelation.UNRELATED, neutral_score, "Unrelated to opponent claim"

    # Low-confidence fallback
    if has_hedge and max_score > 0.40:
        return ArgumentRelation.QUALIFY, max_score, "Qualifies a related claim"

    return ArgumentRelation.NEUTRAL, max_score, "No clear relation"


def analyze_relation(
    current_claim_text: str,
    opponent_claim_text: str,
    current_claim_id: str,
    opponent_claim_id: str,
    similarity_score: float = 0.0
) -> RelationRecord:
    """
    Run NLI cross-encoder on (current_claim, opponent_claim) pair.
    Returns a RelationRecord with the classified relation.
    
    PDF Section 14: "Cross-encoder/NLI only on top-k relevant candidate pairs"
    (not on the entire debate history).
    """
    nli_model = get_nli_model()

    if nli_model is None:
        # Fallback when model unavailable (PDF Section 15.2)
        logger.warning("[Pipeline D] NLI model unavailable — returning NEUTRAL fallback")
        return RelationRecord(
            current_claim_id=current_claim_id,
            opponent_claim_id=opponent_claim_id,
            relation=ArgumentRelation.NEUTRAL,
            score=0.0,
            reasoning_hint="NLI model unavailable"
        )

    try:
        # DeBERTa NLI outputs [contradiction, entailment, neutral] logits
        scores = nli_model.predict(
            [(current_claim_text, opponent_claim_text)],
            apply_softmax=True
        )[0]

        # deberta-v3-small label order: contradiction=0, entailment=1, neutral=2
        if len(scores) >= 3:
            contradiction_score = float(scores[0])
            entailment_score = float(scores[1])
            neutral_score = float(scores[2])
        else:
            # Some models return 2 labels; treat as entailment/contradiction
            contradiction_score = float(scores[0]) if len(scores) > 0 else 0.0
            entailment_score = float(scores[1]) if len(scores) > 1 else 0.0
            neutral_score = 1.0 - contradiction_score - entailment_score

        relation, score, hint = _nli_scores_to_relation(
            contradiction_score, entailment_score, neutral_score,
            similarity_score, current_claim_text, opponent_claim_text
        )

        logger.debug(
            f"[Pipeline D] {relation.value} (conf={score:.2f}): "
            f"'{current_claim_text[:40]}...' vs '{opponent_claim_text[:40]}...'"
        )

        return RelationRecord(
            current_claim_id=current_claim_id,
            opponent_claim_id=opponent_claim_id,
            relation=relation,
            score=score,
            reasoning_hint=hint
        )

    except Exception as e:
        logger.error(f"[Pipeline D] NLI inference error: {e}")
        return RelationRecord(
            current_claim_id=current_claim_id,
            opponent_claim_id=opponent_claim_id,
            relation=ArgumentRelation.NEUTRAL,
            score=0.0,
            reasoning_hint=f"Inference error: {type(e).__name__}"
        )


def analyze_all_relations(
    current_claims: List[ClaimRecord],
    relevant_opponent_pairs: List[Tuple[ClaimRecord, ClaimRecord, float]]
) -> List[RelationRecord]:
    """
    Analyze relations for all (current_claim, opponent_claim) pairs.
    PDF Section 8.3: "One response may challenge two, three, or more previous claims."
    
    relevant_opponent_pairs: list of (current_claim, opponent_claim, similarity_score)
    """
    results = []
    for current, opponent, similarity in relevant_opponent_pairs:
        record = analyze_relation(
            current_claim_text=current.text,
            opponent_claim_text=opponent.text,
            current_claim_id=current.claim_id,
            opponent_claim_id=opponent.claim_id,
            similarity_score=similarity
        )
        results.append(record)
    return results


def compute_rebuttal_signal(
    relations: List[RelationRecord],
    has_reasoning: bool = True
) -> float:
    """
    Compute rebuttal quality signal (0-1) from relation records.
    
    PDF Section 12, Rule 1: Contradiction alone ≠ strong rebuttal.
    A strong rebuttal requires reasoning + contradiction/qualification.
    
    Returns a score that the scoring engine weights at 20%.
    """
    if not relations:
        return 0.0

    score = 0.0
    for rel in relations:
        if rel.relation == ArgumentRelation.CONTRADICT:
            # Base contradiction credit; reasoning multiplier applied below
            score += 0.6 * rel.score
        elif rel.relation == ArgumentRelation.QUALIFY:
            score += 0.7 * rel.score  # qualification shows nuanced engagement
        elif rel.relation == ArgumentRelation.ELABORATE:
            score += 0.3 * rel.score
        elif rel.relation == ArgumentRelation.NEUTRAL:
            score += 0.1 * rel.score
        # SUPPORT/UNRELATED contribute 0 to rebuttal (concession / off-topic)

    # Normalize across number of relations
    score = min(1.0, score / max(len(relations), 1))

    # PDF Rule 1: reasoning required for strong rebuttal
    # If no reasoning detected, cap at 0.5 even if contradiction found
    if not has_reasoning:
        score = min(score, 0.50)

    return score


def detect_has_reasoning(text: str) -> bool:
    """
    Lightweight heuristic to detect presence of reasoning/justification.
    PDF Section 12, Rule 1: "That is wrong" → disagreement but no reasoning.
    """
    import re
    REASONING_MARKERS = re.compile(
        r'\b(because|therefore|thus|hence|since|as a result|'
        r'this means|which means|this shows|this demonstrates|'
        r'the reason|due to|given that|evidence shows|research shows|'
        r'studies show|data shows|for example|for instance|'
        r'this is why|consequently|it follows that|as we can see)\b',
        re.IGNORECASE
    )
    return bool(REASONING_MARKERS.search(text))


def check_stance_consistency(
    current_text: str,
    player_side: str,  # "FOR" or "AGAINST"
    topic: str
) -> float:
    """
    Simple stance consistency heuristic (PDF Section 11: 10% weight).
    Returns 0-1 score.
    
    In a full deployment this would use a stance classifier.
    Here we use a heuristic based on explicit position markers.
    """
    import re
    text_lower = current_text.lower()

    FOR_MARKERS = re.compile(
        r'\b(support|agree|advocate|favor|endorse|benefit|advantage|'
        r'positive|improve|better|should|must|can|will)\b',
        re.IGNORECASE
    )
    AGAINST_MARKERS = re.compile(
        r'\b(oppose|against|reject|disagree|harm|disadvantage|'
        r'negative|worse|risk|danger|should not|must not|cannot|will not)\b',
        re.IGNORECASE
    )

    for_count = len(FOR_MARKERS.findall(current_text))
    against_count = len(AGAINST_MARKERS.findall(current_text))

    if for_count + against_count == 0:
        return 0.6  # Neutral text; moderate stance score

    ratio = for_count / (for_count + against_count)

    if player_side == "FOR":
        # FOR side should have more FOR markers
        return min(1.0, 0.3 + ratio * 0.7)
    else:
        # AGAINST side should have more AGAINST markers
        return min(1.0, 0.3 + (1 - ratio) * 0.7)

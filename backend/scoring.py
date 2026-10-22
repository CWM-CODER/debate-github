"""
Scoring Engine — PDF Section 11 and 12
Deterministic aggregation of 5 independently measured signals.

PDF Section 11: "Scoring is a deterministic aggregation layer over several
independently measured signals. The system should NOT use a single model to
predict a winner directly."

Weights (from PDF Section 11):
- Counterargument/Rebuttal:  20%
- Logical Reasoning:          20%
- Evidence Quality (RAG):     20%
- Topic Relevance:            15%
- Clarity:                    15%
- Stance Consistency:         10%

Fairness rules (from PDF Section 12, all implemented):
1. Contradiction ≠ strong rebuttal (reasoning required)
2. True fact ≠ strong argument (evidence and reasoning scored separately)
3. Complex vocabulary ≠ reward (clarity = structure + coherence)
4. Repetition with new evidence is NOT penalized
5. Fact-check failure only affects evidence component
6. Unverified claims → "unverified", NOT "false"
"""
from __future__ import annotations
import logging
import re
from typing import List, Tuple
from models import (
    TurnScore, ClaimRecord, RelationRecord, VerificationResult,
    RepetitionClass, ClaimType, ArgumentRelation, VerificationVerdict
)

logger = logging.getLogger(__name__)

# ─── Scoring weights (PDF Section 11 exact values) ────────────────────────────
WEIGHT_REBUTTAL = 0.20
WEIGHT_REASONING = 0.20
WEIGHT_EVIDENCE = 0.20
WEIGHT_RELEVANCE = 0.15
WEIGHT_CLARITY = 0.15
WEIGHT_STANCE = 0.10


def _score_topic_relevance(
    claim_embeddings: List,  # np.ndarray per claim
    topic_embedding,          # np.ndarray
    claims: List[ClaimRecord]
) -> float:
    """
    15% weight. Semantic relation between topic and participant's claims.
    PDF Section 11: "semantic relation between the topic/question and participant's actual claims."
    """
    if not claim_embeddings or topic_embedding is None:
        return 0.5  # neutral default

    from pipeline_a_embedding import cosine_similarity
    import numpy as np

    sims = [cosine_similarity(emb, topic_embedding) for emb in claim_embeddings]
    avg_sim = float(np.mean(sims))

    # Normalize: cosine similarity in [0,1] for normalized vectors
    return min(1.0, max(0.0, avg_sim))


def _score_clarity(response_text: str, claims: List[ClaimRecord]) -> float:
    """
    15% weight. Structure, readability, coherent expression.
    PDF Section 11: "Should NOT reward unnecessarily complex vocabulary."
    PDF Rule 3: "Not rewarded for complicated English."
    
    We measure: sentence structure, appropriate length, discourse markers,
    NOT word length or rarity.
    """
    if not response_text:
        return 0.0

    text = response_text.strip()
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    score = 0.5  # baseline

    # Good length (not too short, not rambling) — 30-400 words is ideal
    word_count = len(words)
    if 30 <= word_count <= 400:
        score += 0.15
    elif word_count < 10:
        score -= 0.20

    # Sentence variety (multiple sentences = structured argument)
    if len(sentences) >= 2:
        score += 0.10
    if len(sentences) >= 4:
        score += 0.05

    # Discourse / coherence markers
    DISCOURSE_MARKERS = re.compile(
        r'\b(first|second|third|finally|furthermore|moreover|however|'
        r'therefore|thus|in addition|on the other hand|in contrast|'
        r'as a result|consequently|to summarize|in conclusion)\b',
        re.IGNORECASE
    )
    if DISCOURSE_MARKERS.search(text):
        score += 0.10

    # Penalize very short single sentence (likely underdeveloped)
    if len(sentences) == 1 and word_count < 20:
        score -= 0.15

    return round(min(1.0, max(0.0, score)), 3)


def _score_logical_reasoning(
    response_text: str,
    claims: List[ClaimRecord],
    relations: List[RelationRecord]
) -> float:
    """
    20% weight. Quality of reasoning connecting claims to position.
    PDF Section 11: "supported by argument structure signals."
    PDF Rule 2: "True fact ≠ strong argument — reasoning scored separately."
    """
    score = 0.4  # baseline

    # Reasoning connectors
    REASONING = re.compile(
        r'\b(because|therefore|thus|hence|since|as a result|which means|'
        r'this demonstrates|the reason is|due to|given that|it follows|'
        r'evidence shows|this proves|this suggests|this implies|'
        r'for example|for instance|specifically|in particular)\b',
        re.IGNORECASE
    )

    LOGICAL_STRUCTURE = re.compile(
        r'\b(if|then|when|unless|provided|assuming|given|'
        r'without|despite|although|even though|while|whereas)\b',
        re.IGNORECASE
    )

    reason_count = len(REASONING.findall(response_text))
    logic_count = len(LOGICAL_STRUCTURE.findall(response_text))

    score += min(0.25, reason_count * 0.08)
    score += min(0.15, logic_count * 0.05)

    # Multiple distinct claims with logical flow = better reasoning
    claim_count = len(claims)
    if claim_count >= 2:
        score += 0.10
    if claim_count >= 3:
        score += 0.05

    # Relations in relations show logical engagement
    engaged_relations = [
        r for r in relations
        if r.relation in (
            ArgumentRelation.CONTRADICT, ArgumentRelation.QUALIFY,
            ArgumentRelation.ELABORATE
        )
    ]
    if engaged_relations:
        score += min(0.10, len(engaged_relations) * 0.05)

    return round(min(1.0, max(0.0, score)), 3)


def _score_evidence_quality(
    claims: List[ClaimRecord]
) -> float:
    """
    20% weight. Quality and support of factual claims via Internet RAG.
    PDF Rule 5: Fact-check failure ≠ decides entire match (only evidence component).
    PDF Rule 6: Unverified → 'unverified', NOT 'false' (conservative score, not 0).
    """
    from pipeline_e_rag import evidence_quality_score

    checkable = [
        c for c in claims
        if c.claim_type in (ClaimType.FACTUAL, ClaimType.STATISTICAL)
    ]

    if not checkable:
        # No factual claims to verify — neutral score (not 0)
        # PDF: opinions etc are not checked; don't punish for having no facts
        return 0.55

    scores = [evidence_quality_score(c.verification) for c in checkable]
    return round(sum(scores) / len(scores), 3)


def _score_rebuttal(
    relations: List[RelationRecord],
    response_text: str,
    addressed_opponent_claims_count: int
) -> float:
    """
    20% weight. Whether participant directly addresses relevant opponent claims.
    PDF Section 11: "provides a meaningful response."
    PDF Rule 1: Contradiction alone ≠ strong rebuttal; reasoning required.
    """
    from pipeline_d_nli import compute_rebuttal_signal, detect_has_reasoning

    has_reasoning = detect_has_reasoning(response_text)
    base = compute_rebuttal_signal(relations, has_reasoning)

    # Bonus for addressing multiple opponent claims (PDF Section 8.3)
    if addressed_opponent_claims_count >= 2:
        base = min(1.0, base + 0.08)

    return round(base, 3)


def _compute_repetition_penalty(
    repetition_classes: List[RepetitionClass]
) -> float:
    """
    Repetition penalty.
    PDF Section 7.4 + Rule 4: Penalize ONLY genuine repetition.
    Rephrased, elaborated, or repeated-with-new-evidence → NOT penalized.
    """
    if not repetition_classes:
        return 0.0

    penalty = 0.0
    for cls in repetition_classes:
        if cls == RepetitionClass.REPEATED:
            penalty += 0.08  # genuine repetition penalty
        elif cls == RepetitionClass.REPHRASED:
            penalty += 0.02  # small penalty for rephrasing without adding value
        # ELABORATED, REPEATED_WITH_NEW_EVIDENCE, NEW → no penalty
        elif cls in (RepetitionClass.ELABORATED, RepetitionClass.REPEATED_WITH_NEW_EVIDENCE, RepetitionClass.NEW):
            penalty += 0.0

    # Bound penalty so one bad turn doesn't eliminate a player (PDF Section 13)
    return min(0.25, penalty)


def compute_turn_score(
    turn_id: str,
    player_id: str,
    round_number: int,
    response_text: str,
    claims: List[ClaimRecord],
    claim_embeddings: List,
    topic_embedding,
    relations: List[RelationRecord],
    repetition_classes: List[RepetitionClass],
    addressed_opponent_claims_count: int,
    player_side: str
) -> TurnScore:
    """
    Main scoring function. Aggregates all 5 criteria into a turn score.
    
    All scoring is deterministic (PDF Section 11 principle).
    The function calls each criterion independently and then combines.
    """
    # Compute each criterion independently
    relevance = _score_topic_relevance(claim_embeddings, topic_embedding, claims)
    clarity = _score_clarity(response_text, claims)
    reasoning = _score_logical_reasoning(response_text, claims, relations)
    rebuttal = _score_rebuttal(relations, response_text, addressed_opponent_claims_count)
    evidence = _score_evidence_quality(claims)
    stance = _check_stance_from_nli(response_text, player_side)

    # Repetition penalty (bounded — PDF Section 13)
    penalty = _compute_repetition_penalty(repetition_classes)

    # Weighted total (PDF Section 11 exact weights)
    weighted = (
        rebuttal * WEIGHT_REBUTTAL +
        reasoning * WEIGHT_REASONING +
        evidence * WEIGHT_EVIDENCE +
        relevance * WEIGHT_RELEVANCE +
        clarity * WEIGHT_CLARITY +
        stance * WEIGHT_STANCE
    )

    # Apply penalty (bounded to prevent a single turn from dominating — PDF Section 13)
    weighted_total = max(0.0, weighted - penalty)
    weighted_total = round(weighted_total, 4)

    # Generate human-readable feedback
    feedback = _generate_feedback(
        relevance=relevance, clarity=clarity, reasoning=reasoning,
        rebuttal=rebuttal, evidence=evidence, stance=stance,
        penalty=penalty, relations=relations, claims=claims,
        round_number=round_number
    )

    # Check if any RAG is still pending
    rag_pending = any(
        c.verification.verdict == VerificationVerdict.PENDING
        for c in claims
        if c.claim_type in (ClaimType.FACTUAL, ClaimType.STATISTICAL)
    )

    turn_score = TurnScore(
        turn_id=turn_id,
        player_id=player_id,
        round_number=round_number,
        topic_relevance_raw=relevance,
        clarity_raw=clarity,
        logical_reasoning_raw=reasoning,
        rebuttal_raw=rebuttal,
        evidence_quality_raw=evidence,
        stance_consistency_raw=stance,
        repetition_penalty=penalty,
        weighted_total=weighted_total,
        claims_analyzed=len(claims),
        relations_found=relations,
        repetition_classes=repetition_classes,
        feedback_text=feedback,
        rag_pending=rag_pending
    )

    logger.info(
        f"[Scoring] Turn {turn_id} | Player {player_id} | Round {round_number} | "
        f"Score={weighted_total:.3f} | "
        f"rel={relevance:.2f} cla={clarity:.2f} rea={reasoning:.2f} "
        f"reb={rebuttal:.2f} evi={evidence:.2f} sta={stance:.2f} pen={penalty:.2f}"
    )

    return turn_score


def _check_stance_from_nli(response_text: str, player_side: str) -> float:
    from pipeline_d_nli import check_stance_consistency
    return check_stance_consistency(response_text, player_side, "")


def _generate_feedback(
    relevance: float, clarity: float, reasoning: float,
    rebuttal: float, evidence: float, stance: float,
    penalty: float, relations: List[RelationRecord],
    claims: List[ClaimRecord], round_number: int
) -> str:
    """Generate structured human-readable feedback for the UI."""
    parts = []

    if relevance >= 0.75:
        parts.append("✅ Argument is well-aligned with the debate topic.")
    elif relevance < 0.45:
        parts.append("⚠️ Argument drifted somewhat from the core topic.")

    if clarity >= 0.75:
        parts.append("✅ Argument is clearly structured and easy to follow.")
    elif clarity < 0.45:
        parts.append("💡 Argument could be more clearly organized with better structure.")

    if reasoning >= 0.70:
        parts.append("✅ Strong reasoning connects claims to your position.")
    elif reasoning < 0.45:
        parts.append("💡 Add reasoning (because/therefore/this shows) to strengthen your argument.")

    if rebuttal >= 0.65:
        parts.append("✅ Effectively addressed opponent's argument.")
    elif rebuttal < 0.35 and round_number > 1:
        parts.append("⚠️ Did not directly respond to opponent's key claim.")

    # Evidence feedback
    checkable = [c for c in claims if c.claim_type in (ClaimType.FACTUAL, ClaimType.STATISTICAL)]
    if checkable:
        pending_count = sum(1 for c in checkable if c.verification.verdict == VerificationVerdict.PENDING)
        supported_count = sum(1 for c in checkable if c.verification.verdict == VerificationVerdict.SUPPORTED)
        refuted_count = sum(1 for c in checkable if c.verification.verdict == VerificationVerdict.REFUTED)

        if pending_count > 0:
            parts.append(f"🔍 Fact-checking {pending_count} claim(s) — evidence score will update.")
        if supported_count > 0:
            parts.append(f"✅ {supported_count} factual claim(s) are evidentially supported.")
        if refuted_count > 0:
            parts.append(f"❌ {refuted_count} factual claim(s) not supported by retrieved evidence.")

    if stance >= 0.75:
        parts.append("✅ Maintained consistent stance throughout.")
    elif stance < 0.40:
        parts.append("⚠️ Stance may be inconsistent with your assigned side.")

    if penalty >= 0.08:
        parts.append(f"⚠️ Some arguments were repetitive — try introducing new points.")

    return " ".join(parts) if parts else "Argument submitted and evaluated."


def aggregate_final_scores(p1_turns: List[TurnScore], p2_turns: List[TurnScore]) -> Tuple[float, float]:
    """
    Aggregate turn scores into final match scores.
    PDF Section 13: "aggregate turn-level results with bounded penalties;
    without allowing one anomalous turn to dominate the entire match."
    
    Strategy: weighted average with recency bias (later rounds matter slightly more).
    """
    def weighted_avg(turns: List[TurnScore]) -> float:
        if not turns:
            return 0.0
        # Later rounds (higher round_number) get slightly more weight
        total_weight = 0.0
        total_score = 0.0
        for i, t in enumerate(turns):
            weight = 1.0 + (t.round_number - 1) * 0.1
            total_score += t.weighted_total * weight
            total_weight += weight
        return total_score / total_weight if total_weight > 0 else 0.0

    p1_total = round(weighted_avg(p1_turns) * 100, 1)
    p2_total = round(weighted_avg(p2_turns) * 100, 1)

    return p1_total, p2_total


def generate_final_analysis(
    p1_turns: List[TurnScore], p2_turns: List[TurnScore],
    p1_name: str, p2_name: str, winner: str
) -> dict:
    """
    Return a rich structured analysis dict for the result page.
    Contains per-category scores, winner strengths, loser mistakes.
    """
    if not p1_turns or not p2_turns:
        return {}

    def avg(turns, attr):
        return sum(getattr(t, attr) for t in turns) / len(turns)

    p1 = {
        "relevance": round(avg(p1_turns, "topic_relevance_raw") * 100, 1),
        "clarity":   round(avg(p1_turns, "clarity_raw") * 100, 1),
        "reasoning": round(avg(p1_turns, "logical_reasoning_raw") * 100, 1),
        "rebuttal":  round(avg(p1_turns, "rebuttal_raw") * 100, 1),
        "evidence":  round(avg(p1_turns, "evidence_quality_raw") * 100, 1),
        "stance":    round(avg(p1_turns, "stance_consistency_raw") * 100, 1),
        "penalty":   round(sum(t.repetition_penalty for t in p1_turns) * 100, 1),
    }
    p2 = {
        "relevance": round(avg(p2_turns, "topic_relevance_raw") * 100, 1),
        "clarity":   round(avg(p2_turns, "clarity_raw") * 100, 1),
        "reasoning": round(avg(p2_turns, "logical_reasoning_raw") * 100, 1),
        "rebuttal":  round(avg(p2_turns, "rebuttal_raw") * 100, 1),
        "evidence":  round(avg(p2_turns, "evidence_quality_raw") * 100, 1),
        "stance":    round(avg(p2_turns, "stance_consistency_raw") * 100, 1),
        "penalty":   round(sum(t.repetition_penalty for t in p2_turns) * 100, 1),
    }

    w = p1 if winner == "p1" else p2
    l = p2 if winner == "p1" else p1
    winner_name = p1_name if winner == "p1" else p2_name
    loser_name  = p2_name if winner == "p1" else p1_name
    loser_id    = "p2"    if winner == "p1" else "p1"

    CATEGORIES = [
        ("rebuttal",  "Rebuttal",          "They directly contradicted and dismantled opponent claims."),
        ("evidence",  "Evidence Quality",   "They backed their claims with solid, verifiable facts."),
        ("reasoning", "Logical Reasoning",  "They built well-structured logical chains with strong connectors."),
        ("relevance", "Topic Relevance",    "Their arguments stayed laser-focused on the debate topic."),
        ("clarity",   "Clarity",            "Their arguments were clearly structured and easy to follow."),
        ("stance",    "Stance Consistency", "They maintained a consistent, coherent position throughout."),
    ]

    # Winner strengths: categories where winner beat loser by >=8 pts
    strengths = []
    for key, label, desc in CATEGORIES:
        if w[key] - l[key] >= 8:
            strengths.append({"category": label, "winner_score": w[key], "loser_score": l[key], "description": desc})

    if not strengths:
        best_key = max(CATEGORIES, key=lambda c: w[c[0]])[0]
        best = next(c for c in CATEGORIES if c[0] == best_key)
        strengths.append({"category": best[1], "winner_score": w[best_key], "loser_score": l[best_key], "description": best[2]})

    # Loser mistakes: categories where loser scored below 45
    mistakes = []
    MISTAKE_DETAILS = {
        "rebuttal":  "Failed to directly contradict opponent claims — arguments ran in parallel instead of clashing.",
        "evidence":  "Relied on unverified or opinion-based claims without factual backing.",
        "reasoning": "Lacked reasoning connectors (because/therefore/this shows) — claims felt unsupported.",
        "relevance": "Arguments drifted away from the core debate topic.",
        "clarity":   "Arguments were too short or lacked clear structure and flow.",
        "stance":    "Showed inconsistency in maintaining the assigned debate stance.",
    }
    for key, label, _ in CATEGORIES:
        if l[key] < 45:
            mistakes.append({
                "category": label,
                "score": l[key],
                "gap": round(w[key] - l[key], 1),
                "detail": MISTAKE_DETAILS[key]
            })

    if l["penalty"] > 8:
        mistakes.append({
            "category": "Repetition Penalty",
            "score": 0,
            "gap": l["penalty"],
            "detail": f"Lost {l['penalty']} points for repeating the same arguments without adding new evidence."
        })

    if not mistakes:
        mistakes.append({
            "category": "Overall Performance",
            "score": 0,
            "gap": 0,
            "detail": f"Performance was close — {winner_name} maintained a sharper edge across every category."
        })

    return {
        "winner_name":      winner_name,
        "loser_name":       loser_name,
        "loser_id":         loser_id,
        "p1_breakdown":     p1,
        "p2_breakdown":     p2,
        "winner_strengths": strengths,
        "loser_mistakes":   mistakes,
    }


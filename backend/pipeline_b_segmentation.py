"""
Pipeline B — Claim Segmentation and Claim Typing (PDF Section 6)
Splits a participant's response into atomic or near-atomic claims.
Classifies each claim's type before deeper evaluation.

Key insight from PDF (Section 6): "The same participant response can contain
several independent factual assertions, an opinion, a conditional statement,
and a rhetorical question. Each should be handled differently."

PDF Section 6.4: Do NOT strip qualifiers like if, only, may, could, not, unless.
PDF Section 6.3: Rhetorical questions are NOT automatically fact-checked.
PDF Section 6.2: Only checkable factual claims trigger web verification.
"""
from __future__ import annotations
import re
import logging
from typing import List, Tuple
from models import ClaimType

logger = logging.getLogger(__name__)

# ─── Linguistic patterns for claim typing ─────────────────────────────────────

# Conditional markers (PDF Section 6.4 — must preserve these)
CONDITIONAL_MARKERS = re.compile(
    r'\b(if|unless|provided that|assuming|only if|even if|in case|'
    r'given that|suppose|when|whenever|once|as long as)\b',
    re.IGNORECASE
)

# Uncertainty qualifiers (PDF Section 6.4 — must NOT be stripped)
QUALIFIER_MARKERS = re.compile(
    r'\b(may|might|could|can|possibly|perhaps|probably|likely|'
    r'unlikely|sometimes|often|usually|potentially|arguably)\b',
    re.IGNORECASE
)

# Statistical indicators (PDF Section 6.5)
STATISTICAL_MARKERS = re.compile(
    r'\b(\d+\.?\d*\s*%|\d+\.?\d*\s*(million|billion|thousand|percent)|'
    r'\d{4}\s+(study|report|data|survey)|according to|research shows|'
    r'statistics show|data shows|studies show|survey found)\b',
    re.IGNORECASE
)

# Rhetorical question patterns (PDF Section 6.3)
RHETORICAL_PATTERNS = re.compile(
    r'(don\'t you think|isn\'t it obvious|how can (anyone|we)|'
    r'who (would|could|should)|what (would|could|should)|'
    r'isn\'t that|aren\'t (we|they)|wouldn\'t|couldn\'t|shouldn\'t)',
    re.IGNORECASE
)

# Opinion / value judgment markers
OPINION_MARKERS = re.compile(
    r'\b(I (think|believe|feel|argue|contend)|in my (opinion|view)|'
    r'it seems|it appears|personally|for me|from my perspective|'
    r'we should|society should|people must|everyone ought|'
    r'fundamentally|morally|ethically|clearly|obviously|certainly)\b',
    re.IGNORECASE
)

# Future prediction markers
PREDICTION_MARKERS = re.compile(
    r'\b(will|shall|going to|in the future|soon|eventually|by \d{4}|'
    r'forecast|predict|expected to|projected to|anticipated to)\b',
    re.IGNORECASE
)


def split_into_sentences(text: str) -> List[str]:
    """
    Split response text into sentence-level segments.
    Uses a simple rule-based splitter to avoid spaCy dependency at segment stage.
    Preserves conditional structure — does not split on commas within conditions.
    """
    # Split on sentence-ending punctuation
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    # Additional split on semicolons that act as claim boundaries
    sentences = []
    for s in raw:
        parts = re.split(r';\s+(?=[A-Z])', s)
        sentences.extend(parts)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def classify_claim_type(sentence: str) -> ClaimType:
    """
    Classify a single sentence/claim into a claim type.
    Priority: RHETORICAL > STATISTICAL > CONDITIONAL > PREDICTION > OPINION > FACTUAL > UNCLEAR

    PDF Section 6.2: factual, opinion, prediction, conditional, value_judgment,
                     rhetorical/argumentative, statistic, or unclear.
    """
    text = sentence.strip()

    # Rhetorical questions (PDF 6.3: do not fact-check literally)
    if text.endswith('?') or RHETORICAL_PATTERNS.search(text):
        return ClaimType.RHETORICAL

    # Statistical (PDF 6.5: extract precise metric/value/context)
    if STATISTICAL_MARKERS.search(text):
        return ClaimType.STATISTICAL

    # Conditional (PDF 6.4: preserve qualifiers; different truth conditions)
    if CONDITIONAL_MARKERS.search(text):
        return ClaimType.CONDITIONAL

    # Prediction
    if PREDICTION_MARKERS.search(text):
        return ClaimType.PREDICTION

    # Opinion / value judgment
    if OPINION_MARKERS.search(text):
        return ClaimType.OPINION

    # Default: factual (checkable claim)
    if len(text.split()) >= 4:
        return ClaimType.FACTUAL

    return ClaimType.UNCLEAR


def is_fact_checkable(claim_type: ClaimType) -> bool:
    """
    Determines if a claim should trigger the Internet RAG pipeline.
    PDF Section 6.2: "Only checkable factual claims should normally trigger web verification."
    """
    return claim_type in (ClaimType.FACTUAL, ClaimType.STATISTICAL)


def segment_response(text: str, max_claims: int = 8) -> List[Tuple[str, ClaimType]]:
    """
    Main entry point: split response into (claim_text, claim_type) pairs.

    Implements:
    - PDF Section 6.1: multiple claims inside one response
    - PDF Section 6.2: mixed fact and opinion
    - PDF Section 6.3: hidden argument inside rhetorical question
    - PDF Section 6.4: conditional and qualified statements
    - PDF Section 6.5: numerical and statistical claims

    max_claims bounds inference cost (PDF Section 14, point 6).
    """
    if not text or not text.strip():
        return []

    sentences = split_into_sentences(text)

    # Bound claim count to control inference cost (PDF Section 14)
    sentences = sentences[:max_claims]

    results = []
    for sentence in sentences:
        claim_type = classify_claim_type(sentence)
        results.append((sentence, claim_type))
        logger.debug(f"[Pipeline B] Claim type '{claim_type.value}': {sentence[:60]}...")

    logger.info(f"[Pipeline B] Segmented {len(results)} claims from response")
    return results


def extract_statistical_metadata(text: str) -> dict:
    """
    For statistical claims, extract structured metadata (PDF Section 6.5):
    value, metric, direction, population, time, condition.
    """
    meta = {
        "numbers": re.findall(r'\d+\.?\d*', text),
        "percentages": re.findall(r'\d+\.?\d*\s*%', text),
        "years": re.findall(r'\b(19|20)\d{2}\b', text),
        "has_source_attribution": bool(re.search(
            r'\b(according to|source:|cited by|reported by|study by)\b',
            text, re.IGNORECASE
        )),
        "qualifiers_preserved": [
            m.group() for m in QUALIFIER_MARKERS.finditer(text)
        ]
    }
    return meta

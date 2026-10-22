"""
Pipeline E — Internet RAG Fact Verification (PDF Section 9 + 10)
Retrieves evidence from the Internet for checkable factual claims.
Returns SUPPORTED | PARTIALLY_SUPPORTED | REFUTED | INSUFFICIENT_EVIDENCE | AMBIGUOUS.

PDF Section 9: "Search retrieval and verification are separate steps."
PDF Section 9.1: "Retrieve passages, not just URLs."
PDF Section 9.2: "Retain conflicting evidence, return conflict/uncertainty status."
PDF Section 9.5: "Failure to find evidence does NOT prove participant is wrong."
PDF Section 14: "RAG runs in parallel; 8-second timeout; if delayed, mark pending."
PDF Section 15.6: "Cache repeated claims; backoff on rate limits."

What we will NOT do (PDF Section 22):
- NOT treat first web search result as proof
- NOT force true/false when evidence is incomplete
"""
from __future__ import annotations
import asyncio
import hashlib
import logging
import re
import time
from typing import List, Optional, Tuple
import httpx
from bs4 import BeautifulSoup
from models import VerificationVerdict, VerificationResult, EvidenceSource

logger = logging.getLogger(__name__)

# Global query cache — persists across matches (PDF Section 14, 15.6)
# Separate from match state (PDF Section 16)
_rag_cache: dict = {}
_CACHE_MAX_SIZE = 200  # bound size (PDF Section 15.5)

RAG_TIMEOUT = 8.0  # seconds (PDF Section 14, point 7)
MAX_SOURCES_PER_CLAIM = 4
MAX_REQUESTS_PER_TURN = 3  # rate limit guard (PDF Section 15.6)

# Source quality priority (PDF Section 9.3)
HIGH_QUALITY_DOMAINS = {
    "gov", "edu", "who.int", "un.org", "nih.gov", "cdc.gov",
    "worldbank.org", "nature.com", "science.org", "ncbi.nlm.nih.gov",
    "scholar.google.com", "pubmed.ncbi.nlm.nih.gov"
}
MEDIUM_QUALITY_DOMAINS = {
    "wikipedia.org", "britannica.com", "bbc.com", "reuters.com",
    "apnews.com", "theguardian.com", "nytimes.com", "economist.com"
}


def _get_domain_quality(url: str) -> float:
    """Source quality scoring (PDF Section 9.3)."""
    url_lower = url.lower()
    for domain in HIGH_QUALITY_DOMAINS:
        if domain in url_lower:
            return 0.9
    for domain in MEDIUM_QUALITY_DOMAINS:
        if domain in url_lower:
            return 0.65
    return 0.3  # Unknown/low-quality source


def _cache_key(claim_text: str) -> str:
    return hashlib.sha256(claim_text.strip().lower().encode()).hexdigest()[:16]


def _get_cached(claim_text: str) -> Optional[VerificationResult]:
    key = _cache_key(claim_text)
    return _rag_cache.get(key)


def _set_cache(claim_text: str, result: VerificationResult) -> None:
    key = _cache_key(claim_text)
    if len(_rag_cache) >= _CACHE_MAX_SIZE:
        # Evict oldest entry
        oldest = next(iter(_rag_cache))
        del _rag_cache[oldest]
    _rag_cache[key] = result


def _generate_search_query(claim_text: str) -> str:
    """
    Generate a focused search query from the claim.
    PDF Section 10, Step 2: "For numerical claims, include entities, metric, time context."
    """
    # Remove opinion markers to focus on factual content
    query = re.sub(
        r'\b(I think|I believe|in my opinion|it seems|personally)\b',
        '', claim_text, flags=re.IGNORECASE
    ).strip()

    # Truncate to reasonable search length
    if len(query) > 150:
        # Keep the most information-dense part
        query = query[:150].rsplit(' ', 1)[0]

    return query


async def _search_duckduckgo(query: str, max_results: int = 5) -> List[dict]:
    """
    Search DuckDuckGo for evidence sources (no API key required).
    Returns list of {url, title, snippet} dicts.
    """
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "url": r.get("href", ""),
                    "title": r.get("title", ""),
                    "snippet": r.get("body", "")
                })
        return results
    except Exception as e:
        logger.warning(f"[Pipeline E] DuckDuckGo search failed: {e}")
        return []


async def _extract_passage(url: str, claim_text: str, client: httpx.AsyncClient) -> Optional[str]:
    """
    Extract the most relevant passage from a URL (PDF Section 9.1: passages, not URLs).
    Returns the best passage or None if extraction fails.
    """
    try:
        resp = await client.get(url, timeout=4.0, follow_redirects=True)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, 'lxml')

        # Remove script, style, nav elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()

        # Get paragraph texts
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
        paragraphs = [p for p in paragraphs if len(p) > 50]

        if not paragraphs:
            return None

        # Score paragraphs by keyword overlap with claim
        claim_words = set(claim_text.lower().split())
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'to', 'of', 'in', 'on', 'at'}
        claim_words -= stop_words

        scored = []
        for para in paragraphs[:30]:  # check first 30 paragraphs
            para_words = set(para.lower().split())
            overlap = len(claim_words & para_words)
            scored.append((overlap, para))

        scored.sort(reverse=True)

        if scored and scored[0][0] > 0:
            return scored[0][1][:800]  # bounded passage length

    except Exception as e:
        logger.debug(f"[Pipeline E] Passage extraction failed for {url}: {e}")

    return None


def _verify_with_nli(passage: str, claim_text: str) -> Tuple[str, float]:
    """
    Run NLI verification between passage and claim.
    Returns (nli_label, confidence): "ENTAILMENT" | "CONTRADICTION" | "NEUTRAL"
    """
    try:
        from pipeline_d_nli import get_nli_model
        nli_model = get_nli_model()
        if nli_model is None:
            return "NEUTRAL", 0.0

        scores = nli_model.predict([(passage[:500], claim_text)], apply_softmax=True)[0]

        if len(scores) >= 3:
            labels = ["CONTRADICTION", "ENTAILMENT", "NEUTRAL"]
            best_idx = int(scores.argmax())
            return labels[best_idx], float(scores[best_idx])
        elif len(scores) == 2:
            if scores[0] > scores[1]:
                return "CONTRADICTION", float(scores[0])
            return "ENTAILMENT", float(scores[1])

    except Exception as e:
        logger.warning(f"[Pipeline E] NLI verification failed: {e}")

    return "NEUTRAL", 0.0


def _aggregate_verdicts(
    verdicts: List[Tuple[str, float, EvidenceSource]]
) -> Tuple[VerificationVerdict, float, bool]:
    """
    Aggregate multiple source verdicts into a final decision.
    PDF Section 9.2: "Retain conflicting evidence; return conflict/uncertainty status."
    PDF Section 9.6: "Verify each atomic claim separately."
    """
    if not verdicts:
        return VerificationVerdict.INSUFFICIENT_EVIDENCE, 0.0, False

    entailments = [(s, src) for label, s, src in verdicts if label == "ENTAILMENT"]
    contradictions = [(s, src) for label, s, src in verdicts if label == "CONTRADICTION"]
    neutrals = [(s, src) for label, s, src in verdicts if label == "NEUTRAL"]

    # Check for conflicting evidence (PDF Section 9.2)
    has_conflict = bool(entailments) and bool(contradictions)

    total_sources = len(verdicts)
    support_ratio = len(entailments) / total_sources

    # Weight by source quality
    weighted_support = sum(
        s * src.source_quality for s, src in entailments
    ) / max(len(entailments), 1) if entailments else 0.0

    weighted_refute = sum(
        s * src.source_quality for s, src in contradictions
    ) / max(len(contradictions), 1) if contradictions else 0.0

    if has_conflict:
        return VerificationVerdict.AMBIGUOUS, max(weighted_support, weighted_refute), True

    if support_ratio >= 0.6 and weighted_support >= 0.55:
        if support_ratio == 1.0 and weighted_support >= 0.75:
            return VerificationVerdict.SUPPORTED, weighted_support, False
        return VerificationVerdict.PARTIALLY_SUPPORTED, weighted_support, False

    if contradictions and weighted_refute >= 0.6:
        return VerificationVerdict.REFUTED, weighted_refute, False

    return VerificationVerdict.INSUFFICIENT_EVIDENCE, 0.0, False


async def verify_claim(claim_text: str, claim_type_str: str) -> VerificationResult:
    """
    Main entry point for fact verification (PDF Section 10).
    
    Steps (exact from PDF Section 10):
    1. Extract claim as-is, preserve qualifiers
    2. Generate focused search query
    3. Retrieve multiple candidate sources
    4. Extract relevant passages
    5. Rank passages (semantic similarity)
    6. Run NLI verification
    7. Check full/partial/contradicted support
    8. Return verdict with evidence and uncertainty
    """
    # Check cache first (PDF Section 14, 15.6)
    cached = _get_cached(claim_text)
    if cached is not None:
        logger.debug(f"[Pipeline E] Cache hit for: {claim_text[:50]}...")
        return cached

    # Generate search query (PDF Section 10, Step 2)
    query = _generate_search_query(claim_text)

    result = VerificationResult(verdict=VerificationVerdict.PENDING)

    try:
        async with asyncio.timeout(RAG_TIMEOUT):
            # Step 3: Retrieve sources (PDF Section 10, Step 3)
            search_results = await _search_duckduckgo(query, max_results=MAX_SOURCES_PER_CLAIM)

            if not search_results:
                result.verdict = VerificationVerdict.INSUFFICIENT_EVIDENCE
                result.notes = "No search results returned"
                return result

            # Step 4: Extract passages (PDF Section 10, Step 4)
            # Step 5: Rank passages (PDF Section 10, Step 5)
            # Run passage extraction in parallel
            async with httpx.AsyncClient(
                headers={"User-Agent": "Mozilla/5.0 (research bot)"},
                timeout=4.0
            ) as client:
                passage_tasks = [
                    _extract_passage(r["url"], claim_text, client)
                    for r in search_results[:MAX_SOURCES_PER_CLAIM]
                ]
                passages = await asyncio.gather(*passage_tasks, return_exceptions=True)

            # Step 6: Run NLI verification for each passage (PDF Section 10, Step 6)
            verdicts = []
            sources = []

            for i, (search_res, passage) in enumerate(zip(search_results, passages)):
                if isinstance(passage, Exception) or not passage:
                    # Use snippet as fallback passage
                    passage = search_res.get("snippet", "")

                if not passage:
                    continue

                url = search_res.get("url", "")
                quality = _get_domain_quality(url)

                source = EvidenceSource(
                    url=url,
                    domain=url.split("/")[2] if "/" in url else url,
                    title=search_res.get("title", ""),
                    source_quality=quality
                )

                # NLI verification (PDF Section 10, Step 6)
                nli_label, confidence = _verify_with_nli(passage, claim_text)

                # Weight by source quality (PDF Section 9.3)
                weighted_confidence = confidence * quality

                verdicts.append((nli_label, weighted_confidence, source))
                sources.append(source)

                # Collect evidence excerpts
                if nli_label in ("ENTAILMENT", "CONTRADICTION"):
                    result.evidence_passages.append(passage[:300])

            result.sources = sources

            # Step 7-8: Aggregate (PDF Section 10, Step 7-8)
            verdict, confidence, conflicting = _aggregate_verdicts(verdicts)

            result.verdict = verdict
            result.confidence = confidence
            result.conflicting_evidence = conflicting
            result.notes = "Conflicting sources found" if conflicting else ""

    except asyncio.TimeoutError:
        # PDF Section 14: "If RAG is delayed, mark pending; do not freeze debate UI"
        logger.warning(f"[Pipeline E] RAG timeout for: {claim_text[:50]}...")
        result.verdict = VerificationVerdict.PENDING
        result.notes = "Verification timeout — result pending"
        return result  # Do NOT cache timeouts

    except Exception as e:
        logger.error(f"[Pipeline E] RAG error: {e}")
        result.verdict = VerificationVerdict.INSUFFICIENT_EVIDENCE
        result.notes = f"Verification error: {type(e).__name__}"

    # Cache successful results (not timeouts)
    if result.verdict != VerificationVerdict.PENDING:
        _set_cache(claim_text, result)

    return result


def evidence_quality_score(verification_result: VerificationResult) -> float:
    """
    Convert verification verdict into scoring signal (0-1) for the scoring engine.
    PDF Section 11: Evidence Quality = 20% weight.
    PDF Section 12, Rule 5: "Fact verification failure ≠ decides entire match."
    PDF Section 12, Rule 6: "Unverified → labeled 'unverified', NOT 'false'."
    """
    verdicts_to_score = {
        VerificationVerdict.SUPPORTED: 1.0,
        VerificationVerdict.PARTIALLY_SUPPORTED: 0.6,
        VerificationVerdict.AMBIGUOUS: 0.4,
        VerificationVerdict.INSUFFICIENT_EVIDENCE: 0.3,  # NOT 0 (PDF Rule 6)
        VerificationVerdict.PENDING: 0.3,
        VerificationVerdict.NOT_CHECKABLE: 0.5,  # opinions etc — neutral
        VerificationVerdict.REFUTED: 0.05,  # low but NOT 0 (PDF Rule 5 — bounded)
    }
    base = verdicts_to_score.get(verification_result.verdict, 0.3)

    # Adjust by source quality if we have sources
    if verification_result.sources:
        avg_quality = sum(s.source_quality for s in verification_result.sources) / len(verification_result.sources)
        # Small quality adjustment
        base = base * (0.7 + 0.3 * avg_quality)

    return round(min(1.0, base), 3)

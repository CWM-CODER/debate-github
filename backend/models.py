"""
Pydantic models and enums for Clash of Minds debate state.
Faithful to the PDF workflow — every field corresponds to a documented state.
"""
from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import uuid
import time


# ─── Enums ──────────────────────────────────────────────────────────────────

class ClaimType(str, Enum):
    FACTUAL = "factual"
    OPINION = "opinion"
    CONDITIONAL = "conditional"
    RHETORICAL = "rhetorical"
    STATISTICAL = "statistical"
    PREDICTION = "prediction"
    VALUE_JUDGMENT = "value_judgment"
    UNCLEAR = "unclear"


class ClaimStatus(str, Enum):
    UNANSWERED = "unanswered"
    PARTIALLY_ADDRESSED = "partially_addressed"
    RESOLVED = "resolved"


class VerificationVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    REFUTED = "REFUTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    AMBIGUOUS = "AMBIGUOUS"
    PENDING = "PENDING"
    NOT_CHECKABLE = "NOT_CHECKABLE"  # opinions, rhetorical, etc.


class ArgumentRelation(str, Enum):
    SUPPORT = "support"
    CONTRADICT = "contradict"
    QUALIFY = "qualify"
    ELABORATE = "elaborate"
    NEUTRAL = "neutral"
    UNRELATED = "unrelated"


class RepetitionClass(str, Enum):
    REPEATED = "repeated"
    REPHRASED = "rephrased"
    ELABORATED = "elaborated"
    REPEATED_WITH_NEW_EVIDENCE = "repeated_with_new_evidence"
    NEW = "new"


class PlayerSide(str, Enum):
    FOR = "FOR"
    AGAINST = "AGAINST"


class InputMethod(str, Enum):
    TEXT = "text"
    VOICE = "voice"


class MatchPhase(str, Enum):
    SETUP = "setup"
    ACTIVE = "active"
    FINISHED = "finished"


# ─── Claim-level models ───────────────────────────────────────────────────────

class EvidenceSource(BaseModel):
    url: str
    domain: str
    title: str
    publication_date: Optional[str] = None
    retrieval_date: str = Field(default_factory=lambda: str(time.time()))
    source_quality: float = 0.5  # 0-1 priority score


class VerificationResult(BaseModel):
    verdict: VerificationVerdict = VerificationVerdict.PENDING
    evidence_passages: List[str] = Field(default_factory=list)
    sources: List[EvidenceSource] = Field(default_factory=list)
    confidence: float = 0.0
    conflicting_evidence: bool = False
    notes: str = ""


class ClaimRecord(BaseModel):
    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str
    player_id: str   # "p1" or "p2"
    round_number: int
    claim_type: ClaimType = ClaimType.UNCLEAR
    status: ClaimStatus = ClaimStatus.UNANSWERED
    embedding: Optional[List[float]] = None  # stored separately in FAISS
    verification: VerificationResult = Field(default_factory=VerificationResult)
    addressed_by: List[str] = Field(default_factory=list)  # claim_ids that addressed this


class RelationRecord(BaseModel):
    """Records the NLI relation between a current claim and a retrieved opponent claim."""
    current_claim_id: str
    opponent_claim_id: str
    relation: ArgumentRelation
    score: float  # cross-encoder confidence
    reasoning_hint: str = ""


# ─── Turn-level models ────────────────────────────────────────────────────────

class TurnScore(BaseModel):
    """Per-criterion scores for one turn. Weights from PDF Section 11."""
    turn_id: str
    player_id: str
    round_number: int

    # Raw signals (0-1)
    topic_relevance_raw: float = 0.0       # 15%
    clarity_raw: float = 0.0               # 15%
    logical_reasoning_raw: float = 0.0    # 20%
    rebuttal_raw: float = 0.0             # 20%
    evidence_quality_raw: float = 0.0     # 20%
    stance_consistency_raw: float = 0.0   # 10%

    # Penalty
    repetition_penalty: float = 0.0       # deducted from total

    # Weighted final
    weighted_total: float = 0.0

    # Diagnosis
    claims_analyzed: int = 0
    relations_found: List[RelationRecord] = Field(default_factory=list)
    repetition_classes: List[RepetitionClass] = Field(default_factory=list)
    feedback_text: str = ""
    rag_pending: bool = False  # True if RAG still running when score emitted


class PlayerState(BaseModel):
    player_id: str  # "p1" or "p2"
    name: str
    side: PlayerSide
    total_score: float = 0.0
    turn_scores: List[TurnScore] = Field(default_factory=list)
    claim_ids: List[str] = Field(default_factory=list)  # claims this player made


# ─── Match-level models ───────────────────────────────────────────────────────

class MatchState(BaseModel):
    """
    Top-level debate memory. Cleared on reset; model weights remain loaded.
    Corresponds directly to PDF Section 16 debate memory design.
    """
    match_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    topic: str = ""
    proposition: str = ""  # clearly worded proposition form of topic

    phase: MatchPhase = MatchPhase.SETUP
    current_round: int = 1
    max_rounds: int = 4
    current_player_id: str = "p1"  # whose turn it is

    players: Dict[str, PlayerState] = Field(default_factory=dict)
    claim_state: Dict[str, ClaimRecord] = Field(default_factory=dict)
    turn_scores: List[TurnScore] = Field(default_factory=list)
    unanswered_claim_ids: List[str] = Field(default_factory=list)

    # Per-turn idempotency guard (PDF Section 15.3)
    processed_turn_ids: List[str] = Field(default_factory=list)

    # RAG query cache (PDF Section 14, Section 15.6)
    rag_cache: Dict[str, VerificationResult] = Field(default_factory=dict)

    # Transcript/turn log
    turn_log: List[Dict[str, Any]] = Field(default_factory=list)

    winner: Optional[str] = None  # player_id of winner
    created_at: float = Field(default_factory=time.time)
    last_activity: float = Field(default_factory=time.time)


# ─── API Request/Response models ─────────────────────────────────────────────

class StartMatchRequest(BaseModel):
    p1_name: str
    p2_name: str
    topic: str
    max_rounds: int = 4  # configurable per PDF Section 13


class SubmitTurnRequest(BaseModel):
    match_id: str
    player_id: str
    turn_id: str  # client-generated UUID for idempotency
    text: str
    input_method: InputMethod = InputMethod.TEXT


class TurnResult(BaseModel):
    match_id: str
    turn_id: str
    player_id: str
    round_number: int
    score: TurnScore
    claims: List[ClaimRecord]
    relations: List[RelationRecord]
    debate_over: bool = False
    winner: Optional[str] = None
    next_player_id: Optional[str] = None


class MatchSummary(BaseModel):
    match_id: str
    topic: str
    winner: Optional[str]
    winner_name: Optional[str]
    p1: PlayerState
    p2: PlayerState
    turn_log: List[Dict[str, Any]]
    total_rounds: int

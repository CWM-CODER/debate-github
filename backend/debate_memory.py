"""
Debate Memory Store — PDF Section 16
Structured, per-match, in-memory debate state with claim tracking.
Model weights stay loaded across matches; only match state is cleared on reset.
"""
from __future__ import annotations
import asyncio
import logging
import time
from typing import Dict, Optional

from models import (
    MatchState, PlayerState, PlayerSide, ClaimRecord, ClaimStatus,
    TurnScore, MatchPhase
)

logger = logging.getLogger(__name__)

# Active matches in memory. Key = match_id.
# Separate from model weights (PDF Section 16: "key distinction")
_matches: Dict[str, MatchState] = {}
_lock = asyncio.Lock()

INACTIVITY_TIMEOUT = 3600  # 1 hour (PDF Section 15.4)


async def create_match(
    match_id: str,
    p1_name: str,
    p2_name: str,
    topic: str,
    max_rounds: int = 4
) -> MatchState:
    """Create and register a new match. Clears any old match with same ID."""
    async with _lock:
        state = MatchState(
            match_id=match_id,
            topic=topic,
            proposition=topic,
            max_rounds=max_rounds,
            current_round=1,
            current_player_id="p1",
            phase=MatchPhase.ACTIVE,
            players={
                "p1": PlayerState(
                    player_id="p1",
                    name=p1_name,
                    side=PlayerSide.FOR
                ),
                "p2": PlayerState(
                    player_id="p2",
                    name=p2_name,
                    side=PlayerSide.AGAINST
                )
            }
        )
        _matches[match_id] = state
        logger.info(f"[Memory] Created match {match_id}: {p1_name} vs {p2_name}")
        return state


async def get_match(match_id: str) -> Optional[MatchState]:
    async with _lock:
        return _matches.get(match_id)


async def save_match(state: MatchState) -> None:
    """Persist updated match state back to the in-memory store."""
    async with _lock:
        state.last_activity = time.time()
        _matches[state.match_id] = state


async def reset_match(match_id: str) -> None:
    """
    Delete match-specific state. Model weights remain loaded (PDF Section 16).
    Called by operator after each debate ends.
    """
    async with _lock:
        if match_id in _matches:
            del _matches[match_id]
            logger.info(f"[Memory] Reset match {match_id}")


async def add_claim(state: MatchState, claim: ClaimRecord) -> None:
    """Add a claim to the debate memory and link it to the player."""
    state.claim_state[claim.claim_id] = claim
    if claim.player_id in state.players:
        state.players[claim.player_id].claim_ids.append(claim.claim_id)
    # All new claims start as unanswered for the opponent
    if claim.claim_id not in state.unanswered_claim_ids:
        state.unanswered_claim_ids.append(claim.claim_id)


async def mark_claim_addressed(
    state: MatchState,
    opponent_claim_id: str,
    addressing_claim_id: str,
    partial: bool = False
) -> None:
    """Update the response status of an opponent claim (PDF Section 8.2)."""
    if opponent_claim_id in state.claim_state:
        claim = state.claim_state[opponent_claim_id]
        if not partial:
            claim.status = ClaimStatus.RESOLVED
            if opponent_claim_id in state.unanswered_claim_ids:
                state.unanswered_claim_ids.remove(opponent_claim_id)
        else:
            claim.status = ClaimStatus.PARTIALLY_ADDRESSED
        claim.addressed_by.append(addressing_claim_id)


def get_player_claims(state: MatchState, player_id: str) -> list[ClaimRecord]:
    """Return all claims made by a player, ordered by round."""
    ids = state.players.get(player_id, PlayerState(
        player_id=player_id, name="", side=PlayerSide.FOR
    )).claim_ids
    return [state.claim_state[c] for c in ids if c in state.claim_state]


def get_opponent_claims(state: MatchState, player_id: str) -> list[ClaimRecord]:
    """Return all claims from the opponent (including unanswered)."""
    opp_id = "p2" if player_id == "p1" else "p1"
    return get_player_claims(state, opp_id)


def get_unanswered_opponent_claims(
    state: MatchState, player_id: str
) -> list[ClaimRecord]:
    """
    Return opponent claims still unanswered — used by relevant-point retrieval
    to weight unresolved claims higher (PDF Section 8.2).
    """
    opp_claims = get_opponent_claims(state, player_id)
    return [
        c for c in opp_claims
        if c.status in (ClaimStatus.UNANSWERED, ClaimStatus.PARTIALLY_ADDRESSED)
    ]


async def expire_inactive_matches() -> None:
    """
    Clean up matches that have been inactive for INACTIVITY_TIMEOUT seconds.
    Prevents unbounded memory growth during 5-6 hour exhibition (PDF Section 15.5).
    """
    async with _lock:
        now = time.time()
        to_delete = [
            mid for mid, m in _matches.items()
            if now - m.last_activity > INACTIVITY_TIMEOUT
        ]
        for mid in to_delete:
            del _matches[mid]
            logger.info(f"[Memory] Expired inactive match {mid}")


def get_active_match_count() -> int:
    return len(_matches)


def get_latest_active_match() -> Optional[MatchState]:
    """Return the most recently created or active match."""
    active = [m for m in _matches.values() if m.phase == MatchPhase.ACTIVE]
    if not active:
        return None
    return max(active, key=lambda m: m.last_activity)


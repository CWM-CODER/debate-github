"""
Clash of Minds — FastAPI Backend (PDF Section 18-19)
WebSocket-based live debate evaluation system.

Architecture:
- POST /api/match/start  — create new match
- POST /api/match/submit — submit turn argument
- POST /api/match/reset  — operator reset (PDF Section 15.4)
- GET  /api/match/{id}   — get current state
- WS   /ws/{match_id}    — real-time updates

PDF Section 15.3: Idempotent turn processing via turn_id validation.
PDF Section 14: RAG runs in background; score emitted immediately with pending status.
PDF Section 15.5: Memory bounds — match state cleared on reset; models stay loaded.
"""
from __future__ import annotations
import asyncio
import logging
import uuid
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from models import (
    MatchState, ClaimRecord, ClaimType, VerificationVerdict,
    RepetitionClass, TurnScore, MatchPhase, MatchSummary,
    StartMatchRequest, SubmitTurnRequest, TurnResult, PlayerState
)
import debate_memory as memory
from pipeline_b_segmentation import segment_response, is_fact_checkable
from pipeline_e_rag import verify_claim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("clash_of_minds")

# ─── WebSocket connection manager ────────────────────────────────────────────

class ConnectionManager:
    """Manages WebSocket connections per match_id."""
    def __init__(self):
        self.connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, match_id: str, ws: WebSocket):
        await ws.accept()
        self.connections.setdefault(match_id, set()).add(ws)
        logger.info(f"[WS] Client connected to match {match_id}")

    def disconnect(self, match_id: str, ws: WebSocket):
        if match_id in self.connections:
            self.connections[match_id].discard(ws)
        logger.info(f"[WS] Client disconnected from match {match_id}")

    async def broadcast(self, match_id: str, data: dict):
        """Send JSON data to all connected clients for this match."""
        dead = set()
        for ws in self.connections.get(match_id, set()):
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.connections[match_id].discard(ws)


manager = ConnectionManager()


# ─── Application lifespan ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all NLP models ONCE at startup (PDF Section 14 + 22).
    Never reloaded per match.
    """
    logger.info("=== Clash of Minds starting — loading NLP models ===")

    # Load models with individual error handling (PDF Section 15.2)
    try:
        from pipeline_a_embedding import load_model as load_embedding
        load_embedding()
    except Exception as e:
        logger.error(f"[Startup] Embedding model load failed: {e}")

    try:
        from pipeline_c_retrieval import load_cross_encoder
        load_cross_encoder()
    except Exception as e:
        logger.error(f"[Startup] Cross-encoder load failed: {e}")

    try:
        from pipeline_d_nli import load_nli_model
        load_nli_model()
    except Exception as e:
        logger.error(f"[Startup] NLI model load failed: {e}")

    # Start background cleanup task
    cleanup_task = asyncio.create_task(_periodic_cleanup())

    logger.info("=== All models loaded — ready for debates ===")
    yield

    cleanup_task.cancel()
    logger.info("=== Clash of Minds shutting down ===")


async def _periodic_cleanup():
    """Expire inactive matches every 10 minutes (PDF Section 15.5)."""
    while True:
        await asyncio.sleep(600)
        await memory.expire_inactive_matches()


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Clash of Minds",
    description="AI-Powered Live Debate Evaluation — CSES ML Exhibition NIT Warangal",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.post("/api/match/start")
async def start_match(req: StartMatchRequest):
    """Create a new debate match."""
    match_id = str(uuid.uuid4())[:12]
    state = await memory.create_match(
        match_id=match_id,
        p1_name=req.p1_name.strip(),
        p2_name=req.p2_name.strip(),
        topic=req.topic.strip(),
        max_rounds=req.max_rounds
    )
    logger.info(f"[API] Match created: {match_id} | {req.p1_name} vs {req.p2_name} | Topic: {req.topic}")
    return {
        "match_id": match_id,
        "topic": req.topic,
        "max_rounds": req.max_rounds,
        "p1": {"name": req.p1_name, "side": "FOR"},
        "p2": {"name": req.p2_name, "side": "AGAINST"},
        "current_player": "p1",
        "current_round": 1
    }


@app.get("/api/match/{match_id}")
async def get_match(match_id: str):
    """Get current match state."""
    state = await memory.get_match(match_id)
    if not state:
        raise HTTPException(status_code=404, detail="Match not found")
    return _serialize_state(state)


@app.post("/api/match/submit")
async def submit_turn(req: SubmitTurnRequest):
    """
    Submit a participant's argument for a turn.
    
    Validates: player_id, match_id, turn_id (idempotency), turn state (PDF 15.3).
    Runs pipelines A-D synchronously (fast path).
    Runs pipeline E (RAG) in background (PDF Section 14).
    """
    state = await memory.get_match(req.match_id)
    if not state:
        raise HTTPException(status_code=404, detail="Match not found")

    if state.phase != MatchPhase.ACTIVE:
        raise HTTPException(status_code=400, detail="Match is not active")

    # ── Idempotency guard (PDF Section 15.3) ──
    if req.turn_id in state.processed_turn_ids:
        raise HTTPException(status_code=409, detail="Turn already processed")

    # ── Player and turn validation ──
    if req.player_id != state.current_player_id:
        raise HTTPException(status_code=400, detail=f"It is not {req.player_id}'s turn")

    if req.player_id not in state.players:
        raise HTTPException(status_code=400, detail="Invalid player_id")

    # ── Input validation (PDF Section 4, Step 6) ──
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Argument text cannot be empty")
    if len(text.split()) > 500:  # bound argument length (PDF Section 14)
        text = " ".join(text.split()[:500])

    # Mark turn as being processed
    state.processed_turn_ids.append(req.turn_id)

    # ── Pipeline B: Claim Segmentation ──
    raw_claims = segment_response(text, max_claims=8)
    player = state.players[req.player_id]

    # ── Pipeline A: Semantic Embeddings ──
    from pipeline_a_embedding import encode, encode_single
    import numpy as np

    topic_embedding = encode_single(state.topic)
    claim_texts = [c[0] for c in raw_claims]
    claim_embeddings_arr = encode(claim_texts) if claim_texts else np.array([])

    # Build ClaimRecord objects
    new_claims = []
    for i, (claim_text, claim_type) in enumerate(raw_claims):
        emb = claim_embeddings_arr[i].tolist() if len(claim_embeddings_arr) > i else None
        claim = ClaimRecord(
            text=claim_text,
            player_id=req.player_id,
            round_number=state.current_round,
            claim_type=claim_type,
            embedding=emb
        )
        new_claims.append(claim)
        await memory.add_claim(state, claim)

    # ── Pipeline A: Repetition Detection ──
    own_claims = memory.get_player_claims(state, req.player_id)
    own_prev_embeddings = [
        np.array(c.embedding) for c in own_claims[:-len(new_claims)]
        if c.embedding
    ]

    from pipeline_a_embedding import detect_repetition_type
    repetition_classes = []
    for i, claim in enumerate(new_claims):
        if claim.embedding:
            cls, sim = detect_repetition_type(
                np.array(claim.embedding),
                own_prev_embeddings
            )
            repetition_classes.append(cls)
        else:
            repetition_classes.append(RepetitionClass.NEW)

    # ── Pipeline C: Relevant Point Retrieval ──
    opponent_claims = memory.get_opponent_claims(state, req.player_id)
    opponent_embeddings = [
        np.array(c.embedding) for c in opponent_claims if c.embedding
    ]
    opponent_with_embeddings = [c for c in opponent_claims if c.embedding]

    from pipeline_c_retrieval import (
        retrieve_relevant_opponent_claims,
        check_does_response_address_opponent,
        build_retrieval_context
    )

    # Recent turn window for pronoun resolution (PDF Section 8.4)
    recent_turns = [t.get("text", "") for t in state.turn_log[-4:]]

    all_relations = []
    addressed_count = 0

    if opponent_with_embeddings:
        for claim in new_claims:
            if not claim.embedding:
                continue
            relevant = retrieve_relevant_opponent_claims(
                current_text=claim.text,
                current_claim_embedding=np.array(claim.embedding),
                opponent_claims=opponent_with_embeddings,
                opponent_embeddings=opponent_embeddings,
                top_k=5,
                final_k=3
            )

            does_address = check_does_response_address_opponent(relevant)
            if does_address:
                addressed_count += 1

            # ── Pipeline D: NLI Relation Analysis ──
            from pipeline_d_nli import analyze_relation
            for opp_claim, score in relevant:
                relation = analyze_relation(
                    current_claim_text=claim.text,
                    opponent_claim_text=opp_claim.text,
                    current_claim_id=claim.claim_id,
                    opponent_claim_id=opp_claim.claim_id,
                    similarity_score=score
                )
                all_relations.append(relation)

                # Update claim status in memory
                from models import ArgumentRelation
                if relation.relation in (ArgumentRelation.CONTRADICT, ArgumentRelation.QUALIFY):
                    await memory.mark_claim_addressed(
                        state, opp_claim.claim_id, claim.claim_id,
                        partial=(relation.relation == ArgumentRelation.QUALIFY)
                    )
                elif relation.relation == ArgumentRelation.ELABORATE:
                    await memory.mark_claim_addressed(
                        state, opp_claim.claim_id, claim.claim_id, partial=True
                    )

    # ── Pipeline E: Internet RAG (runs in background) ──
    # Fast path continues; RAG updates evidence async (PDF Section 14)
    async def run_rag_for_claims():
        checkable = [c for c in new_claims if is_fact_checkable(c.claim_type)]
        tasks = []
        for claim in checkable[:3]:  # cap per turn (PDF Section 15.6)
            tasks.append(_verify_and_update(state, claim, req.match_id))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    is_last_turn = (state.current_round >= state.max_rounds and req.player_id == "p2")
    if is_last_turn:
        # Wait for fact-check on the very last turn (15s max) so score is accurate
        try:
            await asyncio.wait_for(run_rag_for_claims(), timeout=15.0)
        except asyncio.TimeoutError:
            logger.warning("[API] RAG timed out on final turn — proceeding with current scores")
    else:
        asyncio.create_task(run_rag_for_claims())

    # ── Scoring Engine ──
    from scoring import compute_turn_score, aggregate_final_scores, generate_final_analysis

    claim_embeddings_list = [
        np.array(c.embedding) for c in new_claims if c.embedding
    ]

    turn_score = compute_turn_score(
        turn_id=req.turn_id,
        player_id=req.player_id,
        round_number=state.current_round,
        response_text=text,
        claims=new_claims,
        claim_embeddings=claim_embeddings_list,
        topic_embedding=topic_embedding,
        relations=all_relations,
        repetition_classes=repetition_classes,
        addressed_opponent_claims_count=addressed_count,
        player_side=player.side.value
    )

    # Update player state
    state.players[req.player_id].turn_scores.append(turn_score)
    state.players[req.player_id].total_score += turn_score.weighted_total * 100

    # Log turn
    state.turn_log.append({
        "turn_id": req.turn_id,
        "player_id": req.player_id,
        "player_name": player.name,
        "round": state.current_round,
        "text": text,
        "input_method": req.input_method.value,
        "score": turn_score.weighted_total,
        "timestamp": time.time()
    })

    # ── Advance turn state ──
    debate_over = False
    winner = None
    final_summary = ""

    next_player = "p2" if req.player_id == "p1" else "p1"

    # Each round = both players speak; after p2 speaks, increment round
    if req.player_id == "p2":
        state.current_round += 1

    if state.current_round > state.max_rounds:
        # Debate finished — compute final winner
        p1_turns = state.players["p1"].turn_scores
        p2_turns = state.players["p2"].turn_scores
        p1_total, p2_total = aggregate_final_scores(p1_turns, p2_turns)

        state.players["p1"].total_score = p1_total
        state.players["p2"].total_score = p2_total

        winner = "p1" if p1_total > p2_total else "p2"
        state.winner = winner
        state.phase = MatchPhase.FINISHED
        debate_over = True
        
        final_summary = generate_final_analysis(
            p1_turns, p2_turns,
            state.players["p1"].name, state.players["p2"].name,
            winner
        )
        logger.info(f"[API] Debate over! Winner: {winner} ({state.players[winner].name})")
    else:
        state.current_player_id = next_player

    await memory.save_match(state)

    # ── Build response ──
    result = {
        "type": "turn_result",
        "match_id": req.match_id,
        "turn_id": req.turn_id,
        "player_id": req.player_id,
        "player_name": player.name,
        "round_number": state.current_round if not debate_over else state.max_rounds,
        "score": {
            "topic_relevance": round(turn_score.topic_relevance_raw * 100, 1),
            "clarity": round(turn_score.clarity_raw * 100, 1),
            "logical_reasoning": round(turn_score.logical_reasoning_raw * 100, 1),
            "rebuttal": round(turn_score.rebuttal_raw * 100, 1),
            "evidence_quality": round(turn_score.evidence_quality_raw * 100, 1),
            "stance_consistency": round(turn_score.stance_consistency_raw * 100, 1),
            "repetition_penalty": round(turn_score.repetition_penalty * 100, 1),
            "weighted_total": round(turn_score.weighted_total * 100, 1),
            "rag_pending": turn_score.rag_pending
        },
        "claims": [
            {
                "id": c.claim_id,
                "text": c.text,
                "type": c.claim_type.value,
                "status": c.status.value,
                "verification": c.verification.verdict.value
            }
            for c in new_claims
        ],
        "relations": [
            {
                "relation": r.relation.value,
                "score": round(r.score, 3),
                "hint": r.reasoning_hint
            }
            for r in all_relations
        ],
        "feedback": turn_score.feedback_text,
        "debate_over": debate_over,
        "winner": winner,
        "winner_name": state.players[winner].name if winner else None,
        "scores": {
            "p1": {
                "name": state.players["p1"].name,
                "total": round(state.players["p1"].total_score, 1)
            },
            "p2": {
                "name": state.players["p2"].name,
                "total": round(state.players["p2"].total_score, 1)
            }
        },
        "final_analysis": final_summary,
        "next_player": next_player if not debate_over else None,
        "next_round": state.current_round if not debate_over else None
    }

    # Broadcast over WebSocket to all connected clients (PDF live interface)
    await manager.broadcast(req.match_id, result)

    return result


async def _verify_and_update(state: MatchState, claim: ClaimRecord, match_id: str):
    """
    Background task: run RAG verification and update claim in memory.
    Broadcasts updated evidence scores via WebSocket (PDF Section 14).
    """
    try:
        result = await verify_claim(claim.text, claim.claim_type.value)
        claim.verification = result

        # Update in state
        if claim.claim_id in state.claim_state:
            state.claim_state[claim.claim_id].verification = result
            await memory.save_match(state)

        # Broadcast evidence update
        await manager.broadcast(match_id, {
            "type": "evidence_update",
            "claim_id": claim.claim_id,
            "claim_text": claim.text[:80],
            "verdict": result.verdict.value,
            "confidence": result.confidence,
            "conflicting": result.conflicting_evidence,
            "sources": [
                {"url": s.url, "title": s.title, "quality": s.source_quality}
                for s in result.sources[:2]
            ]
        })

    except Exception as e:
        logger.error(f"[RAG background] Error for claim {claim.claim_id}: {e}")


@app.post("/api/match/{match_id}/reset")
async def reset_match(match_id: str):
    """
    Operator reset: clear match state while keeping models loaded.
    PDF Section 16: "key distinction between persistent process and temporary session."
    """
    await memory.reset_match(match_id)
    logger.info(f"[API] Match {match_id} reset by operator")
    await manager.broadcast(match_id, {"type": "match_reset", "match_id": match_id})
    return {"status": "reset", "match_id": match_id}


@app.get("/api/health")
async def health():
    """Health check endpoint for exhibition monitoring (PDF Section 24)."""
    return {
        "status": "ok",
        "active_matches": memory.get_active_match_count(),
        "timestamp": time.time()
    }


# ─── WebSocket endpoint ───────────────────────────────────────────────────────

@app.websocket("/ws/{match_id}")
async def websocket_endpoint(websocket: WebSocket, match_id: str):
    await manager.connect(match_id, websocket)
    try:
        # Send current state on connection
        state = await memory.get_match(match_id)
        if state:
            await websocket.send_json({
                "type": "connected",
                "match_id": match_id,
                "state": _serialize_state(state)
            })
        while True:
            # Keep connection alive; all events pushed from server side
            data = await websocket.receive_text()
            # Handle ping
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(match_id, websocket)


# ─── Helper ──────────────────────────────────────────────────────────────────

def _serialize_state(state: MatchState) -> dict:
    return {
        "match_id": state.match_id,
        "topic": state.topic,
        "phase": state.phase.value,
        "current_round": state.current_round,
        "max_rounds": state.max_rounds,
        "current_player": state.current_player_id,
        "players": {
            pid: {
                "name": p.name,
                "side": p.side.value,
                "total_score": round(p.total_score, 1)
            }
            for pid, p in state.players.items()
        },
        "turn_count": len(state.turn_log),
        "winner": state.winner
    }


# ─── Static files for frontend ────────────────────────────────────────────────
import os
import pathlib

frontend_path = pathlib.Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Never reload in exhibition mode (PDF Section 22)
        log_level="info"
    )

/**
 * Clash of Minds — Frontend Application
 * CSES ML Exhibition | NIT Warangal
 *
 * Implements:
 * - WebSocket live connection to FastAPI backend
 * - Voice input via Web Speech API + transcript confirm flow (PDF Section 5)
 * - Full multi-turn debate UI (PDF Section 4, 13)
 * - Live score updates including RAG evidence updates (PDF Section 14)
 * - Input validation, idempotency guards (PDF Section 15.3)
 * - Exhibition reliability: timer, error handling, operator reset (PDF Section 15)
 */

'use strict';

// ─── Config ───────────────────────────────────────────────────────────────────
const API_BASE = 'http://localhost:8000';
const WS_BASE = 'ws://localhost:8000';
const TURN_TIME = 120;  // seconds per turn (configurable, PDF Section 13)
const MAX_WORDS = 500;

// ─── State ────────────────────────────────────────────────────────────────────
let matchId = null;
let currentPlayer = 'p1';
let matchConfig = null;   // {p1_name, p2_name, topic, max_rounds}
let ws = null;
let timerInterval = null;
let timerSeconds = TURN_TIME;
let isSubmitting = false;
let turnCounter = 0;

// Player running totals per criterion (for bar display)
const playerCriteria = {
  p1: { rebuttal: 0, reasoning: 0, evidence: 0, relevance: 0, clarity: 0, stance: 0 },
  p2: { rebuttal: 0, reasoning: 0, evidence: 0, relevance: 0, clarity: 0, stance: 0 }
};

// Voice state
let recognition = null;
let isRecording = false;
let voiceTranscript = '';

// ─── DOM references ───────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const setupScreen = $('setup-screen');
const debateScreen = $('debate-screen');
const finalScreen = $('final-screen');

const startBtn = $('start-match-btn');
const submitBtn = $('submit-btn');
const voiceBtn = $('voice-btn');
const voiceConfirm = $('voice-confirm-btn');
const voiceCancel = $('voice-cancel-btn');
const textarea = $('argument-textarea');
const resetBtn = $('reset-btn');
const newMatchBtn = $('new-match-btn');
const wordCount = $('word-count');
const voicePreview = $('voice-preview');
const voicePreviewText = $('voice-preview-text');
const analyzingOverlay = $('analyzing-overlay');
const debateFeed = $('debate-feed');
const feedEmpty = $('feed-empty');
const connectionBanner = $('connection-banner');


// ─── Utility ─────────────────────────────────────────────────────────────────

function generateTurnId() {
  return 'turn_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
}

function countWords(text) {
  return text.trim() ? text.trim().split(/\s+/).length : 0;
}

function showScreen(screenId) {
  ['setup-screen', 'debate-screen', 'final-screen'].forEach(id => {
    const el = $(id);
    el.classList.toggle('active', el.id === screenId);
  });
  // Scroll to top
  window.scrollTo(0, 0);
}


// ─── Toast notifications ──────────────────────────────────────────────────────

function toast(message, type = 'info', duration = 4000) {
  const icons = { info: 'ℹ️', success: '✅', warning: '⚠️', error: '❌' };
  const container = $('toast-container');
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `<span class="toast-icon">${icons[type] || 'ℹ️'}</span><span>${message}</span>`;
  container.appendChild(el);
  setTimeout(() => {
    el.style.opacity = '0';
    el.style.transform = 'translateX(20px)';
    el.style.transition = '0.3s ease';
    setTimeout(() => el.remove(), 300);
  }, duration);
}


// ─── WebSocket connection ─────────────────────────────────────────────────────

function connectWebSocket(mId) {
  if (ws) ws.close();
  ws = new WebSocket(`${WS_BASE}/ws/${mId}`);

  ws.addEventListener('open', () => {
    connectionBanner.classList.remove('offline');
    updateServerStatus(true);
  });

  ws.addEventListener('message', e => {
    // Ignore plain-text keepalive frames (e.g. "pong")
    if (typeof e.data === 'string' && !e.data.startsWith('{')) return;
    try {
      const data = JSON.parse(e.data);
      handleServerMessage(data);
    } catch (err) {
      console.warn('WS parse error (ignored):', err);
    }
  });

  ws.addEventListener('close', () => {
    connectionBanner.classList.add('offline');
    updateServerStatus(false);
    // Auto-reconnect (PDF Section 15.4)
    setTimeout(() => {
      if (matchId) connectWebSocket(matchId);
    }, 3000);
  });

  ws.addEventListener('error', () => {
    connectionBanner.classList.add('offline');
    updateServerStatus(false);
  });

  // Keepalive ping
  setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send('ping');
    }
  }, 25000);
}


function handleServerMessage(data) {
  switch (data.type) {
    case 'connected':
      console.log('[WS] Connected to match:', data.match_id);
      break;

    case 'turn_result':
      handleTurnResult(data);
      break;

    case 'evidence_update':
      // RAG background update (PDF Section 14)
      handleEvidenceUpdate(data);
      break;

    case 'match_reset':
      toast('Match has been reset by the operator.', 'warning');
      goToSetup();
      break;

    default:
      console.log('[WS] Unknown message type:', data.type);
  }
}


// ─── Server status ────────────────────────────────────────────────────────────

async function checkServerStatus() {
  try {
    const r = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(3000) });
    if (r.ok) {
      updateServerStatus(true);
      return true;
    }
  } catch (_) { }
  updateServerStatus(false);
  return false;
}

function updateServerStatus(online) {
  const dot = $('server-status-dot');
  const text = $('server-status-text');
  if (online) {
    dot.style.background = 'var(--accent-green)';
    text.textContent = 'Backend Online';
  } else {
    dot.style.background = 'var(--accent-red)';
    text.textContent = 'Backend Offline';
  }
}


// ─── Match setup ──────────────────────────────────────────────────────────────

startBtn.addEventListener('click', async () => {
  const p1Name = $('p1-name').value.trim();
  const p2Name = $('p2-name').value.trim();
  const topic = $('topic-input').value.trim();
  const rounds = parseInt($('max-rounds').value);

  if (!p1Name || !p2Name) {
    toast('Please enter names for both players.', 'error');
    return;
  }
  if (!topic) {
    toast('Please enter a debate topic.', 'error');
    return;
  }

  startBtn.disabled = true;
  startBtn.textContent = 'Starting...';

  try {
    const res = await fetch(`${API_BASE}/api/match/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        p1_name: p1Name,
        p2_name: p2Name,
        topic,
        max_rounds: rounds
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to start match');
    }

    const data = await res.json();
    matchId = data.match_id;
    matchConfig = { p1Name, p2Name, topic, maxRounds: rounds };
    currentPlayer = 'p1';

    // Connect WebSocket
    connectWebSocket(matchId);

    // Initialize debate UI
    initDebateUI(data);
    showScreen('debate-screen');
    startTimer();

    toast(`Match started! ${p1Name} vs ${p2Name}`, 'success');

  } catch (err) {
    toast(`Error: ${err.message}`, 'error');
    console.error(err);
  } finally {
    startBtn.disabled = false;
    startBtn.innerHTML = '<span class="btn-icon">⚡</span> Start Debate';
  }
});


// ─── Debate UI initialization ─────────────────────────────────────────────────

function initDebateUI(data) {
  // Header
  $('header-topic').textContent = data.topic;
  $('p1-name-header').textContent = data.p1.name;
  $('p2-name-header').textContent = data.p2.name;
  $('p1-avatar-header').textContent = data.p1.name[0].toUpperCase();
  $('p2-avatar-header').textContent = data.p2.name[0].toUpperCase();
  $('p1-panel-name').textContent = data.p1.name;
  $('p1-score-header').textContent = '0.0';
  $('p2-score-header').textContent = '0.0';
  $('p1-total-score').textContent = '0.0';

  // Round dots
  buildRoundDots(data.max_rounds, 1);
  $('round-label').textContent = `Round 1 / ${data.max_rounds}`;

  // Turn indicator
  updateTurnIndicator('p1', data.p1.name, 'Opening argument — FOR the proposition');

  // Clear feed
  debateFeed.innerHTML = '';
  debateFeed.appendChild(feedEmpty);
  feedEmpty.style.display = '';

  // Clear turn histories
  $('p1-turn-history').innerHTML = '';

  // Clear analysis panel
  $('no-analysis-msg').style.display = '';
  $('criterion-section').style.display = 'none';
  $('feedback-section').style.display = 'none';
  $('claims-section').style.display = 'none';
  $('relations-section').style.display = 'none';
  $('factcheck-section').style.display = 'none';

  // Enable input
  enableInput();
  turnCounter = 0;
}


function buildRoundDots(maxRounds, currentRound) {
  const container = $('round-dots');
  container.innerHTML = '';
  for (let i = 1; i <= maxRounds; i++) {
    const dot = document.createElement('div');
    dot.className = 'round-dot' + (i === currentRound ? ' active' : (i < currentRound ? ' done' : ''));
    dot.setAttribute('aria-label', `Round ${i}${i === currentRound ? ' (current)' : ''}`);
    container.appendChild(dot);
  }
}


function updateTurnIndicator(playerId, playerName, instruction) {
  const avatar = $('current-avatar');
  avatar.className = `turn-avatar ${playerId}`;
  avatar.textContent = playerName[0].toUpperCase();
  $('current-player-name').textContent = playerName + "'s Turn";
  $('turn-instruction').textContent = instruction;
}


// ─── Argument submission ──────────────────────────────────────────────────────

submitBtn.addEventListener('click', async () => {
  if (isSubmitting) return;  // idempotency guard (PDF 15.3)

  const text = textarea.value.trim();
  if (!text) {
    toast('Please enter your argument.', 'warning');
    return;
  }
  if (countWords(text) < 5) {
    toast('Argument is too short (minimum 5 words).', 'warning');
    return;
  }

  isSubmitting = true;
  submitBtn.disabled = true;
  stopTimer();
  showAnalyzingOverlay();

  const turnId = generateTurnId();

  try {
    const res = await fetch(`${API_BASE}/api/match/submit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        match_id: matchId,
        player_id: currentPlayer,
        turn_id: turnId,
        text,
        input_method: voiceTranscript ? 'voice' : 'text'
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Submission failed');
    }

    // Response will also come via WebSocket; REST response is the same data
    const data = await res.json();
    // handleTurnResult called from WS; if WS failed use REST fallback
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      handleTurnResult(data);
    }

    // Clear textarea
    textarea.value = '';
    updateWordCount();
    voiceTranscript = '';
    hideVoicePreview();

  } catch (err) {
    toast(`Submission error: ${err.message}`, 'error');
    console.error(err);
  } finally {
    isSubmitting = false;
    hideAnalyzingOverlay();
  }
});


// ─── Turn result handler ──────────────────────────────────────────────────────

function handleTurnResult(data) {
  hideAnalyzingOverlay();
  isSubmitting = false;

  const pid = data.player_id;
  const pName = data.player_name;

  // Add argument bubble to feed
  addDebateBubble(pid, pName, data);

  // Update score bars and totals
  updatePlayerScores(pid, data.score);

  // Update analysis panel
  updateAnalysisPanel(data);

  // Update turn history
  addTurnHistoryEntry(pid, data);

  // Update round / header
  if (data.next_round) {
    buildRoundDots(matchConfig.maxRounds, data.next_round);
    $('round-label').textContent = `Round ${data.next_round} / ${matchConfig.maxRounds}`;
  }

  if (data.debate_over) {
    setTimeout(() => showFinalScreen(data), 1800);
    return;
  }

  // Advance to next player
  currentPlayer = data.next_player || (pid === 'p1' ? 'p2' : 'p1');
  const nextName = matchConfig[`${currentPlayer === 'p1' ? 'p1' : 'p2'}Name`];
  const round = data.next_round || 1;
  const side = currentPlayer === 'p1' ? 'FOR' : 'AGAINST';
  const instruction = round === 1
    ? `Opening argument — ${side} the proposition`
    : `Round ${round} rebuttal — respond to opponent's arguments`;

  updateTurnIndicator(currentPlayer, nextName, instruction);
  enableInput();
  startTimer();

  toast(`${nextName}'s turn (Round ${round})`, 'info', 2500);
}


// ─── Evidence update from background RAG ─────────────────────────────────────

function handleEvidenceUpdate(data) {
  // Update the fact-check display in the analysis panel
  const factList = $('fact-check-list');
  if (!factList) return;

  // Find existing entry for this claim or add a new one
  const existingId = `fact-claim-${data.claim_id}`;
  let el = document.getElementById(existingId);

  const verdictClass = data.verdict.toLowerCase();
  const verdictEmoji = {
    supported: '✅', partially_supported: '🟡', refuted: '❌',
    insufficient_evidence: '❓', ambiguous: '⚠️', pending: '🔄'
  }[verdictClass] || '❓';

  const html = `
    <div class="fact-verdict">${verdictEmoji} ${data.verdict.replace(/_/g, ' ')}</div>
    <div class="fact-text">${escapeHtml(data.claim_text)}${data.conflicting ? ' <em>(conflicting sources)</em>' : ''}</div>
  `;

  if (el) {
    el.className = `fact-item ${verdictClass}`;
    el.innerHTML = html;
  } else {
    el = document.createElement('div');
    el.id = existingId;
    el.className = `fact-item ${verdictClass}`;
    el.innerHTML = html;
    factList.appendChild(el);
    $('factcheck-section').style.display = '';
  }

  // Update any pending chips on debate bubbles
  const pendingChip = document.querySelector(`[data-claim-id="${data.claim_id}"].rag-pending-chip`);
  if (pendingChip && data.verdict !== 'PENDING') {
    pendingChip.remove();
    toast(`Fact-check updated: ${data.verdict.replace(/_/g, ' ')} for "${data.claim_text.slice(0, 40)}..."`, 'info');
  }
}


// ─── Debate feed bubbles ──────────────────────────────────────────────────────

function addDebateBubble(playerId, playerName, data) {
  if (feedEmpty && feedEmpty.parentNode) feedEmpty.remove();

  const bubble = document.createElement('div');
  bubble.className = `argument-bubble ${playerId}`;
  bubble.setAttribute('role', 'article');
  bubble.setAttribute('aria-label', `${playerName}'s argument`);

  const score = data.score;
  const ragPending = score.rag_pending;

  // Claim type chips
  const claimChipsHtml = (data.claims || []).map(c =>
    `<span class="claim-chip ${c.type}" title="${c.text.slice(0, 60)}...">${c.type}</span>`
  ).join('');

  // Relation brief
  const relSummary = data.relations && data.relations.length > 0
    ? `<span style="font-size:11px; color:var(--text-muted);">Relations: ${data.relations.map(r => r.relation).join(', ')}</span>`
    : '';

  bubble.innerHTML = `
    <div class="bubble-header">
      <span>${playerName}</span>
      <span style="opacity:0.5">·</span>
      <span>${playerId === 'p1' ? '🟦 FOR' : '🟥 AGAINST'}</span>
      <span style="opacity:0.5; margin-left:auto">Round ${data.round_number}</span>
    </div>
    <div class="bubble-text">${escapeHtml(textarea.value || data.text || '').replace(/\n/g, '<br/>')}</div>
    ${claimChipsHtml ? `<div class="claim-chips" style="margin-top:8px">${claimChipsHtml}</div>` : ''}
    <div class="bubble-score">
      <span class="bubble-score-num">${score.weighted_total.toFixed(1)}</span>
      <span style="font-size:11px; color:var(--text-muted)">/ 100</span>
      ${ragPending ? `<span class="rag-pending-chip" data-claim-id="pending">🔍 Fact-check pending</span>` : ''}
      ${relSummary}
    </div>
  `;

  // Store the text before clearing
  bubble.querySelector('.bubble-text').textContent =
    data.turn_log_text || textarea.value || '';

  debateFeed.appendChild(bubble);
  debateFeed.scrollTop = debateFeed.scrollHeight;
}


function addTurnHistoryEntry(playerId, data) {
  const histId = playerId === 'p1' ? 'p1-turn-history' : 'p1-turn-history';
  // Both histories in respective panels (p2 panel has its own turn history)
  const hist = $(playerId + '-turn-history') || $('p1-turn-history');

  const entry = document.createElement('div');
  entry.className = `turn-entry ${playerId}`;
  entry.innerHTML = `
    <div class="turn-entry-score">Round ${data.round_number} · ${data.score.weighted_total.toFixed(1)} pts</div>
    <div>${data.feedback ? data.feedback.slice(0, 80) + '...' : 'Turn evaluated.'}</div>
  `;
  if (hist) hist.appendChild(entry);
}


// ─── Analysis panel updates ───────────────────────────────────────────────────

function updatePlayerScores(playerId, score) {
  const prefix = playerId;

  // Running averages (average of last turn for bars; total for scoreboard)
  const crit = playerCriteria[playerId];
  const turns = playerId === 'p1'
    ? ($('p1-turn-history').children.length + 1)
    : ($('p1-turn-history').children.length);  // simplified

  // Update bars (use last turn value for immediate feedback)
  const fields = [
    ['rebuttal', score.rebuttal],
    ['reasoning', score.logical_reasoning],
    ['evidence', score.evidence_quality],
    ['relevance', score.topic_relevance],
    ['clarity', score.clarity],
    ['stance', score.stance_consistency]
  ];

  fields.forEach(([key, val]) => {
    const bar = $(`${prefix}-${key}-bar`);
    const valEl = $(`${prefix}-${key}-val`);
    if (bar) bar.style.width = val + '%';
    if (valEl) valEl.textContent = val.toFixed(1);
  });

  // Update total score
  const total = score.weighted_total;
  $(`${prefix}-total-score`) && ($(`${prefix}-total-score`).textContent = total.toFixed(1));
  $(`${prefix}-score-header`) && ($(`${prefix}-score-header`).textContent = total.toFixed(1));
}


function updateAnalysisPanel(data) {
  const score = data.score;
  $('no-analysis-msg').style.display = 'none';

  // Criterion breakdown
  $('criterion-section').style.display = '';
  const criteriaMap = {
    'cr-rebuttal': score.rebuttal,
    'cr-reasoning': score.logical_reasoning,
    'cr-evidence': score.evidence_quality,
    'cr-relevance': score.topic_relevance,
    'cr-clarity': score.clarity,
    'cr-stance': score.stance_consistency,
  };
  const barMap = {
    'crb-rebuttal': score.rebuttal,
    'crb-reasoning': score.logical_reasoning,
    'crb-evidence': score.evidence_quality,
    'crb-relevance': score.topic_relevance,
    'crb-clarity': score.clarity,
    'crb-stance': score.stance_consistency,
  };
  Object.entries(criteriaMap).forEach(([id, val]) => {
    const el = $(id);
    if (el) el.textContent = val ? val.toFixed(1) + '%' : '—';
  });
  Object.entries(barMap).forEach(([id, val]) => {
    const el = $(id);
    if (el) el.style.width = (val || 0) + '%';
  });

  // Feedback
  if (data.feedback) {
    $('feedback-section').style.display = '';
    $('feedback-text').textContent = data.feedback;
  }

  // Claim chips
  if (data.claims && data.claims.length > 0) {
    $('claims-section').style.display = '';
    $('claims-chips').innerHTML = data.claims.map(c =>
      `<span class="claim-chip ${c.type}" title="${escapeHtml(c.text.slice(0, 80))}">${c.type}</span>`
    ).join('');
  }

  // Relations (NLI)
  if (data.relations && data.relations.length > 0) {
    $('relations-section').style.display = '';
    $('relation-list').innerHTML = data.relations.map(r => `
      <div class="relation-tag" title="${escapeHtml(r.hint)}">
        <span class="relation-label ${r.relation}">${r.relation}</span>
        <span class="relation-hint">${escapeHtml(r.hint || '')}</span>
        <span style="font-size:10px; color:var(--text-muted);">${(r.score * 100).toFixed(0)}%</span>
      </div>
    `).join('');
  }

  // Fact-check (initial state — will be updated by RAG events)
  const checkableClaims = (data.claims || []).filter(c =>
    c.type === 'factual' || c.type === 'statistical'
  );
  if (checkableClaims.length > 0) {
    $('factcheck-section').style.display = '';
    const factList = $('fact-check-list');
    factList.innerHTML = checkableClaims.map(c => `
      <div class="fact-item ${c.verification === 'PENDING' ? 'pending' : c.verification.toLowerCase()}"
           id="fact-claim-${c.id}">
        <div class="fact-verdict">🔄 ${c.verification.replace(/_/g, ' ')}</div>
        <div class="fact-text" style="word-wrap: break-word; white-space: normal;">${escapeHtml(c.text)}</div>
      </div>
    `).join('');
  }
}



function showFinalScreen(data) {
  const winner = data.winner;
  const winnerName = data.winner_name;
  const p1Score = data.scores.p1.total;
  const p2Score = data.scores.p2.total;

  // Winner display
  $('winner-name-display').textContent = winnerName;
  $('winner-name-display').className = 'winner-name ' + (winner === 'p1' ? 'p1-win' : 'p2-win');
  $('winner-side-display').textContent = winner === 'p1' ? 'FOR the proposition' : 'AGAINST the proposition';

  // Final scoreboard
  $('p1-name-final').textContent = matchConfig.p1Name;
  $('p2-name-final').textContent = matchConfig.p2Name;
  $('p1-score-final').textContent = p1Score.toFixed(1);
  $('p2-score-final').textContent = p2Score.toFixed(1);
  $('final-topic').textContent = matchConfig.topic;

  // Crown & glow
  if (winner === 'p1') {
    $('p1-final-card').classList.add('winner');
    $('p1-crown').style.display = '';
  } else {
    $('p2-final-card').classList.add('winner', 'p2-win');
    $('p2-crown').style.display = '';
  }

  // ── Final Analysis Cards ──────────────────────────────────────────────────
  const analysis = data.final_analysis;
  if (analysis && analysis.winner_strengths) {
    // Remove any previous analysis
    const old = document.getElementById('final-analysis-section');
    if (old) old.remove();

    const container = document.querySelector('#final-screen > div');

    const section = document.createElement('div');
    section.id = 'final-analysis-section';
    section.style.cssText = 'width:100%;max-width:700px;margin:0 auto 2rem;display:flex;flex-direction:column;gap:1.2rem;';

    // ── Winner Strengths ──
    const strengthsHtml = analysis.winner_strengths.map(s => `
      <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:rgba(16,185,129,0.06);border-radius:8px;border-left:3px solid var(--accent-green);">
        <div>
          <div style="font-weight:700;font-size:0.9rem;color:var(--accent-green);">${s.category}</div>
          <div style="font-size:0.82rem;color:var(--text-secondary);margin-top:2px;">${s.description}</div>
        </div>
        <div style="text-align:right;min-width:90px;margin-left:12px;">
          <span style="font-size:1.1rem;font-weight:800;color:var(--accent-green);">${s.winner_score}</span>
          <span style="font-size:0.75rem;color:var(--text-muted);"> vs ${s.loser_score}</span>
        </div>
      </div>`).join('');

    // ── Loser Mistakes ──
    const mistakesHtml = analysis.loser_mistakes.map(m => `
      <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:rgba(244,63,94,0.06);border-radius:8px;border-left:3px solid var(--accent-red);">
        <div style="flex:1;">
          <div style="font-weight:700;font-size:0.9rem;color:var(--accent-red);">${m.category}${m.score > 0 ? ` — ${m.score} pts` : ''}</div>
          <div style="font-size:0.82rem;color:var(--text-secondary);margin-top:2px;">${m.detail}</div>
        </div>
        ${m.gap > 0 ? `<div style="min-width:70px;text-align:right;margin-left:12px;font-size:0.82rem;color:var(--accent-red);">-${m.gap} pts gap</div>` : ''}
      </div>`).join('');

    section.innerHTML = `
      <div style="padding:1.2rem;background:rgba(255,255,255,0.03);border-radius:14px;border:1px solid rgba(255,255,255,0.08);">
        <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:.1em;color:var(--accent-green);margin-bottom:12px;font-weight:700;">
          🏆 Why ${analysis.winner_name} Won
        </div>
        <div style="display:flex;flex-direction:column;gap:8px;">${strengthsHtml}</div>
      </div>
      <div style="padding:1.2rem;background:rgba(255,255,255,0.03);border-radius:14px;border:1px solid rgba(255,255,255,0.08);">
        <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:.1em;color:var(--accent-red);margin-bottom:12px;font-weight:700;">
          📉 Where ${analysis.loser_name} Fell Short
        </div>
        <div style="display:flex;flex-direction:column;gap:8px;">${mistakesHtml}</div>
      </div>`;

    // Insert before the New Match button div
    const actionsDiv = container.querySelector('.final-actions');
    if (actionsDiv) {
      container.insertBefore(section, actionsDiv);
    } else {
      container.appendChild(section);
    }
  }

  showScreen('final-screen');
  launchConfetti(winner);
  toast(`🏆 ${winnerName} wins the debate!`, 'success', 6000);
}


// ─── Timer ────────────────────────────────────────────────────────────────────

function startTimer() {
  timerSeconds = TURN_TIME;
  updateTimerDisplay();

  timerInterval = setInterval(() => {
    timerSeconds--;
    updateTimerDisplay();
    if (timerSeconds <= 0) {
      stopTimer();
      toast('⏱ Time is up! Please submit your argument.', 'warning');
    }
  }, 1000);
}

function stopTimer() {
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
}

function updateTimerDisplay() {
  const min = Math.floor(timerSeconds / 60);
  const sec = timerSeconds % 60;
  $('timer-display').textContent = `${min}:${sec.toString().padStart(2, '0')}`;

  // Update SVG ring
  const circle = $('timer-progress-circle');
  if (circle) {
    const radius = 22;
    const circumference = 2 * Math.PI * radius; // ~138.2
    const fraction = timerSeconds / TURN_TIME;
    const offset = circumference * (1 - fraction);
    circle.style.strokeDashoffset = offset;

    // Color shift when time is low
    if (timerSeconds <= 20) {
      circle.style.stroke = 'var(--accent-red)';
      $('timer-display').style.color = 'var(--accent-red)';
    } else if (timerSeconds <= 45) {
      circle.style.stroke = 'var(--accent-amber)';
      $('timer-display').style.color = 'var(--accent-amber)';
    } else {
      circle.style.stroke = 'var(--accent-cyan)';
      $('timer-display').style.color = 'var(--accent-cyan)';
    }
  }
}


// ─── Input helpers ────────────────────────────────────────────────────────────

function enableInput() {
  const area = $('input-area');
  area.classList.remove('disabled');
  textarea.disabled = false;
  textarea.focus();
  submitBtn.disabled = false;
}

function disableInput() {
  const area = $('input-area');
  area.classList.add('disabled');
  textarea.disabled = true;
  submitBtn.disabled = true;
}

textarea.addEventListener('input', updateWordCount);

function updateWordCount() {
  const words = countWords(textarea.value);
  wordCount.textContent = `${words} / ${MAX_WORDS} words`;
  wordCount.style.color = words > MAX_WORDS * 0.9
    ? 'var(--accent-amber)'
    : 'var(--text-muted)';

  submitBtn.disabled = textarea.value.trim().length === 0;
}


// ─── Voice Input (PDF Section 5) ─────────────────────────────────────────────

function initVoiceRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    voiceBtn.style.display = 'none';
    return;
  }

  recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = 'en-US';  // English enforced (PDF Section 5.4)
  recognition.maxAlternatives = 1;

  let finalTranscript = '';
  let interimTranscript = '';

  recognition.addEventListener('result', e => {
    interimTranscript = '';
    finalTranscript = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      if (e.results[i].isFinal) {
        finalTranscript += e.results[i][0].transcript;
      } else {
        interimTranscript += e.results[i][0].transcript;
      }
    }
    // Show interim in textarea
    textarea.value = finalTranscript + interimTranscript;
    updateWordCount();
  });

  recognition.addEventListener('end', () => {
    if (isRecording) {
      // Stopped naturally — show transcript confirmation (PDF Section 5.1)
      voiceTranscript = textarea.value;
      if (voiceTranscript.trim()) {
        showVoiceConfirmation(voiceTranscript);
      }
      setVoiceState(false);
    }
  });

  recognition.addEventListener('error', e => {
    console.error('[Voice] Error:', e.error);
    setVoiceState(false);

    if (e.error === 'not-allowed') {
      toast('Microphone access denied. Please use text input.', 'error');
      voiceBtn.style.display = 'none';
    } else if (e.error === 'network') {
      toast('Network error with speech recognition. Using text input.', 'warning');
    } else {
      toast('Voice input failed. Please type your argument.', 'warning');
    }
  });
}

voiceBtn.addEventListener('click', () => {
  if (!recognition) {
    toast('Voice input not supported in this browser.', 'warning');
    return;
  }

  if (!isRecording) {
    // Start recording
    textarea.value = '';
    voiceTranscript = '';
    hideVoicePreview();
    try {
      recognition.start();
      setVoiceState(true);
      toast('Recording... speak your argument.', 'info', 2000);
    } catch (e) {
      toast('Could not start microphone: ' + e.message, 'error');
    }
  } else {
    // Stop recording manually
    recognition.stop();
    setVoiceState(false);
  }
});

/**
 * Voice transcript confirmation (PDF Section 5.1):
 * "Never silently score a transcript the participant cannot inspect."
 * Show transcript, allow correction, only judge confirmed version.
 */
function showVoiceConfirmation(transcript) {
  voicePreviewText.textContent = transcript;
  voicePreview.classList.add('visible');
  voiceConfirm.style.display = '';
  voiceCancel.style.display = '';
  textarea.value = transcript;
  textarea.focus();
  toast('🎙️ Transcript ready — review and confirm before submitting.', 'info', 5000);
}

function hideVoicePreview() {
  voicePreview.classList.remove('visible');
  voiceConfirm.style.display = 'none';
  voiceCancel.style.display = 'none';
}

voiceConfirm.addEventListener('click', () => {
  // User confirmed — transcript is already in textarea, ready to submit
  hideVoicePreview();
  toast('Transcript confirmed. Submit when ready.', 'success', 2000);
});

voiceCancel.addEventListener('click', () => {
  // Discard (PDF Section 5.1: participant controls the transcript)
  textarea.value = '';
  voiceTranscript = '';
  hideVoicePreview();
  updateWordCount();
  toast('Transcript discarded. Speak again or type your argument.', 'info', 3000);
});

function setVoiceState(recording) {
  isRecording = recording;
  const icon = $('voice-btn-icon');
  const text = $('voice-btn-text');

  if (recording) {
    voiceBtn.classList.add('recording');
    icon.textContent = '⏹️';
    text.textContent = 'Stop';
    voiceBtn.setAttribute('aria-label', 'Stop recording');
  } else {
    voiceBtn.classList.remove('recording');
    icon.textContent = '🎙️';
    text.textContent = 'Speak';
    voiceBtn.setAttribute('aria-label', 'Start voice input');
  }
}


// ─── Analyzing overlay ────────────────────────────────────────────────────────

const PIPELINE_STEPS = [
  { id: 'step-segment', label: 'Segmenting claims...', ms: 300 },
  { id: 'step-embed', label: 'Encoding semantics...', ms: 500 },
  { id: 'step-retrieve', label: 'Retrieving opponent points...', ms: 400 },
  { id: 'step-nli', label: 'Running NLI analysis...', ms: 600 },
  { id: 'step-score', label: 'Computing score...', ms: 400 },
  { id: 'step-rag', label: 'Fact-checking (background)...', ms: 200 },
];

function showAnalyzingOverlay() {
  analyzingOverlay.classList.add('visible');
  // Animate steps in sequence
  PIPELINE_STEPS.forEach((step, i) => {
    setTimeout(() => {
      const el = $(step.id);
      if (el) {
        // Mark previous as done
        if (i > 0) {
          const prev = $(PIPELINE_STEPS[i - 1].id);
          if (prev) { prev.classList.remove('active'); prev.classList.add('done'); }
        }
        el.classList.add('active');
      }
    }, PIPELINE_STEPS.slice(0, i).reduce((acc, s) => acc + s.ms, 0));
  });
}

function hideAnalyzingOverlay() {
  analyzingOverlay.classList.remove('visible');
  PIPELINE_STEPS.forEach(step => {
    const el = $(step.id);
    if (el) { el.classList.remove('active', 'done'); }
  });
}


// ─── Reset ────────────────────────────────────────────────────────────────────

resetBtn.addEventListener('click', async () => {
  if (!matchId) return;
  if (!confirm('Reset the current match? This will clear all debate state.')) return;

  try {
    await fetch(`${API_BASE}/api/match/${matchId}/reset`, { method: 'POST' });
  } catch (e) { }

  stopTimer();
  goToSetup();
  toast('Match reset.', 'info');
});

newMatchBtn.addEventListener('click', () => {
  goToSetup();
});

function goToSetup() {
  stopTimer();
  matchId = null;
  matchConfig = null;
  currentPlayer = 'p1';
  isSubmitting = false;
  voiceTranscript = '';

  // Reset final screen cards
  $('p1-final-card').classList.remove('winner');
  $('p2-final-card').classList.remove('winner', 'p2-win');
  $('p1-crown').style.display = 'none';
  $('p2-crown').style.display = 'none';

  showScreen('setup-screen');
}


// ─── Confetti celebration ─────────────────────────────────────────────────────

function launchConfetti(winner) {
  const canvas = $('confetti-canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  const color = winner === 'p1' ? '#3b82f6' : '#f43f5e';
  const colors = [color, '#ffffff', '#8b5cf6', '#10b981', '#f59e0b'];

  const particles = Array.from({ length: 120 }, () => ({
    x: Math.random() * canvas.width,
    y: -10,
    r: Math.random() * 6 + 2,
    d: Math.random() * 20,
    color: colors[Math.floor(Math.random() * colors.length)],
    tilt: Math.random() * 10 - 10,
    tiltAngle: 0,
    tiltAngleIncr: 0.07 + Math.random() * 0.05,
    speed: Math.random() * 3 + 2
  }));

  let frame = 0;
  const maxFrames = 150;

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(p => {
      ctx.beginPath();
      ctx.lineWidth = p.r / 2;
      ctx.strokeStyle = p.color;
      ctx.moveTo(p.x + p.tilt + p.r / 4, p.y);
      ctx.lineTo(p.x + p.tilt, p.y + p.tilt + p.r / 4);
      ctx.stroke();

      p.y += p.speed;
      p.tiltAngle += p.tiltAngleIncr;
      p.tilt = Math.sin(p.tiltAngle) * 15;

      // Respawn
      if (p.y > canvas.height + 10) {
        p.x = Math.random() * canvas.width;
        p.y = -10;
      }
    });

    frame++;
    if (frame < maxFrames) {
      requestAnimationFrame(draw);
    } else {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }
  draw();
}


// ─── Utility ─────────────────────────────────────────────────────────────────

function handleEvidenceUpdate(data) {
  const factItem = document.getElementById(`fact-claim-${data.claim_id}`);
  if (factItem) {
    factItem.className = `fact-item ${data.verdict.toLowerCase()}`;
    const verdictLabel = data.verdict.replace(/_/g, ' ');
    const icon = data.verdict === 'VERIFIED' ? '✅' : (data.verdict === 'INSUFFICIENT_EVIDENCE' ? '❓' : '❌');
    factItem.querySelector('.fact-verdict').innerHTML = `${icon} ${verdictLabel}`;
  }

  document.querySelectorAll('.rag-pending-chip').forEach(badge => badge.remove());

  if (data.score) {
    updatePlayerScores(data.player_id, data.score);
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}


// ─── Init ─────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  initVoiceRecognition();
  checkServerStatus();

  // Periodic health check (exhibition reliability — PDF Section 24)
  setInterval(checkServerStatus, 30000);

  // Enter key to submit
  textarea.addEventListener('keydown', e => {
    if (e.key === 'Enter' && e.ctrlKey) {
      e.preventDefault();
      submitBtn.click();
    }
  });

  // Window resize for confetti canvas
  window.addEventListener('resize', () => {
    const canvas = $('confetti-canvas');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  });
});

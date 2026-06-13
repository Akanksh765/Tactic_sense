import os
# 1. Kill HuggingFace Tokenizer & Progress Bar Threads
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# 2. Kill PyTorch/Intel CPU Multithreading (The Segfault Fix)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import hashlib
import io
import re
import tempfile
from html import escape
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
# On Windows, print() in background threads can crash with cp1252 UnicodeEncodeError.
# Safest fix: override PYTHONUTF8 env var for child procs, then try reconfigure.
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("STREAMLIT_SERVER_PORT", "8080")
os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")
os.environ.setdefault("STREAMLIT_SERVER_ENABLE_CORS", "false")
os.environ.setdefault("STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION", "false")
try:
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass  # Streamlit wraps stdout; if it doesn't support reconfigure, ignore

import streamlit as st
import torch
torch.set_num_threads(1)
import queue
import time
import threading
import speech_recognition as sr
try:
    import pyaudio
except ImportError:
    pyaudio = None
try:
    import whisper
except ImportError:
    whisper = None
from transformers.models.xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer

try:
    from src.model import TacticSenseModel
    from src.predict import predict_tactic, get_final_verdict
    from src.llm_coach import get_llm_countermeasure as _llm_coach_fn
except ImportError:
    from model import TacticSenseModel
    from predict import predict_tactic, get_final_verdict
    try:
        from src.llm_coach import get_llm_countermeasure as _llm_coach_fn
    except ImportError:
        from llm_coach import get_llm_countermeasure as _llm_coach_fn

def _get_secret_value(name):
    try:
        return st.secrets.get(name)
    except Exception:
        return None

_openai_api_key = os.getenv("OPENAI_API_KEY") or _get_secret_value("OPENAI_API_KEY")

try:
    openai_client = OpenAI(api_key=_openai_api_key, timeout=6.0) if OpenAI and _openai_api_key else None
except Exception:
    openai_client = None

def _sanitize_app_llm_response(advice: str) -> str:
    if advice is None:
        return ""
    advice = advice.strip()
    if re.search(r"[\[\]]", advice):
        replacements = {
            "[date]": "by this evening",
            "[time]": "within the next hour",
            "[price]": "the agreed amount",
            "[amount]": "the amount on the table",
            "[buyer name]": "the buyer",
            "[seller]": "the seller",
            "[insert text]": "the right wording",
            "[insert text here]": "the right wording",
        }
        def _replace(match):
            token = match.group(0).lower()
            return replacements.get(token, "")
        advice = re.sub(
            r"\[(?:date|time|price|amount|buyer name|seller|insert text here|insert text)\]",
            _replace,
            advice,
            flags=re.IGNORECASE,
        ).replace("[", "").replace("]", "")
    lines = [line.strip() for line in advice.splitlines() if line.strip()]
    return "\n".join(line if line.startswith("- ") else f"- {line}" for line in lines)


def get_llm_countermeasure(transcript, detected_tactic):
    if openai_client is None:
        return "AI coach unavailable: set OPENAI_API_KEY to enable counter-strategies."
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a live negotiation coach. Respond only with 1-3 concise Markdown "
                        "bullet points, no preamble, no filler, and never output bracketed placeholders "
                        "or template tokens like [date], [price], [buyer name], or [insert text]. "
                        "If you need a deadline, use natural relative phrasing such as 'by tonight' "
                        "or 'within the next hour'. If you need a price anchor, use actual figures "
                        "from the transcript instead of placeholders."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Transcript:\n{transcript}\n\n"
                        f"Detected tactic: {detected_tactic}\n\n"
                        "Counter-strategy:"
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=50,
        )
        return _sanitize_app_llm_response(response.choices[0].message.content.strip())
    except Exception:
        return "Pause, ask a clarifying question, and anchor your response to objective value."

# Ã¢â€â‚¬Ã¢â€â‚¬ Page config Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
st.set_page_config(page_title="TacticSense Live", page_icon="TS", layout="wide")

# Ã¢â€â‚¬Ã¢â€â‚¬ Process-level singletons (thread-safe, survive reruns) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
@st.cache_resource
def _global_queue():
    return queue.Queue()

@st.cache_resource
def _listener():
    return {
        "stop_fn": None,
        "active": False,
        "starting": False,
        "error": None,
        "active_speaker_id": 0,
    }

_q   = _global_queue()
_lst = _listener()

# Ã¢â€â‚¬Ã¢â€â‚¬ CSS Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
*, html, body { font-family: 'Inter', sans-serif; }
.stApp { background: #0b0f1e; color: #e2e8f0; }

/* Title */
.ts-title {
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(90deg,#60a5fa,#a78bfa,#34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.ts-sub { color:#475569; font-size:.9rem; letter-spacing:.08em; margin-bottom:1.5rem; }

/* Cards */
.card {
    background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
    border-radius:12px; padding:1rem; margin-bottom:.8rem;
}
.metric-val { font-size:1.8rem; font-weight:700; color:#60a5fa; font-family:'JetBrains Mono',monospace; }
.metric-lbl { font-size:.7rem; color:#64748b; letter-spacing:.1em; text-transform:uppercase; }

/* Verdict */
.verdict {
    background:linear-gradient(135deg,rgba(96,165,250,.12),rgba(167,139,250,.08));
    border:1px solid rgba(96,165,250,.25); border-radius:10px;
    padding:1rem; margin-bottom:1rem;
    font-family:'JetBrains Mono',monospace; font-size:.82rem; color:#93c5fd;
}

/* Tactic row */
.t-row { display:flex; justify-content:space-between; align-items:center;
         margin-bottom:.25rem; font-size:.8rem; }
.t-name { color:#cbd5e1; }
.t-score { font-family:'JetBrains Mono',monospace; font-weight:600; }

/* Status pills */
.pill-active {
    display:inline-block; background:rgba(52,211,153,.15);
    border:1px solid #34d399; color:#34d399;
    padding:2px 10px; border-radius:20px; font-size:.75rem;
    font-weight:600; letter-spacing:.08em; animation:blink 2s infinite;
}
.pill-starting {
    display:inline-block; background:rgba(251,191,36,.12);
    border:1px solid #fbbf24; color:#fbbf24;
    padding:2px 10px; border-radius:20px; font-size:.75rem; font-weight:600;
}
.pill-off {
    display:inline-block; background:rgba(239,68,68,.1);
    border:1px solid rgba(239,68,68,.4); color:#f87171;
    padding:2px 10px; border-radius:20px; font-size:.75rem; font-weight:600;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.5} }

/* Sidebar */
section[data-testid="stSidebar"] {
    background:rgba(8,12,24,.97) !important;
    border-right:1px solid rgba(255,255,255,.06);
}

/* Buttons */
.stButton > button {
    background:linear-gradient(135deg,#3b82f6,#6366f1) !important;
    color:#fff !important; border:none !important; border-radius:8px !important;
    font-weight:600 !important; width:100%; transition:all .2s !important;
}
.stButton > button:hover { box-shadow:0 4px 16px rgba(99,102,241,.45) !important; }
.stButton > button:disabled { background:#1e293b !important; color:#475569 !important; }

/* Progress */
.stProgress > div > div > div > div {
    background:linear-gradient(90deg,#3b82f6,#8b5cf6) !important; border-radius:4px;
}
.stProgress > div > div > div { background:rgba(255,255,255,.06) !important; border-radius:4px; }

div[data-testid="stExpander"] {
    background:rgba(255,255,255,.02) !important;
    border:1px solid rgba(255,255,255,.06) !important; border-radius:8px !important;
}

/* Tactical analysis right panel */
.analysis-panel {
    background:
        radial-gradient(circle at 12% 0%, rgba(139,92,246,.16), transparent 28%),
        linear-gradient(180deg, rgba(17,24,39,.96), rgba(8,12,22,.98));
    border:1px solid rgba(148,163,184,.18);
    border-radius:18px;
    box-shadow:0 24px 70px rgba(0,0,0,.38), inset 0 1px 0 rgba(255,255,255,.04);
    color:#F8FAFC;
    padding:22px;
}
.analysis-header, .section-header, .metric-grid, .verdict-card,
.dimension-row, .chart-toolbar {
    display:flex;
    align-items:center;
}
.analysis-header, .section-header, .chart-toolbar {
    justify-content:space-between;
}
.analysis-title {
    color:#fff;
    font-size:1.18rem;
    font-weight:750;
    letter-spacing:.01em;
}
.live-badge {
    display:inline-flex;
    align-items:center;
    gap:6px;
    color:#86efac;
    background:rgba(34,197,94,.14);
    border:1px solid rgba(74,222,128,.26);
    border-radius:999px;
    padding:5px 11px;
    font-size:.76rem;
    font-weight:700;
    box-shadow:0 0 18px rgba(34,197,94,.16);
    animation:livePulse 3.6s ease-in-out infinite;
}
.live-dot {
    width:7px;
    height:7px;
    border-radius:50%;
    background:#22c55e;
    box-shadow:0 0 12px #22c55e;
    animation:liveDotPulse 3.6s ease-in-out infinite;
}
@keyframes livePulse {
    0%,100% { opacity:.78; box-shadow:0 0 10px rgba(34,197,94,.08); }
    50% { opacity:1; box-shadow:0 0 24px rgba(34,197,94,.26); }
}
@keyframes liveDotPulse {
    0%,100% { opacity:.52; transform:scale(.88); }
    50% { opacity:1; transform:scale(1.18); }
}
.metric-grid {
    gap:14px;
    margin-top:18px;
}
.metric-card {
    flex:1;
    min-height:104px;
    background:linear-gradient(180deg, rgba(17,24,39,.92), rgba(13,18,31,.96));
    border:1px solid rgba(148,163,184,.14);
    border-radius:14px;
    padding:16px;
    box-shadow:inset 0 1px 0 rgba(255,255,255,.04), 0 14px 34px rgba(0,0,0,.25);
}
.metric-icon {
    width:30px;
    height:30px;
    display:grid;
    place-items:center;
    color:#A78BFA;
    background:rgba(139,92,246,.13);
    border:1px solid rgba(167,139,250,.22);
    border-radius:9px;
    margin-bottom:12px;
}
.metric-value {
    color:#fff;
    font-family:'JetBrains Mono', monospace;
    font-size:1.7rem;
    font-weight:750;
    line-height:1;
}
.metric-label {
    margin-top:7px;
    color:#7B8496;
    font-size:.78rem;
    font-weight:600;
}
.verdict-card {
    gap:14px;
    margin-top:16px;
    padding:18px;
    background:linear-gradient(135deg, rgba(69,39,123,.72), rgba(32,24,63,.88));
    border:1px solid rgba(167,139,250,.32);
    border-radius:16px;
    box-shadow:0 0 35px rgba(139,92,246,.15), inset 0 1px 0 rgba(255,255,255,.06);
}
.shield-icon {
    width:44px;
    height:44px;
    flex:0 0 44px;
    display:grid;
    place-items:center;
    color:#C4B5FD;
    background:rgba(167,139,250,.14);
    border:1px solid rgba(196,181,253,.32);
    border-radius:14px;
    box-shadow:0 0 24px rgba(167,139,250,.35);
}
.verdict-copy {
    flex:1;
    min-width:0;
}
.eyebrow {
    color:#A6AEC0;
    font-size:.72rem;
    font-weight:700;
    letter-spacing:.08em;
    text-transform:uppercase;
}
.verdict-text {
    color:#fff;
    font-size:.95rem;
    font-weight:700;
    line-height:1.3;
    margin-top:4px;
}
.score-pill {
    color:#fff;
    border:1px solid rgba(255,255,255,.72);
    border-radius:999px;
    padding:7px 12px;
    font-family:'JetBrains Mono', monospace;
    font-size:.86rem;
    font-weight:700;
    box-shadow:0 0 18px rgba(255,255,255,.12);
}
.verdict-card.is-empty {
    background:linear-gradient(135deg, rgba(31,41,55,.74), rgba(15,23,42,.9));
    border-color:rgba(148,163,184,.16);
    box-shadow:inset 0 1px 0 rgba(255,255,255,.035);
}
.verdict-card.is-empty .shield-icon {
    opacity:.24;
    color:#94A3B8;
    background:rgba(148,163,184,.08);
    border-color:rgba(148,163,184,.14);
    box-shadow:none;
}
.verdict-card.is-empty .verdict-text,
.verdict-card.is-empty .score-pill {
    color:#94A3B8;
}
.verdict-card.is-empty .score-pill {
    border-color:rgba(148,163,184,.22);
    box-shadow:none;
}
.analysis-section {
    margin-top:22px;
}
.section-title {
    color:#F8FAFC;
    font-size:.94rem;
    font-weight:750;
}
.details-link {
    color:#A78BFA;
    background:transparent;
    border:none;
    padding:0;
    font-size:.78rem;
    font-weight:700;
    cursor:pointer;
}
.dimension-list {
    margin-top:13px;
}
.dimension-row {
    gap:10px;
    min-height:40px;
    margin-bottom:11px;
}
.dim-icon {
    width:26px;
    height:26px;
    flex:0 0 26px;
    display:grid;
    place-items:center;
    background:rgba(255,255,255,.035);
    border:1px solid rgba(255,255,255,.08);
    border-radius:8px;
}
.dim-main {
    flex:1;
    min-width:0;
}
.dim-top {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:10px;
    margin-bottom:7px;
}
.dim-label {
    color:#DDE5F3;
    font-size:.82rem;
    font-weight:650;
    white-space:nowrap;
    overflow:hidden;
    text-overflow:ellipsis;
}
.dim-value {
    color:#E5E7EB;
    font-family:'JetBrains Mono', monospace;
    font-size:.78rem;
    font-weight:700;
}
.progress-track {
    height:7px;
    overflow:hidden;
    background:#222A3C;
    border-radius:999px;
    box-shadow:inset 0 0 0 1px rgba(255,255,255,.025);
}
.progress-fill {
    height:100%;
    border-radius:inherit;
}
.priority-badge {
    min-width:56px;
    text-align:center;
    border-radius:999px;
    padding:4px 8px;
    color:#fff;
    font-size:.68rem;
    font-weight:800;
}
.priority-high { background:#E5484D; box-shadow:0 0 16px rgba(229,72,77,.22); }
.priority-medium { background:#F59E0B; box-shadow:0 0 16px rgba(245,158,11,.2); }
.priority-low { background:#22C55E; box-shadow:0 0 16px rgba(34,197,94,.2); }
.priority-neutral {
    color:#CBD5E1;
    background:#262730;
    border:1px solid rgba(148,163,184,.12);
    box-shadow:none;
}
.transcript-empty {
    min-height:360px;
    display:flex;
    align-items:center;
    justify-content:center;
    text-align:center;
    color:#64748B;
    font-size:.92rem;
    font-weight:500;
}
.chart-card {
    margin-top:13px;
    padding:12px 12px 6px;
    background:linear-gradient(180deg, rgba(11,15,25,.98), rgba(13,18,31,.92));
    border:1px solid rgba(148,163,184,.14);
    border-radius:16px;
}
.range-select {
    color:#CBD5E1;
    background:#101827;
    border:1px solid rgba(148,163,184,.2);
    border-radius:999px;
    padding:6px 30px 6px 12px;
    font-size:.75rem;
    font-weight:700;
}
.intensity-chart {
    width:100%;
    height:auto;
    display:block;
}
@media (max-width: 900px) {
    .analysis-panel { padding:18px; }
    .metric-grid { flex-direction:column; }
    .verdict-card { align-items:flex-start; }
    .priority-badge { min-width:50px; }
}

/* ── AI Strategy Coach Card ────────────────────────────────── */
.ai-coach-card {
    margin-top:18px;
    padding:20px 22px;
    background:linear-gradient(135deg,rgba(26,18,53,.97),rgba(45,22,88,.85));
    border:1px solid rgba(167,139,250,.42);
    border-radius:16px;
    box-shadow:0 0 32px rgba(139,92,246,.22),
               0 0 6px rgba(167,139,250,.18),
               inset 0 1px 0 rgba(255,255,255,.06);
    position:relative;
    overflow:hidden;
}
.ai-coach-card::before {
    content:'';
    position:absolute;
    top:-40px; right:-40px;
    width:140px; height:140px;
    border-radius:50%;
    background:radial-gradient(circle,rgba(167,139,250,.18),transparent 70%);
    pointer-events:none;
}
.ai-coach-header {
    display:flex;
    align-items:center;
    gap:10px;
    margin-bottom:14px;
}
.ai-coach-icon {
    font-size:1.35rem;
    line-height:1;
}
.ai-coach-title {
    color:#fff;
    font-size:.97rem;
    font-weight:750;
    letter-spacing:.01em;
}
.ai-coach-badge {
    margin-left:auto;
    font-size:.68rem;
    font-weight:800;
    color:#C4B5FD;
    background:rgba(139,92,246,.22);
    border:1px solid rgba(167,139,250,.38);
    border-radius:999px;
    padding:3px 10px;
    letter-spacing:.06em;
    text-transform:uppercase;
}
.ai-coach-divider {
    height:1px;
    background:linear-gradient(90deg,rgba(167,139,250,.35),transparent);
    margin-bottom:14px;
}
.ai-coach-body {
    color:#DDE5F3;
    font-size:.88rem;
    line-height:1.65;
    font-weight:450;
}
.ai-coach-body em {
    color:#E9D5FF;
    font-style:italic;
}
.ai-coach-body strong {
    color:#fff;
    font-weight:700;
}
</style>
""", unsafe_allow_html=True)

# Ã¢â€â‚¬Ã¢â€â‚¬ Model loading Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
@st.cache_resource
def load_model():
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    model = TacticSenseModel()
    try:
        sd = torch.load("models/tacticsense_v1.pt", map_location=torch.device('cpu'))
        new_sd = {}
        mp = model.state_dict()
        for k, v in sd.items():
            nk = k.replace("roberta.", "transformer.")
            if nk in mp and hasattr(v, "shape") and v.shape != mp[nk].shape:
                continue
            new_sd[nk] = v
        model.load_state_dict(new_sd, strict=False)
        _MODEL_STATUS[0] = "Weights loaded (tacticsense_v1.pt)"
    except FileNotFoundError:
        _MODEL_STATUS[0] = "No weights file; using random weights"
    except Exception as e:
        _MODEL_STATUS[0] = f"Weight error: {e}"
    model.eval()
    return model, tokenizer

_MODEL_STATUS = [""]
model, tokenizer = load_model()


def reset_xlmr_runtime_state(xlmr_model):
    """Keep cached weights, but clear runtime state from the current session."""
    if xlmr_model is not None:
        xlmr_model.eval()
        if hasattr(xlmr_model, "zero_grad"):
            xlmr_model.zero_grad(set_to_none=True)
        if hasattr(xlmr_model, "reset_state"):
            xlmr_model.reset_state()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Ã¢â€â‚¬Ã¢â€â‚¬ STT helpers Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
LABEL_LIST = ["Small Talk", "Empathy", "No-need", "Vouching",
              "Non-hostile need", "Firmness", "Elicit-Pref", "Walk Away"]
WINDOW_SIZE = 5
SPEAKER_LABELS = {0: "BUYER", 1: "SELLER"}
SPEAKER_OPTIONS = ["Buyer", "Seller"]

@st.cache_resource
def load_whisper_model():
    if whisper is None:
        return None
    return whisper.load_model("base")

def pad_window(turns, speaker_ids, window_size=WINDOW_SIZE):
    """The model expects a 5-turn window; pad the left side for early turns."""
    turns = list(turns)[-window_size:]
    speaker_ids = list(speaker_ids)[-window_size:]
    missing = window_size - len(turns)
    return ([""] * missing) + turns, ([0] * missing) + speaker_ids

DIMENSION_PRESETS = [
    {
        "label": "Flakiness",
        "aliases": ["flakiness"],
        "score": 0.70,
        "color": "#A855F7",
        "icon": "spark",
    },
    {
        "label": "Non-hostile Need",
        "aliases": ["non hostile need", "non hostile", "non hostile need", "nonhostile need"],
        "score": 0.66,
        "color": "#22C55E",
        "icon": "heart",
    },
    {
        "label": "Walk Away",
        "aliases": ["walk away"],
        "score": 0.63,
        "color": "#14B8A6",
        "icon": "exit",
    },
    {
        "label": "No-need",
        "aliases": ["no need", "noneed"],
        "score": 0.59,
        "color": "#F97316",
        "icon": "slash",
    },
    {
        "label": "Elicit-Preference",
        "aliases": ["elicit preference", "elicit pref"],
        "score": 0.45,
        "color": "#C084FC",
        "icon": "target",
    },
    {
        "label": "Empathy",
        "aliases": ["empathy"],
        "score": 0.39,
        "color": "#EC4899",
        "icon": "pulse",
    },
    {
        "label": "Small Talk",
        "aliases": ["small talk"],
        "score": 0.27,
        "color": "#3B82F6",
        "icon": "chat",
    },
]

ICON_SVGS = {
    "chat": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z"/></svg>',
    "calendar": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 2v4M16 2v4M3 10h18"/><rect x="3" y="4" width="18" height="18" rx="3"/></svg>',
    "shield": '<svg width="23" height="23" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
    "spark": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m12 2 2.4 7.2L22 12l-7.6 2.8L12 22l-2.4-7.2L2 12l7.6-2.8z"/></svg>',
    "heart": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.6l-1-1a5.5 5.5 0 1 0-7.8 7.8l1 1L12 21l7.8-7.6 1-1a5.5 5.5 0 0 0 0-7.8z"/></svg>',
    "exit": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"/></svg>',
    "slash": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M5.6 5.6 18.4 18.4"/></svg>',
    "target": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/></svg>',
    "pulse": '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12h-4l-3 8-6-16-3 8H2"/></svg>',
}


def _normalise_label(label):
    return re.sub(r"[^a-z0-9]+", " ", str(label).lower()).strip()


def _priority(score):
    if score >= 0.67:
        return "High", "priority-high"
    if score >= 0.50:
        return "Medium", "priority-medium"
    return "Low", "priority-low"


def _compact_html(markup):
    """Streamlit Markdown escapes raw HTML after blank lines, so keep blocks tight."""
    return "\n".join(line.strip() for line in markup.splitlines() if line.strip())


def _dimension_rows(scores_by_label=None):
    """Return the seven right-panel dimensions, preserving the requested order."""
    score_lookup = {}
    for raw_label, score in (scores_by_label or {}).items():
        score_lookup[_normalise_label(raw_label)] = float(score)

    rows = []
    for item in DIMENSION_PRESETS:
        score = item["score"]
        for alias in item["aliases"] + [item["label"]]:
            match = score_lookup.get(_normalise_label(alias))
            if match is not None:
                score = match
                break
        rows.append({**item, "score": max(0.0, min(float(score), 1.0))})
    return rows


def _empty_dimension_rows():
    return [
        {**item, "score": 0.0, "color": "#262730", "neutral": True}
        for item in DIMENSION_PRESETS
    ]


def _render_dimension_rows(rows):
    html_rows = []
    for row in rows:
        if row.get("neutral"):
            badge, badge_class = "Neutral", "priority-neutral"
            fill_style = "background:#262730;box-shadow:none"
            score_text = "0.0"
        else:
            badge, badge_class = _priority(row["score"])
            fill_style = (
                f"background:linear-gradient(90deg,{row['color']},#F8FAFC);"
                f"box-shadow:0 0 14px {row['color']}"
            )
            score_text = f"{row['score']:.2f}"
        html_rows.append(
            f"""
            <div class="dimension-row">
                <div class="dim-icon" style="color:{row['color']}">{ICON_SVGS[row['icon']]}</div>
                <div class="dim-main">
                    <div class="dim-top">
                        <span class="dim-label">{escape(row['label'])}</span>
                        <span class="dim-value">{score_text}</span>
                    </div>
                    <div class="progress-track">
                        <div class="progress-fill" style="width:{row['score'] * 100:.1f}%;{fill_style}"></div>
                    </div>
                </div>
                <span class="priority-badge {badge_class}">{badge}</span>
            </div>
            """
        )
    return "\n".join(html_rows)


def _render_intensity_chart(current_score, is_empty=False):
    """Inline SVG keeps the glowing endpoint and dark-grid styling deterministic."""
    values = [0.0] * 6 if is_empty else [0.18, 0.30, 0.27, 0.50, 0.60, max(0.0, min(float(current_score), 1.0))]
    labels = ["-5m", "-4m", "-3m", "-2m", "-1m", "Now"]
    line_color = "#262730" if is_empty else "#A855F7"
    area_opacity = "0" if is_empty else ".28"
    point_label = "--" if is_empty else f"{values[-1]:.2f}"
    width, height = 520, 220
    left, right, top, bottom = 44, 24, 20, 40
    plot_w = width - left - right
    plot_h = height - top - bottom

    points = []
    for i, value in enumerate(values):
        x = left + (plot_w * i / (len(values) - 1))
        y = top + (1.0 - value) * plot_h
        points.append((x, y))

    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    x_labels = "".join(
        f'<text x="{points[i][0]:.1f}" y="205" text-anchor="middle" fill="#697386" font-size="12">{label}</text>'
        for i, label in enumerate(labels)
    )
    y_ticks = []
    for value, label in [(0, "0"), (0.5, "0.5"), (1.0, "1.0")]:
        y = top + (1.0 - value) * plot_h
        y_ticks.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="rgba(148,163,184,.12)" />'
            f'<text x="18" y="{y + 4:.1f}" fill="#697386" font-size="12">{label}</text>'
        )
    end_x, end_y = points[-1]
    glow_points = " ".join(f"{x:.1f},{y + 10:.1f}" for x, y in points)

    return f"""
    <svg class="intensity-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Tactic intensity over time">
        <defs>
            <filter id="purpleGlow" x="-40%" y="-40%" width="180%" height="180%">
                <feGaussianBlur stdDeviation="5" result="blur"/>
                <feMerge>
                    <feMergeNode in="blur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
            <linearGradient id="areaFill" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stop-color="#A855F7" stop-opacity="{area_opacity}"/>
                <stop offset="100%" stop-color="#A855F7" stop-opacity="0"/>
            </linearGradient>
        </defs>
        <rect x="0" y="0" width="{width}" height="{height}" rx="14" fill="#0B0F19"/>
        {''.join(y_ticks)}
        <path d="M {glow_points} L {width-right},{height-bottom} L {left},{height-bottom} Z" fill="url(#areaFill)"/>
        <polyline points="{polyline}" fill="none" stroke="{line_color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" filter="url(#purpleGlow)"/>
        <circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="9" fill="#FFFFFF" filter="url(#purpleGlow)"/>
        <circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="4" fill="{line_color}"/>
        <rect x="{end_x - 27:.1f}" y="{end_y - 35:.1f}" width="54" height="24" rx="12" fill="#FFFFFF"/>
        <text x="{end_x:.1f}" y="{end_y - 18:.1f}" text-anchor="middle" fill="#111827" font-size="12" font-weight="800">{point_label}</text>
        {x_labels}
    </svg>
    """


def render_tactical_analysis_panel(total_turns, window_turns, dimensions, verdict_label, verdict_score):
    is_empty = int(total_turns) == 0
    display_dimensions = _empty_dimension_rows() if is_empty else dimensions
    safe_verdict_label = escape(verdict_label)
    verdict_text = "Awaiting input..." if is_empty else f"Pressure Tactic detected via '{safe_verdict_label}'"
    score_text = "--" if is_empty else f"{float(verdict_score):.2f}"
    verdict_state_class = " is-empty" if is_empty else ""
    return _compact_html(f"""
    <div class="analysis-panel">
        <div class="analysis-header">
            <div class="analysis-title">Tactical Analysis</div>
            <div class="live-badge"><span class="live-dot"></span>Live</div>
        </div>

        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-icon">{ICON_SVGS['chat']}</div>
                <div class="metric-value">{int(total_turns)}</div>
                <div class="metric-label">Total Turns</div>
            </div>
            <div class="metric-card">
                <div class="metric-icon">{ICON_SVGS['calendar']}</div>
                <div class="metric-value">{int(window_turns)}/{WINDOW_SIZE}</div>
                <div class="metric-label">Window</div>
            </div>
        </div>

        <div class="verdict-card{verdict_state_class}">
            <div class="shield-icon">{ICON_SVGS['shield']}</div>
            <div class="verdict-copy">
                <div class="eyebrow">Final Verdict</div>
                <div class="verdict-text">{verdict_text}</div>
            </div>
            <div class="score-pill">{score_text}</div>
        </div>

        <div class="analysis-section">
            <div class="section-header">
                <div class="section-title">Behavior Dimensions</div>
                <button class="details-link" type="button">Details</button>
            </div>
            <div class="dimension-list">
                {_render_dimension_rows(display_dimensions)}
            </div>
        </div>

        <div class="analysis-section">
            <div class="chart-toolbar">
                <div class="section-title">Tactic Intensity Over Time</div>
                <select class="range-select" aria-label="Time range">
                    <option selected>Last 5 min</option>
                    <option>Last 15 min</option>
                    <option>Last 30 min</option>
                </select>
            </div>
            <div class="chart-card">
                {_render_intensity_chart(verdict_score, is_empty=is_empty)}
            </div>
        </div>
    </div>
    """)

def _coerce_speaker_id(value, fallback=0):
    try:
        sid = int(value)
    except (TypeError, ValueError):
        sid = fallback
    return sid if sid in SPEAKER_LABELS else fallback

def parse_speaker_prefix(text):
    """Let transcripts like 'buyer: final offer' override the current speaker."""
    cleaned = (text or "").strip()
    match = re.match(r"^(buyer|seller)\s*[:\-]\s*(.+)$", cleaned, flags=re.I)
    if not match:
        return None, cleaned
    role, rest = match.groups()
    sid = 0 if role.lower() == "buyer" else 1
    return sid, rest.strip()

def auto_detect_role(text):
    text_lower = text.lower()
    buyer_signals = [
        # English buyer signals
        "buy", "budget", "affordable", "discount", "price drop", "can you do",
        "maximum i can", "interested in buying", "listed online",
        # Hinglish buyer signals
        "bhai", "bhaya", "thoda kam karo", "mehenga", "discount de do",
        "student hu", "budget nahi hai", "last price",
    ]
    seller_signals = [
        # English seller signals
        "sell", "asking price", "mint condition", "available", "market value",
        "firm on", "listed for", "selling", "upgraded",
        # Hinglish seller signals
        "fix rate", "nuksan", "ekdam naya", "sahi daam", "market me", "is se kam nahi",
    ]
    buyer_score = sum(1 for word in buyer_signals if word in text_lower)
    seller_score = sum(1 for word in seller_signals if word in text_lower)
    if buyer_score > seller_score:
        return 0  # BUYER
    elif seller_score > buyer_score:
        return 1  # SELLER
    else:
        return 0 if len(st.session_state.transcript) % 2 == 0 else 1

def append_turn(text, speaker_id=None):
    parsed_sid, cleaned = parse_speaker_prefix(text)
    if not cleaned:
        return
    sid = parsed_sid if parsed_sid is not None else _coerce_speaker_id(speaker_id, auto_detect_role(cleaned))
    st.session_state.transcript.append((sid, cleaned))
    st.session_state.turns.append(cleaned)
    st.session_state.ids.append(sid)
    st.session_state.log.append(("OK", f"{SPEAKER_LABELS[sid]}: {cleaned}"))

def transcribe_with_whisper(audio_bytes: bytes):
    """Local fallback STT for clips Google cannot understand."""
    w_model = load_whisper_model()
    if w_model is None:
        raise RuntimeError("Whisper is not installed.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        result = w_model.transcribe(tmp_path, language="en", fp16=False)
        return (result.get("text") or "").strip()
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def _get_input_devices():
    """Return PortAudio devices that can actually record audio."""
    if pyaudio is None:
        names = sr.Microphone.list_microphone_names()
        return [
            {"index": i, "name": name, "channels": "?", "rate": "?"}
            for i, name in enumerate(names)
        ], None, None

    pa = pyaudio.PyAudio()
    devices = []
    default_index = None
    try:
        try:
            default_index = int(pa.get_default_input_device_info()["index"])
        except Exception:
            default_index = None

        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            channels = int(info.get("maxInputChannels", 0) or 0)
            if channels <= 0:
                continue

            name = str(info.get("name", f"Device {i}")).strip()
            lowered = name.lower()
            if any(skip in lowered for skip in ("stereo mix", "output", "speaker", "headphone")):
                continue

            devices.append({
                "index": i,
                "name": name,
                "channels": channels,
                "rate": int(float(info.get("defaultSampleRate", 0) or 0)),
            })
    finally:
        pa.terminate()

    return devices, default_index, None

def _label_device(device, default_index):
    default = " (default)" if device["index"] == default_index else ""
    details = f"{device['channels']}ch"
    if device["rate"] != "?":
        details += f", {device['rate']}Hz"
    name = (
        str(device["name"])
        .replace("Ã‚Â®", "")
        .replace("Â®", "")
        .replace("Ã¢", "")
    )
    return f"[{device['index']}] {name[:48]} - {details}{default}"

def _stt_callback(recognizer, audio):
    """Background thread: Google STT -> global queue."""
    try:
        text = recognizer.recognize_google(audio, language="en-IN")
        print(f"[STT OK] {text}")
        _q.put({
            "type": "speech",
            "text": text,
            "speaker_id": _coerce_speaker_id(_lst.get("active_speaker_id"), 0),
        })
    except sr.UnknownValueError:
        print("[STT] Google did not understand; trying Whisper fallback")
        try:
            text = transcribe_with_whisper(audio.get_wav_data())
            if text:
                print(f"[WHISPER OK] {text}")
                _q.put({
                    "type": "speech",
                    "text": text,
                    "speaker_id": _coerce_speaker_id(_lst.get("active_speaker_id"), 0),
                })
                _q.put({"type": "info", "text": f"Whisper fallback transcribed: {text}"})
            else:
                _q.put({"type": "warn", "text": "Audio detected, but no speech was recognized."})
        except Exception as e:
            print(f"[WHISPER ERR] {e}")
            _q.put({"type": "warn", "text": "Audio detected, but speech was not understood. Try a clearer/longer recording."})
    except sr.RequestError as e:
        print(f"[STT NET] {e}")
        _q.put({"type": "error", "text": f"Network error: {e}"})
    except Exception as e:
        print(f"[STT ERR] {e}")

def transcribe_recorded_audio(audio_bytes: bytes):
    """Transcribe a browser-recorded WAV clip with local fallback."""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio, language="en-IN"), "Google"
    except sr.UnknownValueError:
        text = transcribe_with_whisper(audio_bytes)
        if text:
            return text, "Whisper"
        raise

def _do_start(device_index: int, energy: int, dynamic: bool, pause: float):
    """Runs in daemon thread. Calibrates mic then starts background listening."""
    r = sr.Recognizer()

    # â”€â”€ STEALTH / AGGRESSIVE-CUT SETTINGS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # dynamic_energy_threshold MUST be False so the background thread never
    # overwrites the calibrated room baseline with live audio readings.
    r.dynamic_energy_threshold = False

    # Hardcode to 0.5 s so the engine slices a new phrase after every
    # half-second of silence â€” critical for multi-turn rapid delivery.
    r.pause_threshold      = 0.5
    r.phrase_threshold     = 0.1   # min voiced seconds to count as speech
    r.non_speaking_duration = 0.3  # trailing silence kept at end of clip
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    try:
        # â”€â”€ DIAGNOSTIC: Calibration phase â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        try:
            mic = sr.Microphone(device_index=device_index)
            with mic as src:
                r.adjust_for_ambient_noise(src, duration=2.0)
        except Exception as mic_err:
            raise  # re-raise so the outer except still fires correctly

        # Cap threshold so loud rooms don't block all speech
        calibrated = r.energy_threshold
        cap = max(energy, min(calibrated, energy * 2))
        r.energy_threshold = cap
        print(
            f"[MIC] Calibrated raw={calibrated:.0f}  ->"
            f"  capped={cap:.0f}  (user={energy}) | "
            f"pause_threshold={r.pause_threshold}s | dynamic=OFF"
        )

        # â”€â”€ DIAGNOSTIC: Background listener phase â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        try:
            live_mic = sr.Microphone(device_index=device_index)
            stop_fn = r.listen_in_background(live_mic, _stt_callback, phrase_time_limit=8)
        except Exception as bg_err:
            raise

        _lst["stop_fn"] = stop_fn
        _lst["active"]   = True
        _lst["starting"] = False
        _lst["error"]    = None
        _q.put({"type": "info",
                "text": f"Live on device [{device_index}] threshold={cap:.0f}; speak now!"})

    except Exception as e:
        _lst["starting"] = False
        _lst["active"]   = False
        _lst["error"]    = str(e)
        _q.put({"type": "error", "text": f"Mic failed: {e}"})

def start_mic(device_index: int, energy: int, dynamic: bool, pause: float):
    # ── Singleton guard ──────────────────────────────────────────────────────
    # Prevent the PyAudio thread-bomb: if a background listener is already
    # active or starting (e.g. because st.rerun() re-executed this code path),
    # do NOT spawn another one.  _lst is a @st.cache_resource dict that
    # persists across reruns at the process level, so this check is reliable.
    if _lst["active"] or _lst["starting"]:
        return
    # ────────────────────────────────────────────────────────────────────────
    if device_index is None:
        _lst["starting"] = False
        _lst["active"] = False
        _lst["error"] = "No recording microphone was found."
        _q.put({"type": "error", "text": _lst["error"]})
        return
    _lst["starting"] = True
    _lst["error"]    = None
    threading.Thread(
        target=_do_start,
        args=(device_index, energy, dynamic, pause),
        daemon=True
    ).start()

def stop_mic():
    fn = _lst.get("stop_fn")
    if fn:
        fn(wait_for_stop=False)
    _lst["stop_fn"]  = None
    _lst["active"]   = False
    _lst["starting"] = False

# Ã¢â€â‚¬Ã¢â€â‚¬ Session state Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
for k, v in [
    ("transcript", []),
    ("turns", []),
    ("ids", []),
    ("log", []),
    ("total_turns", 0),
    ("window", 0),
    ("speech_enabled", True),
    ("last_audio_hash", None),
    # Singleton guard: ensures the PyAudio background listener is spawned
    # exactly ONCE per browser session, regardless of how many times
    # st.rerun() fires.  Without this flag, every rerun that hits the
    # "Start Browser Mic" branch creates a NEW C-level PortAudio thread,
    # producing a thread-bomb that causes Exit-code 3221225477.
    ("audio_thread_started", False),
    # AI Coach cache: stores the last coach response so the LLM is NOT
    # re-called on every Streamlit rerun — only when the transcript or
    # detected tactic actually changes.
    ("_coach_cache_key", None),
    ("_coach_advice",    None),   # None = not yet fetched; str = advice/fallback
    # Public alias surfaced in the AI Strategy Coach render block
    ("llm_tips",         None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

def drain():
    """Pull everything from _q into session state (main thread only)."""
    while not _q.empty():
        try:
            item = _q.get_nowait()
        except queue.Empty:
            break
        kind = item.get("type")
        text = item.get("text", "")
        if kind == "speech":
            s_id = auto_detect_role(text)
            append_turn(text, s_id)
        elif kind == "warn":
            st.session_state.log.append(("OK", text))
        elif kind == "error":
            st.session_state.log.append(("ERR", text))
        elif kind == "info":
            st.session_state.log.append(("INFO", text))
    # keep log bounded
    if len(st.session_state.log) > 30:
        st.session_state.log = st.session_state.log[-30:]

# Ã¢â€â‚¬Ã¢â€â‚¬ DRAIN before any rendering Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
drain()

# Ã¢â€â‚¬Ã¢â€â‚¬ SIDEBAR Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
sb = st.sidebar
sb.markdown("## TacticSense")
sb.markdown("---")

# Status pill
if _lst["starting"]:
    sb.markdown('<span class="pill-starting">CALIBRATING...</span>', unsafe_allow_html=True)
elif _lst["active"]:
    sb.markdown('<span class="pill-active">LIVE</span>', unsafe_allow_html=True)
elif st.session_state.get("speech_enabled", True):
    sb.markdown('<span class="pill-active">MIC READY</span>', unsafe_allow_html=True)
else:
    sb.markdown('<span class="pill-off">INACTIVE</span>', unsafe_allow_html=True)

if _lst["error"]:
    sb.error(f"{_lst['error']}")

sb.markdown("---")
sb.markdown("### Microphone")

# Build mic list Ã¢â‚¬â€ only show input-capable devices
try:
    _input_devices, _default_device, _device_error = _get_input_devices()
except Exception as e:
    _input_devices, _default_device, _device_error = [], None, str(e)

if _device_error:
    sb.warning(f"Microphone scan failed: {_device_error}")

_input_mics = {
    _label_device(device, _default_device): device["index"]
    for device in _input_devices
}

if _input_mics:
    _mic_labels = list(_input_mics.keys())
    _default_mic_idx = next(
        (j for j, lbl in enumerate(_mic_labels) if f"[{_default_device}]" in lbl),
        0
    )
    sel_label = sb.selectbox("Select Input Device", _mic_labels, index=_default_mic_idx)
    sel_device = _input_mics[sel_label]
else:
    sb.error("No input-capable microphone devices were found.")
    sel_device = None

energy_val  = sb.slider("Energy Threshold", 50, 3000, 300, 25,
    help="How loud audio must be before treated as speech. Lower = more sensitive. "
         "The app auto-caps this after calibration so a room with high background noise "
         "won't block all speech.")
use_dynamic = sb.checkbox("Auto-adjust threshold", value=False,
    help="Let the recognizer continuously re-calibrate. DISABLE if speech keeps being missed.")
pause_val   = sb.slider("Pause Duration (s)", 0.3, 2.5, 0.8, 0.1,
    help="Silence needed to mark end of a phrase.")

sb.markdown("---")
sb.markdown("### Control")

if st.session_state.speech_enabled:
    if sb.button("Stop Browser Mic", key="ctl_btn"):
        st.session_state.speech_enabled = False
        st.rerun()
else:
    if sb.button("Start Browser Mic", key="ctl_btn"):
        st.session_state.speech_enabled = True
        st.rerun()

if sb.button("Clear Session", key="clear_btn"):
    reset_xlmr_runtime_state(model)
    st.session_state.transcript = []
    st.session_state.turns = []
    st.session_state.ids = []
    st.session_state.log = []
    st.session_state.total_turns = 0
    st.session_state.window = 0
    st.session_state.last_audio_hash = None
    st.session_state._coach_cache_key = None
    st.session_state._coach_advice = None
    st.session_state.llm_tips = None
    while not _q.empty():
        try:
            _q.get_nowait()
        except queue.Empty:
            break
    st.rerun()

sb.markdown("---")
sb.caption(_MODEL_STATUS[0])
sb.markdown("---")
sb.markdown("**Tactics**")
for lbl in LABEL_LIST:
    sb.markdown(f"<span style='color:#475569;font-size:.78rem'>- {lbl}</span>",
                unsafe_allow_html=True)

# Ã¢â€â‚¬Ã¢â€â‚¬ MAIN PAGE Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
st.markdown('<div class="ts-title">TacticSense Live</div>', unsafe_allow_html=True)
st.markdown('<div class="ts-sub">REAL-TIME NEGOTIATION BEHAVIOR ANALYSIS</div>',
            unsafe_allow_html=True)

col1, col2 = st.columns([2, 1], gap="large")

# Ã¢â€â‚¬Ã¢â€â‚¬ LEFT: Transcript Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
with col1:
    st.markdown("#### Live Transcript")

    # Manual entry (always visible, not in expander)
    mc1, mc2 = st.columns([5, 1])
    with mc1:
        manual_text = st.text_input(
            "Manual input", label_visibility="collapsed",
            placeholder="Type a turn, or prefix with Buyer: / Seller:",
            key="manual_in"
        )
    with mc2:
        if st.button("Add", key="add_btn"):
            if manual_text.strip():
                _q.put({
                    "type": "speech",
                    "text": manual_text.strip(),
                })
                st.rerun()

    audio_clip = st.audio_input(
        "Speech input",
        key="speech_clip",
        disabled=not st.session_state.speech_enabled,
        label_visibility="collapsed",
    )
    if audio_clip is not None:
        audio_bytes = audio_clip.getvalue()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        if audio_hash != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = audio_hash
            try:
                text, engine = transcribe_recorded_audio(audio_bytes)
                _q.put({
                    "type": "speech",
                    "text": text,
                })
                _q.put({"type": "info", "text": f"Browser mic transcribed by {engine}: {text}"})
            except sr.UnknownValueError:
                _q.put({"type": "warn", "text": "Audio detected, but speech was not understood. Try a clearer/longer recording."})
            except sr.RequestError as e:
                _q.put({"type": "error", "text": f"Speech recognition network error: {e}"})
            except Exception as e:
                _q.put({"type": "error", "text": f"Speech input failed: {e}"})
            st.rerun()

    # STT log
    with st.expander("STT Debug Log", expanded=_lst["active"]):
        if st.session_state.log:
            for icon, msg in reversed(st.session_state.log[-12:]):
                st.markdown(
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:.75rem;"
                    f"color:#64748b;padding:2px 0'>{icon} {msg}</div>",
                    unsafe_allow_html=True
                )
        else:
            st.caption("No activity yet.")

    # Chat box â€” STEALTH UI
    # All turns are rendered with a neutral avatar so no BUYER / SELLER label
    # is visible to the presenter.  The backend auto_detect_role() still runs
    # in drain(), populates s_id, and feeds ids_buffer to the PyTorch model.
    chat = st.container(height=450)
    with chat:
        if not st.session_state.transcript:
            st.markdown(
                "<div class='transcript-empty'>Start browser mic or type a turn above</div>",
                unsafe_allow_html=True
            )
        else:
            for sid, msg in st.session_state.transcript:
                # sid is still tracked internally; only the visual label is hidden
                with st.chat_message("user"):
                    st.markdown(msg)

# Ã¢â€â‚¬Ã¢â€â‚¬ RIGHT: Analysis Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
with col2:
    turns_w = st.session_state.turns[-WINDOW_SIZE:]
    ids_w   = st.session_state.ids[-WINDOW_SIZE:]
    total_turns = len(st.session_state.transcript)
    window_turns = min(len(st.session_state.turns), WINDOW_SIZE)
    st.session_state.total_turns = total_turns
    st.session_state.window = window_turns
    score_map = {}

    if turns_w:
        try:
            model_turns, model_ids = pad_window(turns_w, ids_w)
            labels, scores = predict_tactic(model_turns, model_ids, model, tokenizer)
            score_map = dict(zip(labels, scores))
        except Exception as e:
            st.error(f"Prediction error: {e}")

    dimensions = _dimension_rows(score_map)
    top_dimension = max(dimensions, key=lambda item: item["score"])
    st.markdown(
        render_tactical_analysis_panel(
            total_turns=total_turns,
            window_turns=window_turns,
            dimensions=dimensions,
            verdict_label=top_dimension["label"],
            verdict_score=top_dimension["score"],
        ),
        unsafe_allow_html=True,
    )

    # ── AI STRATEGY COACH ────────────────────────────────────────────────────
    # Always render the coach card container so it's visible from turn 1.
    # Trigger: fire the LLM only after ≥ 5 turns (enough transcript context).
    # Cache:   re-call the LLM only when the tactic label changes, NOT on
    #          every Streamlit rerun — prevents API spam.

    _top_label = top_dimension["label"]

    if total_turns >= 5:
        _cache_key = f"{total_turns}:{_top_label}"

        if st.session_state._coach_cache_key != _cache_key:
            # Build a compact, token-efficient transcript string.
            # Do NOT HTML-escape here — the LLM needs plain text.
            _recent_turns = st.session_state.transcript[-10:]
            _transcript_str = "\n".join(
                f"{SPEAKER_LABELS.get(sid, 'SPEAKER')}: {msg}"
                for sid, msg in _recent_turns
            )

            # Show spinner inline (no st.empty wrapper — that silently eats it)
            with st.spinner("🤖 AI Coach is analyzing strategies..."):
                try:
                    _advice = _llm_coach_fn(
                        transcript=_transcript_str,
                        detected_tactic=_top_label,
                    )
                except Exception as _exc:
                    # Belt-and-suspenders: llm_coach already swallows errors,
                    # but guard here too so the panel never crashes.
                    _advice = (
                        f"\u26a0\ufe0f **Fallback Tip:** The buyer is showing high "
                        f"**{escape(_top_label)}**. Consider anchoring your "
                        "price firmly and asking for a token advance to verify intent."
                    )

            st.session_state._coach_advice    = _advice
            st.session_state._coach_cache_key = _cache_key
            st.session_state.llm_tips         = _advice

    # ── Render the AI Strategy Coach card (always shown, content varies) ──────
    _display_advice = st.session_state._coach_advice

    if _display_advice:
        # Convert newlines → <br> so multi-sentence advice renders properly
        # inside the HTML card (st.markdown strips bare newlines in HTML blocks)
        _advice_html = escape(_display_advice).replace("\n", "<br>").replace(
            "**", "<strong>", 1
        )
        # Simpler: keep the raw text safe and let CSS handle wrapping
        _advice_safe = escape(_display_advice).replace("&#x27;", "'").replace("&amp;", "&")
        _coach_html = (
            '<div class="ai-coach-card">'
            '<div class="ai-coach-header">'
            '<span class="ai-coach-icon">\U0001f916</span>'
            '<span class="ai-coach-title">AI Strategy Coach</span>'
            '<span class="ai-coach-badge">Live Insight</span>'
            '</div>'
            '<div class="ai-coach-divider"></div>'
            f'<div class="ai-coach-body">{_advice_safe}</div>'
            '</div>'
        )
        st.markdown(_coach_html, unsafe_allow_html=True)
    elif total_turns > 0 and total_turns < 5:
        # Teaser card: visible from turn 1 so user knows the feature is coming
        _turns_left = 5 - total_turns
        _coach_html = (
            '<div class="ai-coach-card">'
            '<div class="ai-coach-header">'
            '<span class="ai-coach-icon">\U0001f916</span>'
            '<span class="ai-coach-title">AI Strategy Coach</span>'
            '<span class="ai-coach-badge">Warming Up</span>'
            '</div>'
            '<div class="ai-coach-divider"></div>'
            '<div class="ai-coach-body" style="color:#7B8496;font-style:italic;">'
            f'Gathering context\u2026 <strong style="color:#C4B5FD">{_turns_left} more '
            f'turn{"s" if _turns_left != 1 else ""}</strong> until live coaching activates.'
            '</div>'
            '</div>'
        )
        st.markdown(_coach_html, unsafe_allow_html=True)
    else:
        # Zero turns — show locked placeholder
        _coach_html = (
            '<div class="ai-coach-card" style="opacity:0.45;">'
            '<div class="ai-coach-header">'
            '<span class="ai-coach-icon">\U0001f916</span>'
            '<span class="ai-coach-title">AI Strategy Coach</span>'
            '<span class="ai-coach-badge" style="color:#64748b;border-color:rgba(100,116,139,.3);">Standby</span>'
            '</div>'
            '<div class="ai-coach-divider"></div>'
            '<div class="ai-coach-body" style="color:#475569;font-style:italic;">'
            'Start the negotiation to unlock real-time counter-strategy coaching.'
            '</div>'
            '</div>'
        )
        st.markdown(_coach_html, unsafe_allow_html=True)

# Ã¢â€â‚¬Ã¢â€â‚¬ AUTO-REFRESH Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
if _lst["active"] or _lst["starting"]:
    time.sleep(1.0)
    st.rerun()

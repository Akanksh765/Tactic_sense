"""
src/llm_coach.py  ─  LLM Counter-Strategy Engine for TacticSense
═══════════════════════════════════════════════════════════════════

Drop-in LLM helper module. Zero coupling to the rest of the app.

HOW TO SWAP PROVIDERS
─────────────────────
Change the single constant `LLM_PROVIDER` below:
    "gemini"  → Google Gemini 1.5 Flash  (default, free tier available)
    "openai"  → OpenAI GPT-4o-mini

HOW TO SET YOUR API KEY (pick any one method)
──────────────────────────────────────────────
1. .streamlit/secrets.toml  →  GEMINI_API_KEY = "AIza..."
2. Environment variable     →  set GEMINI_API_KEY=AIza...
3. Direct fallback          →  set _HARDCODED_KEY below (dev only, never commit)

If no key is found the function silently returns a tactic-specific
static tip — the Streamlit app will never crash.
"""

import os
import re
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ── PROVIDER SWITCH  (only line you need to change to swap LLMs) ─────────────
# ─────────────────────────────────────────────────────────────────────────────
LLM_PROVIDER: str = "gemini"   # "gemini"  |  "openai"

# Dev-only escape hatch — leave empty in production, never commit a real key
_HARDCODED_KEY: str = ""

# ─────────────────────────────────────────────────────────────────────────────
# ── SYSTEM PROMPT  (negotiation coach persona) ───────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are an elite negotiation coach with 20+ years of experience in high-stakes \
commercial, procurement, and conflict-resolution deals.

Your job: study the live negotiation transcript and the tactic that was just \
detected, then deliver 1-3 immediately-actionable Markdown bullet points the \
negotiator can use RIGHT NOW.

=================================================================
CRITICAL CONSTRAINT — THIS OVERRIDES ALL OTHER RULES BELOW
=================================================================
- CRITICAL CONSTRAINT: You are a live, real-time negotiation coach. You must \
NEVER generate generic bracketed placeholders, wildcards, or template tokens \
such as [date], [time], [price], [buyer name], or [insert text].
- If a deadline or time constraint is strategically required, derive it \
naturally from the context (e.g. 'by tonight', 'before this evening', 'within \
the next hour') or use relative timeframes. Never use exact calendar dates \
unless explicitly stated in the transcript history.
- If a price anchor is needed, use the actual numerical figures mentioned in the \
conversation history (e.g. 42.5k, 40k) instead of generic placeholders.
- If a specific detail (name, date, figure) is absent from the transcript, omit it \
entirely rather than inventing placeholders.
=================================================================

Strict output format
────────────────────
- Respond ONLY with 1-3 Markdown bullet points, each starting with '- '.
- Each bullet: one punchy, confident, specific action or sentence. Max 25 words.
- No introduction, no preamble, no sign-off, no conversational filler.
- Do NOT start with: 'Sure,', 'Here is', 'As a coach,', 'Based on the transcript,'.
- Where possible, include a verbatim sentence the negotiator can say aloud, \
formatted in *italics*.
- Never expose that you 'detected a tactic' — frame it as natural coaching.
- Respond in English, even if the transcript is multilingual or Hinglish.\
"""

# ─────────────────────────────────────────────────────────────────────────────
# ── TACTIC-SPECIFIC FALLBACK TIPS ────────────────────────────────────────────
#    Used whenever the API call fails (no key, no internet, rate-limit, etc.)
# ─────────────────────────────────────────────────────────────────────────────
_FALLBACK_TIPS: dict[str, str] = {
    "Firmness": (
        "⚠️ **Counter:** Acknowledge their position without conceding ground — "
        "say *\"I respect that stance; let's explore what flexibility looks like "
        "on delivery terms or bundled value instead of price.\"*"
    ),
    "Walk Away": (
        "⚠️ **Counter:** Don't chase or panic. Calmly anchor: "
        "*\"I understand — take the time you need. The offer holds until end of day; "
        "I'm confident it's the best fit once you've compared fully.\"*"
    ),
    "No-need": (
        "⚠️ **Counter:** Surface the latent cost of inaction — ask: "
        "*\"What does solving nothing cost your team by end of quarter "
        "in time, rework, or missed targets?\"*"
    ),
    "Non-hostile need": (
        "⚠️ **Counter:** Align on shared outcome — say: "
        "*\"We both want a smooth transition. Walk me through exactly where "
        "the current approach falls short for your team.\"*"
    ),
    "Vouching": (
        "⚠️ **Counter:** Probe the reference neutrally — ask: "
        "*\"That's a strong endorsement. What specifically made that alternative "
        "a better fit than what we're proposing today?\"*"
    ),
    "Elicit-Pref": (
        "⚠️ **Counter:** Redirect with a question before revealing your hand — "
        "*\"Before I go further, what's the single outcome that would make "
        "this a clear win for your side?\"*"
    ),
    "Small Talk": (
        "⚠️ **Counter:** Match rapport briefly, then redirect with intent — "
        "*\"Great to connect! I know your time is valuable — "
        "shall we focus on the two or three points that matter most today?\"*"
    ),
    "Empathy": (
        "⚠️ **Counter:** Validate without conceding — say: "
        "*\"I genuinely hear that concern. Let's make sure whatever we agree "
        "on today directly addresses it — can you tell me more?\"*"
    ),
}

_DEFAULT_FALLBACK: str = (
    "⚠️ **Counter:** Remind the counterpart of mutual goals and ask an open-ended "
    "question — *\"What outcome would make this negotiation a clear success for "
    "both sides?\"*"
)


# ─────────────────────────────────────────────────────────────────────────────
# ── PRIVATE: Provider-specific API callers ───────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def _build_user_message(transcript: str, detected_tactic: str) -> str:
    """Format the user turn that is sent to any LLM provider."""
    return (
        f"**Detected negotiation tactic:** {detected_tactic}\n\n"
        f"**Live transcript (last few turns):**\n"
        f"───────────────────────────────────\n"
        f"{transcript}\n"
        f"───────────────────────────────────\n\n"
        f"Give me one counter-move I can use right now."
    )


_BRACKET_REPLACEMENTS: dict[str, str] = {
    "[date]": "by this evening",
    "[time]": "within the next hour",
    "[price]": "the agreed amount",
    "[amount]": "the amount on the table",
    "[buyer name]": "the buyer",
    "[seller]": "the seller",
    "[insert text]": "the right wording",
    "[insert text here]": "the right wording",
}


def _sanitize_llm_response(advice: str) -> str:
    """Ensure the LLM output is free of bracketed placeholder tokens and is bullet formatted."""
    if advice is None:
        return ""
    advice = advice.strip()
    if not re.search(r"[\[\]]", advice):
        advice_lines = [line.strip() for line in advice.splitlines() if line.strip()]
    else:
        def _replace_token(match: re.Match) -> str:
            token = match.group(0).lower()
            return _BRACKET_REPLACEMENTS.get(token, "")

        advice = re.sub(
            r"\[(?:date|time|price|amount|buyer name|seller|insert text here|insert text)\]",
            _replace_token,
            advice,
            flags=re.IGNORECASE,
        ).replace("[", "").replace("]", "")
        advice_lines = [line.strip() for line in advice.splitlines() if line.strip()]

    normalized_lines = []
    for line in advice_lines:
        if line.startswith("- "):
            normalized_lines.append(line)
        else:
            normalized_lines.append(f"- {line}")

    return "\n".join(normalized_lines)


def _call_gemini(transcript: str, detected_tactic: str, api_key: str) -> str:
    """Call Google Gemini 1.5 Flash and return the counter-strategy text."""
    import google.generativeai as genai  # pip install google-generativeai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=_SYSTEM_PROMPT,
    )
    response = model.generate_content(
        _build_user_message(transcript, detected_tactic),
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=150,
            temperature=0.65,
        ),
    )
    return response.text.strip()


def _call_openai(transcript: str, detected_tactic: str, api_key: str) -> str:
    """Call OpenAI GPT-4o-mini and return the counter-strategy text."""
    from openai import OpenAI  # pip install openai

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_message(transcript, detected_tactic)},
        ],
        max_tokens=150,
        temperature=0.65,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# ── PUBLIC API ───────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def get_llm_countermeasure(transcript: str, detected_tactic: str) -> str:
    """
    Generate a real-time counter-strategy for the detected negotiation tactic.

    Args:
        transcript:      The last N turns of conversation as a single string.
                         Each line should be formatted as "SPEAKER: utterance".
        detected_tactic: Primary tactic label from predict_tactic(), e.g. "Firmness".

    Returns:
        A short (≤2 sentences), actionable counter-strategy string.
        Falls back gracefully to a static tactic-specific tip on any failure —
        the Streamlit app will never crash or block due to this call.

    Error-handling hierarchy:
        1. No API key found          → silent fallback
        2. LLM library not installed → warns user to install, then fallback
        3. Network / timeout error   → silent fallback
        4. API quota / auth error    → silent fallback
        5. Unknown exception         → logs error, silent fallback
    """
    fallback = _FALLBACK_TIPS.get(detected_tactic, _DEFAULT_FALLBACK)

    # ── Step 1: Resolve API key from all possible sources ────────────────────
    key_env_name = "GEMINI_API_KEY" if LLM_PROVIDER == "gemini" else "OPENAI_API_KEY"
    api_key = _HARDCODED_KEY  # Start with hardcoded (empty by default)

    if not api_key:
        # Try st.secrets first (works inside a running Streamlit app)
        try:
            import streamlit as st
            api_key = st.secrets.get(key_env_name, "")
        except Exception:
            api_key = ""

    if not api_key:
        # Final attempt: OS environment variable
        api_key = os.environ.get(key_env_name, "")

    if not api_key:
        logger.warning(
            "[llm_coach] No API key found for provider '%s' "
            "(checked st.secrets, env var %s). Using static fallback.",
            LLM_PROVIDER, key_env_name,
        )
        return fallback

    # ── Step 2: Dispatch to the selected provider ────────────────────────────
    try:
        if LLM_PROVIDER == "gemini":
            advice = _call_gemini(transcript, detected_tactic, api_key)
        elif LLM_PROVIDER == "openai":
            advice = _call_openai(transcript, detected_tactic, api_key)
        else:
            logger.error("[llm_coach] Unknown LLM_PROVIDER value: '%s'", LLM_PROVIDER)
            return fallback

        return _sanitize_llm_response(advice)

    except ImportError as exc:
        # Library not installed — give a helpful install hint
        pkg = "google-generativeai" if LLM_PROVIDER == "gemini" else "openai"
        logger.error("[llm_coach] Client library missing: %s", exc)
        return (
            f"⚠️ Run `pip install {pkg}` to enable live AI coaching.  \n"
            f"{fallback}"
        )

    except Exception as exc:
        # Catches: network errors, auth/quota errors, timeouts, API changes, etc.
        logger.error(
            "[llm_coach] LLM call failed (%s): %s — returning static fallback.",
            type(exc).__name__, exc,
        )
        return fallback


def get_ai_strategy_advice(transcript: str, detected_tactic: str, confidence: float) -> str:
    """Return an actionable AI strategy advice string tailored for high-confidence
    detections and closing-phase triggers (price anchors).

    Behavior:
    - Scans the last two turns for price markers and sets a closing-phase flag.
    - If the top tactic is an "Elicit-Preference" (or similar) with
      confidence > 0.60, return a deterministic, tactical anchor-and-justify
      recommendation (no API call).
    - Otherwise, call the configured LLM provider with injected context
      (tactic name + confidence + closing hint) and return a sanitized,
      bullet-formatted response.
    """

    # --- Helper: detect price markers in most recent two turns
    lines = [l.strip() for l in transcript.splitlines() if l.strip()]
    last_two = lines[-2:] if len(lines) >= 2 else lines

    price_pattern = re.compile(
        r"(\b\d{1,3}(?:[.,]\d+)?k\b|\b(?:₹|Rs\.?|INR)\s?\d[\d,\.]*\b|\bbudget\b)",
        re.IGNORECASE,
    )
    found_price = None
    for ln in reversed(last_two):
        m = price_pattern.search(ln)
        if m:
            found_price = m.group(0)
            break

    # --- Normalise tactic name checks
    t_lower = (detected_tactic or "").lower()
    is_elicit = "elicit" in t_lower or "elicit-pref" in t_lower or "elicit-preference" in t_lower

    # --- Deterministic handling for high-confidence Elicit-Preference
    if is_elicit and confidence and confidence > 0.60:
        if found_price:
            advice = (
                f"- The counterpart is fishing for your floor. Do not lower your {found_price} "
                "anchor. Instead, ask what specific value (condition, delivery, or speed) "
                "would make this price acceptable to them."
            )
        else:
            advice = (
                "- The counterpart is fishing for your floor. Anchor firmly to your opening "
                "offer and ask which specific condition would make it acceptable (delivery, "
                "warranty, timeline)."
            )
        return advice

    # --- Otherwise, call the LLM with injected contextual vars
    closing_flag = bool(found_price)

    # Build an augmented user message that includes tactic + confidence
    augmented_user = (
        _build_user_message(transcript, detected_tactic)
        + "\n\n"
        + f"Highest detected tactic: {detected_tactic} (confidence: {confidence:.2f}).\n"
    )
    if closing_flag:
        augmented_user += (
            f"Recent price indicator found in the transcript: {found_price}. "
            "Transition advice from exploration to closing: prefer closing-phase actions "
            "that use the exact numbers mentioned.\n"
        )

    augmented_user += (
        "Tone: Direct, Tactical, and Executive. Respond only with 1-3 Markdown bullets, "
        "each actionable and immediately speakable by a seller. No philosophical fluff."
    )

    # Resolve API key (same logic as public API)
    fallback = _FALLBACK_TIPS.get(detected_tactic, _DEFAULT_FALLBACK)
    key_env_name = "GEMINI_API_KEY" if LLM_PROVIDER == "gemini" else "OPENAI_API_KEY"
    api_key = _HARDCODED_KEY

    if not api_key:
        try:
            import streamlit as st

            api_key = st.secrets.get(key_env_name, "")
        except Exception:
            api_key = ""

    if not api_key:
        api_key = os.environ.get(key_env_name, "")

    if not api_key:
        logger.warning(
            "[llm_coach] No API key found for provider '%s' (injected advice). Using static fallback.",
            LLM_PROVIDER,
        )
        return fallback

    try:
        if LLM_PROVIDER == "gemini":
            # reuse the gemini caller but pass augmented_user as the prompt
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                system_instruction=_SYSTEM_PROMPT,
            )
            response = model.generate_content(
                augmented_user,
                generation_config=genai.types.GenerationConfig(max_output_tokens=150, temperature=0.6),
            )
            raw = response.text.strip()

        else:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": augmented_user},
                ],
                max_tokens=150,
                temperature=0.6,
            )
            raw = response.choices[0].message.content.strip()

        return _sanitize_llm_response(raw)

    except Exception as exc:
        logger.error("[llm_coach] get_ai_strategy_advice LLM call failed: %s", exc)
        return fallback

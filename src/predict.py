import torch
from transformers.models.xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer

try:
    from model import TacticSenseModel
except ImportError:
    from src.model import TacticSenseModel

# ── Label / category constants ────────────────────────────────────────────────
LABELS = [
    "Small Talk", "Empathy", "No-need", "Vouching",
    "Non-hostile need", "Firmness", "Elicit-Pref", "Walk Away",
]

CATEGORIES = {
    "Pressure Tactic":       ["Firmness", "Walk Away"],
    "Rapport Building":      ["Small Talk", "Empathy"],
    "Information/Authority": ["Vouching", "Elicit-Pref"],
    "Constraint Framing":    ["No-need", "Non-hostile need"],
}

_MAX_TOKENS = 512


# ── Input formatter ───────────────────────────────────────────────────────────

def _build_focused_input(turns: list) -> tuple:
    """
    Build a (context_str, current_str) pair for the tokenizer.

    XLM-RoBERTa sentence-pair encoding layout:
        <s> [context] </s></s> [current] </s>

    WHY THIS WORKS
    ──────────────
    The double </s></s> inter-segment boundary is a hard structural marker
    that the pretrained attention heads already associate with a topic shift
    (it is the same layout used during XLM-RoBERTa's NSP-style pretraining).
    Because the two strings share the same self-attention matrix, every
    context token can attend to every current-turn token and vice versa —
    unlike the old approach that tokenized each of the 5 turns separately,
    where cross-turn attention was architecturally impossible.

    Temporal role tags ([T1]..[T4], [CURRENT]) are prepended so positional
    ordering is preserved after the context turns are collapsed into one string.
    """
    *ctx_turns, final_turn = turns
    ctx_parts  = [f"[T{i+1}] {t.strip()}" for i, t in enumerate(ctx_turns)]
    ctx_str    = " ".join(ctx_parts) if ctx_parts else ""
    curr_str   = f"[CURRENT] {final_turn.strip()}"
    return ctx_str, curr_str


# ── Weighted mean-pool ────────────────────────────────────────────────────────

def _last_turn_weighted_pool(hidden_states, input_ids, tokenizer, boost=3.0):
    """
    Weighted mean-pool over all non-padding, non-special tokens.

    Tokens belonging to the CURRENT TURN segment receive `boost`x more weight,
    biasing the pooled representation toward the final utterance's semantics
    without any retraining.

    CURRENT segment boundary: immediately after the second </s> (sep_id = 2)
    in the encoded sequence.

    Args:
        hidden_states : (1, seq_len, hidden_dim)
        input_ids     : (1, seq_len)
        tokenizer     : XLMRobertaTokenizer
        boost         : weight multiplier for current-turn tokens (default 3.0)
    Returns:
        pooled        : (1, hidden_dim)
    """
    sep_id = tokenizer.sep_token_id    # 2
    pad_id = tokenizer.pad_token_id    # 1
    cls_id = tokenizer.cls_token_id    # 0
    ids    = input_ids[0]              # (seq_len,)

    sep_positions = (ids == sep_id).nonzero(as_tuple=True)[0]
    # Second </s> is the inter-segment divider; current-turn tokens follow it
    current_start = int(sep_positions[1]) + 1 if len(sep_positions) >= 2 else 0

    weights = torch.ones(ids.size(0), device=hidden_states.device)
    weights[current_start:] = boost
    # Zero-out special / padding tokens — they should not influence the pool
    weights[(ids == pad_id) | (ids == sep_id) | (ids == cls_id)] = 0.0

    weights      = weights.unsqueeze(0).unsqueeze(-1)         # (1, seq_len, 1)
    weighted_sum = (hidden_states * weights).sum(dim=1)        # (1, hidden)
    weight_norm  = weights.sum(dim=1).clamp(min=1e-9)          # (1, 1)
    return weighted_sum / weight_norm


# ── Main inference function ───────────────────────────────────────────────────

def predict_tactic(turns_list, ids_list, model, tokenizer):
    """
    Classify the negotiation tactic of the latest turn.

    Sequence-pair encoding layout used by XLM-RoBERTa:
        <s> context </s></s> target </s>

    - target_turn   : the final utterance being classified
    - context_string: all preceding turns joined as one string
                      (empty string when turns_list has only one element,
                       which still triggers proper single-sequence encoding)
    """
    model.eval()

    # ── Sequence-pair split ──────────────────────────────────────────────────
    target_turn    = turns_list[-1]
    context_string = " ".join(turns_list[:-1]) if len(turns_list) > 1 else ""

    # ── XLM-RoBERTa sequence-pair tokenization ───────────────────────────────
    inputs = tokenizer(
        text=context_string,
        text_pair=target_turn,
        return_tensors="pt",
        max_length=_MAX_TOKENS,
        truncation="only_first",
        padding=True,
    )
    input_ids      = inputs["input_ids"].unsqueeze(1)       # (1, 1, seq_len)
    attention_mask = inputs["attention_mask"].unsqueeze(1)  # (1, 1, seq_len)
    speaker_ids    = torch.tensor([[ids_list[-1]]], dtype=torch.long)

    # ── OOM guard: no gradient tracking during inference ────────────────────
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            speaker_ids=speaker_ids,
        )

    logits = logits.view(-1)                              # (num_labels,)
    probabilities = torch.sigmoid(logits).squeeze().tolist()  # plain Python list of floats

    return LABELS, probabilities


# ── Verdict formatter ─────────────────────────────────────────────────────────

def get_final_verdict(labels, scores, threshold=0.50):
    """
    Interprets the scores to provide a human-readable manipulation analysis.
    """
    detected = [
        (labels[i], float(scores[i]))
        for i in range(len(labels))
        if float(scores[i]) > threshold
    ]
    detected.sort(key=lambda x: x[1], reverse=True)

    if not detected:
        top_idx = max(range(len(scores)), key=lambda i: scores[i])
        return (
            f"SUBTLE: Weak signal of {labels[top_idx]} "
            f"(Score: {float(scores[top_idx]):.2f})"
        )

    primary_tactic, score = detected[0]
    final_cat = "General Negotiation"
    for cat, tactic_list in CATEGORIES.items():
        if primary_tactic in tactic_list:
            final_cat = cat
            break

    return (
        f"FINAL VERDICT: {final_cat.upper()} "
        f"detected via '{primary_tactic}' ({score:.2f})"
    )


# ── Smoke-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    model     = TacticSenseModel()

    example_turns = [
        "Hey there! I saw your listing for the iPhone 15. Is it still available?",
        "Yes, it is! I've had a few people message me, but no one has come to see it yet.",
        "Great. I've been looking for this specific color for a while now. Is the price flexible at all?",
        "I understand you're looking for a deal, but I've kept it in a case since day one.",
        "I really appreciate you keeping it in good shape, but $600 is my absolute limit today.",
    ]
    example_speaker_ids = [0, 1, 0, 1, 0]

    labels, scores = predict_tactic(
        example_turns, example_speaker_ids, model, tokenizer
    )
    verdict = get_final_verdict(labels, scores)

    print("\n" + "=" * 55)
    print("  TACTICSENSE LIVE ANALYSIS ENGINE")
    print("=" * 55)
    for label, prob in sorted(zip(labels, scores), key=lambda x: x[1], reverse=True):
        prob   = float(prob)
        bar    = "█" * int(prob * 20)
        marker = ">>" if prob > 0.50 else "  "
        print(f"{marker} {label:<20} {prob:.4f}  {bar}")
    print("-" * 55)
    print(verdict)
    print("=" * 55)

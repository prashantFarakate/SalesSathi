"""
LLM wrapper for lead scoring and follow-up drafting.

Uses Groq (via LangChain's ChatGroq) — the same provider/model as the RAG
answerer — so the whole app runs on one LLM key (GROQ_API_KEY). If the key is
missing or the call fails (e.g. no network in the room you're demoing in), both
functions fall back to a transparent rule-based heuristic. The demo therefore
always runs, and the graceful degradation is itself a reasonable product
decision to talk through.

Each function reports which path produced the result via a "mode" field:
"llm" when Groq answered, "heuristic" when the fallback ran.
"""

import json
import os

# Groq model, shared with the RAG answerer. Overridable via env.
MODEL_ID = os.getenv("SALESSAATHI_MODEL", "openai/gpt-oss-120b")

# Fallback sender used only if the caller doesn't pass one. In the app the
# signed-in user's name is passed down, so the draft is always "from" whoever
# is logged in — there is no second hardcoded identity.
DEFAULT_SENDER = "the Tata Motors sales team"
DEALERSHIP = "Tata Motors Pune Central"


def _get_llm():
    """Return a ChatGroq client, or None if no key / library unavailable."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.strip() in ("", "your_groq_api_key_here"):
        return None
    try:
        from langchain_groq import ChatGroq

        return ChatGroq(model=MODEL_ID, temperature=0.2, api_key=api_key)
    except Exception:
        return None


def _invoke(llm, prompt: str) -> str:
    """Send a single-user-message prompt to Groq and return the text."""
    from langchain_core.messages import HumanMessage

    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def score_lead(lead: dict, in_stock: bool) -> dict:
    """Returns {"score": int, "tier": "hot"|"warm"|"cold", "reasons": [str], "mode": "llm"|"heuristic"}"""
    llm = _get_llm()
    if llm:
        prompt = f"""You are a sales-lead scoring assistant for a car dealership.
Score this lead from 0-100 on likelihood to convert to a test drive within a week,
and classify it as "hot" (>=75), "warm" (55-74), or "cold" (<55).

Lead: {json.dumps(lead)}
Requested model currently in stock at nearest dealership: {in_stock}

Respond with ONLY valid JSON, no other text, in this exact shape:
{{"score": <int>, "tier": "<hot|warm|cold>", "reasons": ["<short reason>", ...]}}"""
        try:
            text = _invoke(llm, prompt).strip().strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
            parsed = json.loads(text)
            parsed["mode"] = "llm"
            return parsed
        except Exception:
            pass  # fall through to heuristic

    return {**_heuristic_score(lead, in_stock), "mode": "heuristic"}


def _heuristic_score(lead: dict, in_stock: bool) -> dict:
    score = 40
    reasons = []
    if lead.get("engagement") == "high":
        score += 20
        reasons.append("High engagement on recent campaign touchpoints (+20)")
    elif lead.get("engagement") == "medium":
        score += 10
        reasons.append("Moderate engagement signal (+10)")
    if lead.get("testDriveRequested"):
        score += 20
        reasons.append("Test drive explicitly requested (+20)")
    if "ev" in lead.get("model", "").lower():
        score += 5
        reasons.append("EV interest — aligns with current regional push (+5)")
    if in_stock:
        score += 10
        reasons.append("Requested variant is in stock at nearest dealership (+10)")
    else:
        reasons.append("Requested variant currently low stock — flag for allocation")
    score = min(score, 98)
    tier = "hot" if score >= 75 else "warm" if score >= 55 else "cold"
    return {"score": score, "tier": tier, "reasons": reasons}


def _ensure_signoff_on_new_line(text: str, signoff: str) -> str:
    """Guarantee the sign-off sits on its own line, with a blank line before it.

    The model sometimes trails the sign-off straight after the last sentence
    ("...works. — Prashant..."). Split it back onto its own line so the UI
    renders it as a separate line."""
    text = text.strip()
    # If the exact sign-off is already present, split the body before it.
    idx = text.find(signoff)
    if idx != -1:
        body = text[:idx].rstrip()
        return f"{body}\n\n{signoff}"
    # Otherwise, split on the sign-off's dash marker if the model used one.
    dash = signoff.split(",")[0]  # e.g. "— Prashant Farakate"
    idx = text.find(dash)
    if idx != -1:
        body = text[:idx].rstrip()
        return f"{body}\n\n{signoff}"
    # No sign-off found at all — append ours.
    return f"{text}\n\n{signoff}"


def draft_followup(lead: dict, scheme_context: str, slot: str, sender_name: str | None = None) -> dict:
    """Draft a follow-up message that is 'from' the signed-in sales executive.

    `sender_name` is the logged-in user's name, passed down from the UI so the
    draft's identity always matches whoever is using the app. Returns
    {"message": str, "mode": "llm"|"heuristic"}.
    """
    sender = (sender_name or "").strip() or DEFAULT_SENDER
    signoff = f"— {sender}, {DEALERSHIP}"

    llm = _get_llm()
    if llm:
        prompt = f"""Write a short, warm WhatsApp follow-up message to a car-dealership lead,
in plain conversational English.

The message is sent BY the sales executive named "{sender}". This is the sender's
real name — use it as the sender and nothing else. Do NOT invent or substitute any
other sales-rep name.

Mention the test-drive slot and, if relevant, the finance scheme context below.
Keep it under 60 words. Do not invent facts not given here.

End the message with the sign-off on its OWN separate line, preceded by a blank
line, exactly like this (keep the line break):

{signoff}

Lead: {json.dumps(lead)}
Test-drive slot held: {slot}
Relevant finance/scheme context: {scheme_context}

Respond with ONLY the message text, nothing else."""
        try:
            text = _invoke(llm, prompt).strip()
            if text:
                text = _ensure_signoff_on_new_line(text, signoff)
                return {"message": text, "mode": "llm"}
        except Exception:
            pass  # fall through to heuristic

    return {"message": _heuristic_draft(lead, scheme_context, slot, signoff), "mode": "heuristic"}


def _heuristic_draft(lead: dict, scheme_context: str, slot: str, signoff: str) -> str:
    first_name = lead["name"].split(" ")[0]
    scheme_line = (
        f" Also, {scheme_context}" if scheme_context else " Let me know your budget and I'll share the best current offer."
    )
    return (
        f"Hi {first_name}, thanks for your interest in the {lead['model']}! "
        f"I've held a test drive slot for {slot}, just confirm if that works."
        f"{scheme_line}\n\n{signoff}"
    )

"""
LLM wrapper for lead scoring and follow-up drafting.

If ANTHROPIC_API_KEY is set, both functions call the real Claude API and ask
for structured JSON output. If it isn't set (or the call fails, e.g. no
network in the room you're demoing in), both functions fall back to a
transparent rule-based heuristic -- so the demo always runs, and you can
show the interviewer exactly how it degrades gracefully, which is itself a
reasonable product decision to talk through.
"""

import json
import os

MODEL = os.environ.get("SALESSAATHI_MODEL", "claude-sonnet-5")


def _get_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic

        return anthropic.Anthropic(api_key=api_key)
    except Exception:
        return None


def score_lead(lead: dict, in_stock: bool) -> dict:
    """Returns {"score": int, "tier": "hot"|"warm"|"cold", "reasons": [str], "mode": "llm"|"heuristic"}"""
    client = _get_client()
    if client:
        prompt = f"""You are a sales-lead scoring assistant for a car dealership.
Score this lead from 0-100 on likelihood to convert to a test drive within a week,
and classify it as "hot" (>=75), "warm" (55-74), or "cold" (<55).

Lead: {json.dumps(lead)}
Requested model currently in stock at nearest dealership: {in_stock}

Respond with ONLY valid JSON, no other text, in this exact shape:
{{"score": <int>, "tier": "<hot|warm|cold>", "reasons": ["<short reason>", ...]}}"""
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            parsed = json.loads(text.strip().strip("`").removeprefix("json").strip())
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


def draft_followup(lead: dict, scheme_context: str, slot: str) -> dict:
    """Returns {"message": str, "mode": "llm"|"heuristic"}"""
    client = _get_client()
    if client:
        prompt = f"""Write a short, warm WhatsApp follow-up message from a Tata Motors
sales executive named Rohan to a lead, in plain conversational English.
Mention the test-drive slot and, if relevant, the finance scheme context below.
Keep it under 60 words. Do not invent facts not given here.

Lead: {json.dumps(lead)}
Test-drive slot held: {slot}
Relevant finance/scheme context: {scheme_context}

Respond with ONLY the message text, nothing else."""
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
            if text:
                return {"message": text, "mode": "llm"}
        except Exception:
            pass  # fall through to heuristic

    return {"message": _heuristic_draft(lead, scheme_context, slot), "mode": "heuristic"}


def _heuristic_draft(lead: dict, scheme_context: str, slot: str) -> str:
    first_name = lead["name"].split(" ")[0]
    scheme_line = (
        f" Also, {scheme_context}" if scheme_context else " Let me know your budget and I'll share the best current offer."
    )
    return (
        f"Hi {first_name}, thanks for your interest in the {lead['model']}! "
        f"I've held a test drive slot for {slot}, just confirm if that works."
        f"{scheme_line}\n\n— Rohan, Tata Motors Pune Central"
    )

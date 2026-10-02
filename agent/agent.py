"""
SalesSaathi Agent
------------------
Orchestrates the lead-qualification pipeline:

  1. get_lead_from_crm      (real MCP tool call, over stdio, to mcp_server/server.py)
  2. check_inventory        (real MCP tool call)
  3. retrieve finance/spec context from the RAG knowledge base
  4. score the lead (LLM if ANTHROPIC_API_KEY set, else heuristic fallback)
  5. schedule_test_drive    (real MCP tool call)
  6. draft a follow-up message (LLM or heuristic fallback)
  7. STOP and wait for human approval before anything would be "sent"

The MCP server is launched as a subprocess and spoken to over stdio using the
official `mcp` Python SDK client -- this is the real protocol, not a mock.
"""

import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.retriever import retrieve, build_index  # noqa: E402
from agent.llm import score_lead, draft_followup  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_SCRIPT = os.path.join(BASE_DIR, "mcp_server", "server.py")


def _tool_result_to_obj(result):
    """MCP tool results come back as content blocks; our tools return JSON-able
    dicts, which FastMCP serializes as a text block. Parse it back to a dict."""
    for block in result.content:
        if getattr(block, "type", "") == "text":
            try:
                return json.loads(block.text)
            except json.JSONDecodeError:
                return block.text
    return None


async def run_pipeline(lead_id: str, on_step=print, sender_name: str | None = None):
    """Runs the full pipeline for one lead. `on_step(step_name, detail)` is
    called after each step so a CLI or UI can render progress live.

    `sender_name` is the signed-in sales executive; the drafted follow-up is
    written as being from them, so the message identity matches the logged-in
    user instead of a hardcoded name."""
    server_params = StdioServerParameters(command=sys.executable, args=[SERVER_SCRIPT])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Step 1
            raw_lead = await session.call_tool("get_lead_from_crm", {"lead_id": lead_id})
            lead = _tool_result_to_obj(raw_lead)
            if not lead or "error" in lead:
                on_step("error", f"lead {lead_id} not found in CRM")
                return None
            on_step("get_lead_from_crm", lead)

            # Step 2
            raw_inv = await session.call_tool(
                "check_inventory", {"model": lead["model"], "region": lead["region"]}
            )
            inventory = _tool_result_to_obj(raw_inv)
            on_step("check_inventory", inventory)

            # Step 3: RAG retrieval for scheme + spec context
            hits = retrieve(f"finance scheme {lead['region']} {lead['model']}")
            scheme_context = ""
            if hits:
                # Pull the "Benefit" / "Valid till" lines out of the top chunk for the draft message
                top = hits[0]
                lines = [
                    l for l in top.text.splitlines()
                    if l.strip().startswith("- Benefit") or l.strip().startswith("- Valid")
                ]
                if lines:
                    scheme_context = "; ".join(l.lstrip("- ").strip() for l in lines) + f" (source: {top.source})"
            on_step("rag_retrieve", {"query_hits": [h.source for h in hits], "scheme_context": scheme_context})

            # Step 4: score
            score_info = score_lead(lead, inventory.get("inStock", False))
            on_step("score_lead", score_info)

            # Step 5
            raw_slot = await session.call_tool("schedule_test_drive", {"lead_id": lead_id})
            slot_info = _tool_result_to_obj(raw_slot)
            on_step("schedule_test_drive", slot_info)

            # Step 6: draft
            draft_info = draft_followup(
                lead, scheme_context, slot_info.get("slot", "an upcoming slot"),
                sender_name=sender_name,
            )
            on_step("draft_followup", draft_info)

            return {
                "lead": lead,
                "inventory": inventory,
                "score": score_info,
                "slot": slot_info,
                "draft": draft_info,
            }


async def ask_knowledge_base(query: str):
    """Standalone RAG query, no MCP/agent involved -- used by the CLI's
    'ask' mode to demo the Knowledge Assistant module on its own.

    Returns a list of RetrievedChunk (may be empty if nothing clears the
    confidence threshold, i.e. "I don't have that")."""
    build_index()  # no-op if the index already exists
    return retrieve(query)

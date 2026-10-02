"""
SalesSaathi — Streamlit demo UI

Two tabs:
  - Phase 1: RAG Product & Finance Q&A — grounded, cited answers that refuse to
    guess when the knowledge base has no confident match.
  - Phase 2: Lead Agent — an agentic pipeline over a real MCP tool layer
    (mock CRM / DMS / calendar) that qualifies a lead, scores it, and drafts a
    human-approved follow-up. No autonomous send: human-in-the-loop by design.

Run locally:
    streamlit run app.py
"""

import asyncio
import json
import os
import threading

# On corporate networks with an SSL-inspecting proxy (e.g. Zscaler), make Python
# trust the OS certificate store so outbound HTTPS to the LLM works. Safe no-op
# elsewhere and when truststore isn't installed (e.g. Streamlit Cloud).
try:
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass

import streamlit as st

from rag.retriever import build_index
from rag.answerer import answer, MODEL_ID
from agent.agent import run_pipeline

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

MAX_QUESTIONS = 12  # per-session cap, protects the shared demo's LLM quota

SAMPLE_QUESTIONS = [
    "What is the range and charging time of the Nexon EV Long Range?",
    "Compare the Nexon EV and Punch EV on range and price.",
    "Is there any EV finance scheme in Maharashtra this quarter?",
    "When was Ratan Tata born?",
]

st.set_page_config(page_title="SalesSaathi", page_icon="🚗", layout="wide")


@st.cache_resource(show_spinner=False)
def _ensure_index() -> int:
    """Build the vector index once per server session."""
    return build_index()


# --------------------------------------------------------------------------- #
# Phase 2 — Lead Agent helpers
# --------------------------------------------------------------------------- #

LEADS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mcp_server", "mock_data", "leads.json"
)


@st.cache_data(show_spinner=False)
def _load_leads() -> list[dict]:
    """Load the mock CRM leads so the UI can offer a picker."""
    try:
        with open(LEADS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _run_agent_pipeline(lead_id: str) -> tuple[list[tuple[str, dict]], dict | None]:
    """Run the async agent pipeline from Streamlit safely.

    run_pipeline() is async and launches the MCP server as a subprocess over
    stdio. Streamlit runs its own event loop on the main thread, so we execute
    the pipeline in a dedicated worker thread with its own fresh event loop to
    avoid "event loop already running" / loop-ownership errors. Each step the
    pipeline emits is collected into a list so we can render it after the run.
    """
    steps: list[tuple[str, dict]] = []
    result_box: dict = {}
    error_box: dict = {}

    def _collect(step, detail):
        steps.append((step, detail))

    def _worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_box["result"] = loop.run_until_complete(
                run_pipeline(lead_id, on_step=_collect)
            )
        except Exception as exc:  # noqa: BLE001 - surface, never crash the demo
            error_box["error"] = str(exc)
        finally:
            loop.close()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join()

    if error_box:
        steps.append(("error", {"message": error_box["error"]}))
    return steps, result_box.get("result")


def _tier_color(tier: str) -> str:
    return {"hot": "#2e7d32", "warm": "#ed9b00", "cold": "#1976d2"}.get(tier, "#777")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown("## **SalesSaathi**")
    st.divider()
    st.header("About SalesSaathi")
    st.markdown(
        "An AI co-pilot that helps Tata Motors sales executives answer customer "
        "**product and finance questions** accurately, using a "
        "**Retrieval-Augmented Generation (RAG)** pipeline grounded in approved "
        "product documents. It cites its sources and refuses to guess."
    )

    st.divider()
    st.subheader("How it works")
    st.markdown(
        "1. Your question is embedded and matched against the knowledge base.\n"
        "2. The most relevant fact sheets are retrieved.\n"
        "3. An LLM answers **only** from that retrieved context, with citations.\n"
        "4. If nothing relevant is found, it says so — no hallucinated specs."
    )

    st.divider()
    st.caption(f"Model: `{MODEL_ID}`  ·  Retrieval: local MiniLM embeddings + ChromaDB")
    st.caption(
        "⚠️ Specs, prices, and finance schemes here are **illustrative demo "
        "placeholders**, not official Tata Motors figures."
    )


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.title("🚗 SalesSaathi")
st.caption("AI co-pilot for Tata Motors sales teams")

with st.spinner("Loading the product knowledge base..."):
    chunk_count = _ensure_index()

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

if "messages" not in st.session_state:
    st.session_state.messages = []
if "asked" not in st.session_state:
    st.session_state.asked = 0
if "pending" not in st.session_state:
    st.session_state.pending = None
if "agent_steps" not in st.session_state:
    st.session_state.agent_steps = None
if "agent_result" not in st.session_state:
    st.session_state.agent_result = None
if "agent_lead_id" not in st.session_state:
    st.session_state.agent_lead_id = None


def _queue(question: str) -> None:
    """Queue a question (from a sample button or the chat box) to be answered."""
    st.session_state.pending = question


# --------------------------------------------------------------------------- #
# Tabs: Phase 1 (Q&A) and Phase 2 (Lead Agent)
# --------------------------------------------------------------------------- #

tab_qa, tab_agent = st.tabs(["💬 Product Q&A", "🤖 Lead Agent"])

with tab_qa:
    st.caption("Phase 1 · Grounded Product & Finance Q&A — cites sources, refuses to guess")

    # --------------------------------------------------------------------------- #
    # Sample questions
    # --------------------------------------------------------------------------- #

    st.markdown("**Try one of these:**")
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for col, q in zip(cols, SAMPLE_QUESTIONS):
        with col:
            st.button(q, key=f"sample::{q}", use_container_width=True, on_click=_queue, args=(q,))

    # ----------------------------------------------------------------------- #
    # Render prior conversation
    # ----------------------------------------------------------------------- #

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.caption("Sources: " + ", ".join(msg["sources"]))
            if msg.get("chunks"):
                with st.expander("🔍 Retrieved context (what the answer is grounded in)"):
                    for c in msg["chunks"]:
                        st.markdown(f"**{c['source']}** · similarity {c['score']:.2f}")
                        st.text(c["text"][:600] + ("..." if len(c["text"]) > 600 else ""))

    # ----------------------------------------------------------------------- #
    # Chat input
    # ----------------------------------------------------------------------- #

    typed = st.chat_input("Ask about Tata models, specs, range, charging, price, warranty, or finance...")
    if typed:
        _queue(typed)

    question = st.session_state.pending
    if question:
        st.session_state.pending = None

        if st.session_state.asked >= MAX_QUESTIONS:
            st.warning(
                f"This shared demo is capped at {MAX_QUESTIONS} questions per session "
                "to protect its API quota. Refresh the page to start a new session."
            )
        else:
            st.session_state.asked += 1
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving product info and answering..."):
                    try:
                        result = answer(question)
                        text = result.text
                        sources = result.sources
                        chunks = [
                            {"source": c.source, "score": c.score, "text": c.text}
                            for c in result.chunks
                        ]
                        mode = result.mode
                    except Exception as exc:  # noqa: BLE001 - never crash the demo
                        text = (
                            "Something went wrong while answering. Please try again in a "
                            "moment."
                        )
                        sources, chunks, mode = [], [], "error"

                st.markdown(text)
                if mode == "no_context":
                    st.info("No confident match in the knowledge base — the assistant declined to guess.")
                if sources:
                    st.caption("Sources: " + ", ".join(sources))
                if chunks:
                    with st.expander("🔍 Retrieved context (what the answer is grounded in)"):
                        for c in chunks:
                            st.markdown(f"**{c['source']}** · similarity {c['score']:.2f}")
                            st.text(c["text"][:600] + ("..." if len(c["text"]) > 600 else ""))

            st.session_state.messages.append(
                {"role": "assistant", "content": text, "sources": sources, "chunks": chunks}
            )


# --------------------------------------------------------------------------- #
# Phase 2 — Lead Agent tab
# --------------------------------------------------------------------------- #

with tab_agent:
    st.caption(
        "Phase 2 · Agentic lead pipeline over a real **MCP** tool layer "
        "(mock CRM / DMS / calendar) — scores the lead and drafts a follow-up. "
        "Nothing is sent without human approval."
    )

    leads = _load_leads()
    if not leads:
        st.error("Couldn't load mock leads from mcp_server/mock_data/leads.json.")
    else:
        lead_labels = {
            f"{l['id']} · {l['name']} · {l['model']} ({l['engagement']} engagement)": l["id"]
            for l in leads
        }
        picked_label = st.selectbox("Pick a lead from the CRM", list(lead_labels.keys()))
        picked_id = lead_labels[picked_label]

        run_col, note_col = st.columns([1, 3])
        with run_col:
            run_clicked = st.button("▶ Run agent", type="primary", use_container_width=True)
        with note_col:
            st.caption(
                "Pipeline: get lead → check inventory → retrieve finance scheme → "
                "score → hold test-drive slot → draft message → **stop for approval**."
            )

        if run_clicked:
            with st.spinner("Running the agent over the MCP tool layer..."):
                steps, result = _run_agent_pipeline(picked_id)
            st.session_state.agent_steps = steps
            st.session_state.agent_result = result
            st.session_state.agent_lead_id = picked_id

        steps = st.session_state.agent_steps
        result = st.session_state.agent_result

        if steps is not None and st.session_state.agent_lead_id == picked_id:
            step_map = {name: detail for name, detail in steps}

            if "error" in step_map:
                st.error(f"Pipeline error: {step_map['error'].get('message', 'unknown error')}")

            # Live step trace
            with st.expander("🧰 MCP tool trace (each step the agent ran)", expanded=False):
                for name, detail in steps:
                    st.markdown(f"**→ {name}**")
                    st.code(json.dumps(detail, indent=2, ensure_ascii=False), language="json")

            if result:
                lead = result["lead"]
                inv = result["inventory"]
                score = result["score"]
                slot = result["slot"]
                draft = result["draft"]

                # Lead + inventory summary
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Lead")
                    st.markdown(
                        f"**{lead['name']}**  ·  {lead['id']}\n\n"
                        f"- Model: **{lead['model']}**\n"
                        f"- Region: {lead['region']}\n"
                        f"- Budget: {lead['budget']}\n"
                        f"- Source: {lead['source']}  ·  Engagement: {lead['engagement']}\n"
                        f"- Test drive requested: {'Yes' if lead.get('testDriveRequested') else 'No'}"
                    )
                with c2:
                    st.subheader("Inventory (DMS)")
                    if inv.get("inStock"):
                        st.success(f"In stock — {inv.get('units', '?')} unit(s) at nearest dealership")
                    else:
                        st.warning("Not in stock at nearest dealership — flag for allocation")

                # Score
                st.subheader("Lead score")
                color = _tier_color(score["tier"])
                st.markdown(
                    f"<span style='font-size:1.6rem;font-weight:700;color:{color}'>"
                    f"{score['score']}/100 — {score['tier'].upper()}</span>",
                    unsafe_allow_html=True,
                )
                for r in score["reasons"]:
                    st.markdown(f"- {r}")
                st.caption(f"Scored via: {score['mode']}")

                # Draft + human approval (mock)
                st.subheader("Draft follow-up (needs approval)")
                st.info(draft["message"])
                st.caption(f"Drafted via: {draft['mode']}  ·  Suggested slot: {slot.get('slot', 'n/a')}")

                a1, a2 = st.columns(2)
                with a1:
                    if st.button("✅ Approve & queue to send", use_container_width=True):
                        st.success("Approved — queued to send via WhatsApp (demo: not actually sent).")
                with a2:
                    if st.button("✏️ Needs edits", use_container_width=True):
                        st.warning("Marked for manual editing. Nothing sent.")

                st.caption(
                    "🔒 Human-in-the-loop by design: in Phase 1/2 the agent never sends "
                    "a customer message autonomously."
                )

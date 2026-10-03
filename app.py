"""
SalesSaathi — Streamlit demo UI

A dealer sales co-pilot with three views (sidebar nav):
  - Knowledge Assistant : grounded, cited Product & Finance Q&A (RAG). Refuses
    to guess when the knowledge base has no confident match.
  - Lead Pipeline       : an agentic pipeline over a real MCP tool layer
    (mock CRM / DMS / calendar) that qualifies a lead, scores it, and drafts a
    human-approved follow-up. No autonomous send: human-in-the-loop by design.
  - Integration Log     : a live trace of the MCP tool calls the agent made.

Run locally:
    streamlit run app.py
"""

import asyncio
import json
import os
import threading
import time

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

# Presentational knowledge-base listing (mock dates) for the KB panel.
KB_DOCS = [
    ("Nexon EV", "Aug 2026"),
    ("Curvv EV", "Jul 2026"),
    ("Punch.ev", "Jun 2026"),
    ("Competitor Note", "Jun 2026"),
    ("Finance Schemes", "Sep 2026"),
]

USER_NAME = "Prashant Farakate"
USER_ROLE = "Sales Executive"
USER_SITE = "Pune Central Showroom"

st.set_page_config(page_title="SalesSaathi", page_icon="🚗", layout="wide")


# --------------------------------------------------------------------------- #
# Styling — dark, polished shell to match the product mock
# --------------------------------------------------------------------------- #

st.markdown(
    """
    <style>
      /* App background */
      .stApp { background-color: #0e1117; }

      /* Sidebar */
      section[data-testid="stSidebar"] { background-color: #0b0e14; border-right: 1px solid #1c2230; }
      .brand { font-size: 1.35rem; font-weight: 800; color: #f3f4f6; letter-spacing: .2px; }
      .brand-sub { font-size: .78rem; color: #7b8292; margin-top: -4px; margin-bottom: 10px; }
      .side-foot { font-size: .72rem; color: #5b6372; line-height: 1.4; }
      .side-foot b { color: #8a92a3; }

      /* Top bar */
      .topbar { display:flex; justify-content:space-between; align-items:center;
                padding: 2px 0 14px 0; border-bottom: 1px solid #1c2230; margin-bottom: 18px; }
      .topbar .who { color:#8a92a3; font-size:.86rem; }
      .topbar .who b { color:#e5e7eb; }
      .pill { background:#2a1d0e; color:#e39a3b; border:1px solid #6b4a1c;
              padding:4px 12px; border-radius:999px; font-size:.74rem; font-weight:600; }

      /* Headings */
      .page-title { font-size:1.9rem; font-weight:800; color:#f3f4f6; margin-bottom:2px; }
      .page-sub { color:#8a92a3; font-size:.95rem; margin-bottom:18px; max-width:720px; }

      /* Cards / panels */
      .panel { background:#121724; border:1px solid #1f2636; border-radius:12px; padding:18px 20px; }
      .panel-h { font-size:.72rem; letter-spacing:.12em; color:#7b8292; font-weight:700; margin-bottom:10px; }
      .kb-row { display:flex; justify-content:space-between; padding:9px 2px; border-bottom:1px solid #1a2130; }
      .kb-row:last-child { border-bottom:none; }
      .kb-name { color:#e5e7eb; font-size:.92rem; }
      .kb-date { color:#e39a3b; font-size:.82rem; }

      /* Lead table */
      .lead-head { display:grid; grid-template-columns: 2.2fr 1fr 1.3fr 1.8fr 1.1fr 1fr;
                   padding:0 6px 8px 6px; border-bottom:1px solid #1f2636;
                   font-size:.72rem; letter-spacing:.1em; color:#7b8292; font-weight:700; }

      /* Source pills */
      .src { padding:3px 12px; border-radius:999px; font-size:.76rem; font-weight:600; }
      .src.web    { background:#0f2d1c; color:#4ade80; border:1px solid #1f5235; }
      .src.social { background:#2a1030; color:#e879f9; border:1px solid #5b2366; }
      .src.walkin { background:#2a230f; color:#eab308; border:1px solid #5b4a1c; }

      /* Agent run step chips */
      .step-chip { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                   background:#0b0e14; border:1px solid #243049; color:#9fb0c9;
                   border-radius:6px; padding:6px 10px; font-size:.82rem; display:inline-block; }
      .step-chip .hot { color:#4ade80; font-weight:700; }
      .step-num { background:#e07a1a; color:#fff; border-radius:999px; width:22px; height:22px;
                  display:inline-flex; align-items:center; justify-content:center;
                  font-size:.76rem; font-weight:700; }
      .step-title { color:#e5e7eb; font-weight:600; font-size:.92rem; }

      /* Score card */
      .score-big { font-size:2.6rem; font-weight:800; line-height:1; }
      .score-tier { font-size:.72rem; letter-spacing:.1em; color:#7b8292; font-weight:700; }

      /* Draft */
      .draft { background:#121724; border:1px solid #1f2636; border-radius:12px;
               padding:16px 18px; color:#cdd4e0; line-height:1.55; }

      /* LLM status box */
      .llm-box { background:#121724; border:1px solid #1f2636; border-radius:10px; padding:12px 14px; }
      .llm-h { font-size:.68rem; letter-spacing:.14em; color:#7b8292; font-weight:700; margin-bottom:6px; }
      .llm-row { color:#e5e7eb; font-size:.9rem; display:flex; align-items:center; gap:8px; }
      .llm-row code { background:#0b0e14; border:1px solid #243049; border-radius:4px;
                      padding:1px 6px; font-size:.8rem; color:#9fb0c9; }
      .llm-sub { color:#7b8292; font-size:.74rem; margin-top:6px; }
      .dot { width:9px; height:9px; border-radius:999px; display:inline-block; }

      /* KB panel header row with action buttons */
      .kb-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }

      /* Left-align the sample-question prompt chips (buttons).
         Streamlit tags each button container with a class derived from its key
         (st-key-<key>); our sample buttons use keys starting with "sample::". */
      div[class*="st-key-sample"] button { text-align: left !important; justify-content: flex-start !important; }
      div[class*="st-key-sample"] button p { text-align: left !important; width: 100%; }

      /* Terminal log */
      .term { background:#0b0e14; border:1px solid #1f2636; border-radius:10px;
              padding:16px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
              font-size:.82rem; color:#8a92a3; min-height:160px; white-space:pre-wrap; }
      .term .ok { color:#4ade80; }
      .term .tool { color:#60a5fa; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def _ensure_index() -> int:
    """Build the vector index once per server session."""
    return build_index()


# --------------------------------------------------------------------------- #
# Lead Agent helpers
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


def _run_agent_pipeline(lead_id: str, sender_name: str) -> tuple[list[tuple[str, dict]], dict | None]:
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
                run_pipeline(lead_id, on_step=_collect, sender_name=sender_name)
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
    return {"hot": "#4ade80", "warm": "#eab308", "cold": "#60a5fa"}.get(tier, "#9fb0c9")


# --------------------------------------------------------------------------- #
# Knowledge Base dialogs (presentation only — this prototype does not ingest
# uploaded files; it shows how a dealer admin would add approved documents).
# --------------------------------------------------------------------------- #

@st.dialog("Add documents to the knowledge base")
def _kb_add_dialog():
    st.caption(
        "Upload approved brochures, spec sheets or finance-scheme documents. "
        "They'd be chunked, embedded, and indexed so the assistant can cite them."
    )
    st.file_uploader(
        "Drop files here",
        type=["md", "txt", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    st.info(
        "Prototype note: ingestion is disabled in this demo, so files aren't actually "
        "added. In production this writes to the document store and rebuilds the index."
    )
    if st.button("Add to knowledge base", type="primary", use_container_width=True):
        st.success("In production these would now be indexed. (Demo: nothing was added.)")


@st.dialog("Search the knowledge base")
def _kb_search_dialog():
    st.caption("Find a document or passage across the indexed knowledge base.")
    st.text_input("Search", placeholder="e.g. Nexon EV charging time, Q3 finance scheme...",
                  label_visibility="collapsed")
    st.info(
        "Prototype note: this dialog is illustrative. Use the chat on the right to run "
        "a real grounded search over the knowledge base."
    )


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
if "kb_collapsed" not in st.session_state:
    st.session_state.kb_collapsed = False


def _queue(question: str) -> None:
    st.session_state.pending = question
    # Auto-collapse the knowledge base once the user starts chatting, so the
    # conversation gets full width. They can toggle it back on anytime.
    st.session_state.kb_collapsed = True


# --------------------------------------------------------------------------- #
# Sidebar — brand + nav
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown('<div class="brand">SalesSaathi</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">AI co-pilot prototype for sales &amp; marketing</div>',
                unsafe_allow_html=True)
    st.write("")
    page = st.radio(
        "Navigation",
        ["Knowledge Assistant", "Lead Pipeline", "Integration Log"],
        label_visibility="collapsed",
    )
    st.write("")

    # LLM status — honest, live indicator of the provider/model actually used.
    _groq_live = bool(os.getenv("GROQ_API_KEY", "").strip()) and \
        os.getenv("GROQ_API_KEY", "").strip() != "your_groq_api_key_here"
    _dot = "#4ade80" if _groq_live else "#eab308"
    _state = "live" if _groq_live else "fallback (heuristic)"
    st.markdown(
        f'<div class="llm-box">'
        f'<div class="llm-h">LLM</div>'
        f'<div class="llm-row"><span class="dot" style="background:{_dot}"></span>'
        f'<span>Groq · <code>{MODEL_ID}</code></span></div>'
        f'<div class="llm-sub">Status: {_state} · powers Q&amp;A, lead scoring &amp; drafting</div>'
        f'<div class="llm-sub">Embeddings: local MiniLM + ChromaDB</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown(
        '<div class="side-foot"><b>Concept demo.</b> Uses mock CRM/DMS data and '
        "simulated MCP tool calls. No live Tata Motors systems are connected.</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Top bar (shared)
# --------------------------------------------------------------------------- #

st.markdown(
    f"""
    <div class="topbar">
      <div class="who">Signed in as <b>{USER_NAME}</b> · {USER_ROLE} · {USER_SITE}</div>
      <div class="pill">Pilot Prototype</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Loading the product knowledge base..."):
    _ensure_index()


# =========================================================================== #
# PAGE 1 — Knowledge Assistant
# =========================================================================== #

if page == "Knowledge Assistant":
    st.markdown('<div class="page-title">Product &amp; Scheme Knowledge Assistant</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Retrieval-grounded answers over brochures, spec sheets and '
        "finance schemes. Every answer cites its source, and the assistant admits when it "
        "doesn't know rather than guessing.</div>",
        unsafe_allow_html=True,
    )

    # Let the user collapse the Knowledge Base panel to give the chat full width.
    # It also auto-collapses the first time the user sends a message (see _queue).
    def _sync_kb_toggle():
        st.session_state.kb_collapsed = not st.session_state.kb_toggle

    # Keep the toggle widget in sync with kb_collapsed (which _queue may have just
    # flipped), then render it. Writing the widget key before instantiation is the
    # safe way to force its displayed state.
    st.session_state.kb_toggle = not st.session_state.kb_collapsed
    st.toggle("Show knowledge base", key="kb_toggle", on_change=_sync_kb_toggle)
    show_kb = not st.session_state.kb_collapsed

    if show_kb:
        left, right = st.columns([1, 2], gap="large")
        with left:
            kb_rows = "".join(
                f'<div class="kb-row"><span class="kb-name">{name}</span>'
                f'<span class="kb-date">{date}</span></div>'
                for name, date in KB_DOCS
            )
            st.markdown(
                f'<div class="panel"><div class="panel-h">KNOWLEDGE BASE</div>{kb_rows}</div>',
                unsafe_allow_html=True,
            )
            st.write("")
            ba, bs = st.columns(2)
            with ba:
                add_clicked = st.button("＋ Add documents", use_container_width=True)
            with bs:
                search_clicked = st.button("🔍 Search", use_container_width=True)

            if add_clicked:
                _kb_add_dialog()
            if search_clicked:
                _kb_search_dialog()
    else:
        # KB hidden: chat uses the full width.
        right = st.container()

    with right:
        st.caption(
            "Ask a comparison, spec, or finance question below. Try one of the prompts "
            "or write your own."
        )

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

        chip_cols = st.columns(2)
        for i, q in enumerate(SAMPLE_QUESTIONS):
            with chip_cols[i % 2]:
                st.button(q, key=f"sample::{q}", use_container_width=True,
                          on_click=_queue, args=(q,))

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
            with right:
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
                        except Exception:  # noqa: BLE001 - never crash the demo
                            text = ("Something went wrong while answering. "
                                    "Please try again in a moment.")
                            sources, chunks, mode = [], [], "error"

                    st.markdown(text)
                    if mode == "no_context":
                        st.info("No confident match in the knowledge base, so the assistant declined to guess.")
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


# =========================================================================== #
# PAGE 2 — Lead Pipeline
# =========================================================================== #

elif page == "Lead Pipeline":
    st.markdown('<div class="page-title">Lead Qualification &amp; Follow-up Agent</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Every step is human-approved before anything is sent to a '
        "customer. This visualizes what the agent does, not just its final output.</div>",
        unsafe_allow_html=True,
    )

    leads = _load_leads()
    if not leads:
        st.error("Couldn't load mock leads from mcp_server/mock_data/leads.json.")
    else:
        st.markdown(
            '<div class="lead-head"><span>LEAD</span><span>SOURCE</span><span>INTEREST</span>'
            "<span>REGION</span><span>BUDGET</span><span></span></div>",
            unsafe_allow_html=True,
        )

        for lead in leads:
            c = st.columns([2.2, 1, 1.3, 1.8, 1.1, 1])
            with c[0]:
                st.markdown(
                    f"**{lead['name']}**  \n<span style='color:#7b8292;font-size:.8rem'>{lead['id']}</span>",
                    unsafe_allow_html=True,
                )
            with c[1]:
                src = lead["source"]
                st.markdown(f'<span class="src {src}">{src}</span>', unsafe_allow_html=True)
            with c[2]:
                st.markdown(lead["model"])
            with c[3]:
                st.markdown(lead["region"])
            with c[4]:
                st.markdown(lead["budget"])
            with c[5]:
                if st.button("Run agent →", key=f"run::{lead['id']}", use_container_width=True):
                    with st.spinner("Running the agent over the MCP tool layer..."):
                        steps, result = _run_agent_pipeline(lead["id"], USER_NAME)
                    st.session_state.agent_steps = steps
                    st.session_state.agent_result = result
                    st.session_state.agent_lead_id = lead["id"]
                    st.session_state.agent_animate = True
                    # Reset any prior draft edit/sent state for this lead.
                    st.session_state.pop(f"draft_text::{lead['id']}", None)
                    st.session_state[f"draft_editing::{lead['id']}"] = False
                    st.session_state[f"draft_sent::{lead['id']}"] = False
            st.markdown("<hr style='border-color:#1a2130;margin:6px 0'>", unsafe_allow_html=True)

        # ------- Agent run result (inline, below the table) ------- #
        result = st.session_state.agent_result
        steps = st.session_state.agent_steps
        run_id = st.session_state.agent_lead_id

        if steps is not None:
            step_map = {name: detail for name, detail in steps}

            st.write("")
            with st.container(border=True):
                st.markdown(f"### Agent run · {run_id}")

                if "error" in step_map:
                    st.error(f"Pipeline error: {step_map['error'].get('message', 'unknown error')}")

                if result:
                    lead = result["lead"]
                    inv = result["inventory"]
                    score = result["score"]
                    slot = result["slot"]
                    draft = result["draft"]

                    inv_txt = ("in stock at nearest dealership" if inv.get("inStock")
                               else "not in stock, flag for allocation")

                    progressive = [
                        ("Fetch lead from CRM",
                         f"lead found · source: {lead['source']} · model: {lead['model']}"),
                        ("Check dealer inventory", inv_txt),
                        ("Score lead",
                         f"score {score['score']}/100 · <span class='hot'>{score['tier'].upper()}</span>"),
                        ("Hold test-drive slot", f"slot held: {slot.get('slot', 'n/a')}"),
                        ("Draft follow-up message", "draft ready, awaiting human approval"),
                    ]

                    animate = st.session_state.get("agent_animate", False)
                    for i, (title, chip) in enumerate(progressive, start=1):
                        a, b = st.columns([0.5, 11])
                        with a:
                            st.markdown(f'<span class="step-num">{i}</span>', unsafe_allow_html=True)
                        with b:
                            st.markdown(f'<span class="step-title">{title}</span>', unsafe_allow_html=True)
                            slot_ph = st.empty()
                            if animate:
                                # Brief "working..." beat, then reveal the result.
                                slot_ph.markdown(
                                    '<span class="step-chip">· working…</span>',
                                    unsafe_allow_html=True,
                                )
                                time.sleep(0.45)
                            slot_ph.markdown(f'<span class="step-chip">{chip}</span>',
                                             unsafe_allow_html=True)
                        st.write("")
                    # Only animate once per run; later reruns (approve/edit) render instantly.
                    st.session_state.agent_animate = False

                    # Score card
                    color = _tier_color(score["tier"])
                    sc1, sc2 = st.columns([1, 5])
                    with sc1:
                        st.markdown(
                            f'<div class="score-big" style="color:{color}">{score["score"]}</div>'
                            f'<div class="score-tier">{score["tier"].upper()} LEAD</div>',
                            unsafe_allow_html=True,
                        )
                    with sc2:
                        for r in score["reasons"]:
                            st.markdown(f"<span style='color:#9fb0c9'>· {r}</span>", unsafe_allow_html=True)
                    st.caption(f"Scored via: {score['mode']}")

                    # Draft message (editable)
                    st.write("")

                    # Seed the editable copy once per run; keyed by lead so a new
                    # run refreshes it.
                    draft_key = f"draft_text::{run_id}"
                    if draft_key not in st.session_state:
                        st.session_state[draft_key] = draft["message"]
                    edit_flag = f"draft_editing::{run_id}"
                    sent_flag = f"draft_sent::{run_id}"

                    if st.session_state.get(edit_flag):
                        # ---- edit mode ----
                        new_text = st.text_area(
                            "Edit the follow-up message",
                            value=st.session_state[draft_key],
                            height=160,
                        )
                        e1, e2, _ = st.columns([1, 1, 4])
                        with e1:
                            if st.button("Save", type="primary", use_container_width=True):
                                st.session_state[draft_key] = new_text
                                st.session_state[edit_flag] = False
                                st.rerun()
                        with e2:
                            if st.button("Cancel", use_container_width=True):
                                st.session_state[edit_flag] = False
                                st.rerun()
                    else:
                        # ---- view mode ----
                        # Render newlines as <br> so the sign-off shows on its own line
                        # (HTML collapses raw newlines inside a div to whitespace).
                        _draft_html = st.session_state[draft_key].replace("\n", "<br>")
                        st.markdown(
                            f'<div class="draft">{_draft_html}</div>',
                            unsafe_allow_html=True,
                        )
                        st.caption(f"Drafted via: {draft['mode']}")

                        if st.session_state.get(sent_flag):
                            st.success("Approved and queued to send via WhatsApp (demo: not actually sent).")

                        b1, b2, _ = st.columns([1, 1, 4])
                        with b1:
                            if st.button("Approve & send", type="primary", use_container_width=True):
                                st.session_state[sent_flag] = True
                                st.rerun()
                        with b2:
                            if st.button("Edit draft", use_container_width=True):
                                st.session_state[edit_flag] = True
                                st.rerun()


# =========================================================================== #
# PAGE 3 — Integration Log
# =========================================================================== #

elif page == "Integration Log":
    st.markdown('<div class="page-title">MCP Tool Layer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Standardized tool contracts the agent calls, instead of a '
        "bespoke integration for every new AI feature.</div>",
        unsafe_allow_html=True,
    )

    st.info(
        "In production, **get_lead_from_crm**, **check_inventory** and **schedule_test_drive** "
        "would be served by an MCP server connected to Salesforce Sales Cloud, the Dealer "
        "Management System, and the showroom calendar. Below is a live log of tool calls made "
        "by the agent during this session. Run a lead in the Pipeline tab to populate it."
    )

    steps = st.session_state.agent_steps
    if not steps:
        st.markdown('<div class="term">// waiting for tool calls...</div>', unsafe_allow_html=True)
    else:
        # Only the real MCP tool steps (not the RAG / score / draft local steps).
        mcp_tools = {"get_lead_from_crm", "check_inventory", "schedule_test_drive"}
        lines = []
        for name, detail in steps:
            if name in mcp_tools:
                lines.append(
                    f'<span class="tool">→ {name}</span>  '
                    f'<span class="ok">OK</span>\n'
                    f'{json.dumps(detail, ensure_ascii=False)}'
                )
            elif name == "error":
                lines.append(f'ERROR: {detail.get("message", "unknown")}')
        body = "\n\n".join(lines) if lines else "// no MCP tool calls recorded"
        st.markdown(f'<div class="term">{body}</div>', unsafe_allow_html=True)

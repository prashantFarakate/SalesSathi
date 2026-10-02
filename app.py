"""
SalesSaathi — Streamlit demo UI (Phase 1: RAG Product & Finance Assistant)

A grounded product Q&A co-pilot for Tata Motors sales executives. Answers are
retrieved from approved product/finance documents and cited, and the assistant
says "I don't have that" instead of inventing specs or prices.

Run locally:
    streamlit run app.py
"""

import os

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
st.caption("Phase 1 · Grounded Product & Finance Q&A for Tata Motors sales teams")

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


def _queue(question: str) -> None:
    """Queue a question (from a sample button or the chat box) to be answered."""
    st.session_state.pending = question


# --------------------------------------------------------------------------- #
# Sample questions
# --------------------------------------------------------------------------- #

st.markdown("**Try one of these:**")
cols = st.columns(len(SAMPLE_QUESTIONS))
for col, q in zip(cols, SAMPLE_QUESTIONS):
    with col:
        st.button(q, key=f"sample::{q}", use_container_width=True, on_click=_queue, args=(q,))

# --------------------------------------------------------------------------- #
# Render prior conversation
# --------------------------------------------------------------------------- #

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

# --------------------------------------------------------------------------- #
# Chat input
# --------------------------------------------------------------------------- #

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

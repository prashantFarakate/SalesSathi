"""
SalesSaathi RAG answerer
------------------------
The generation ("G") half of Retrieval-Augmented Generation. Sits on top of
``retriever.retrieve()``:

    query -> retrieve grounded chunks -> build a grounded prompt -> Groq LLM
          -> cited natural-language answer

Uses LangChain's ChatGroq wrapper rather than the raw Groq SDK, so the model /
provider can be swapped with a one-line change and the code stays framework-
idiomatic (prompt -> messages -> invoke).

Design decisions worth noting for a demo:
- The answer is *grounded*: the model is instructed to use ONLY the retrieved
  context and to say it doesn't have the detail rather than invent specs/prices.
  This matters for a customer-facing sales tool (compliance / no hallucinated
  pricing).
- If retrieval returns nothing above the confidence threshold, we DON'T call the
  LLM at all -- we return an honest "I don't have that" answer. Saves tokens and
  is the correct product behaviour.
- If the LLM key is missing or the call fails, we degrade gracefully: the
  retrieved chunks are still returned so the demo never hard-crashes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv

from .retriever import retrieve, build_index, RetrievedChunk

load_dotenv()

# Groq's recommended production model (llama-3.3-70b-versatile was retired in
# 2026; Groq points production traffic at the gpt-oss models). Override via env.
MODEL_ID = os.getenv("SALESSAATHI_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """You are SalesSaathi, an AI co-pilot that helps Tata Motors sales
executives answer customer product and finance questions accurately.

Rules:
- Answer ONLY using the provided product context below.
- If the context does not contain the answer, say you do not have that detail
  and suggest checking with the dealership or finance team. Do NOT invent specs,
  ranges, charging times, or prices.
- Be concise, friendly, and helpful, like a knowledgeable sales advisor.
- Use bullet points or a short table for comparisons.
- Always note that prices and specs are indicative and vary by city and offers.
"""


@dataclass
class Answer:
    text: str
    sources: list[str]
    chunks: list[RetrievedChunk] = field(default_factory=list)
    mode: str = "llm"  # "llm" | "no_context" | "degraded"


def _build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[Source {i}: {c.source}]\n{c.text}")
    return "\n\n".join(blocks)


def answer(query: str, top_k: int = 6) -> Answer:
    """Retrieve grounded chunks and generate a cited answer.

    Returns an Answer with the generated text, the source files it was grounded
    in, and the raw chunks (for a "show the grounding" UI panel)."""
    build_index()  # no-op if the index already exists
    chunks = retrieve(query, top_k=top_k)

    # No confident match -> refuse honestly, no LLM call.
    if not chunks:
        return Answer(
            text=(
                "I don't have that information in my current product knowledge base. "
                "Please check with the dealership or finance team for an accurate answer."
            ),
            sources=[],
            chunks=[],
            mode="no_context",
        )

    sources = sorted({c.source for c in chunks})

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.strip() in ("", "your_groq_api_key_here"):
        # Degrade gracefully: no key, but retrieval still works -- surface the
        # grounded chunks so the demo remains useful and never crashes.
        return Answer(
            text=(
                "GROQ_API_KEY is not set, so I can't generate a written answer right now. "
                "Below is the grounded source material retrieved for your question."
            ),
            sources=sources,
            chunks=chunks,
            mode="degraded",
        )

    context = _build_context(chunks)
    user_prompt = (
        f"Product context:\n{context}\n\n"
        f"Customer/advisor question: {query}\n\n"
        "Answer using only the context above."
    )

    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatGroq(model=MODEL_ID, temperature=0.2, api_key=api_key)
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        text = response.content
        return Answer(text=text, sources=sources, chunks=chunks, mode="llm")
    except Exception as exc:  # noqa: BLE001 - degrade instead of crashing the demo
        return Answer(
            text=(
                f"Couldn't reach the language model ({exc}). "
                "Below is the grounded source material retrieved for your question."
            ),
            sources=sources,
            chunks=chunks,
            mode="degraded",
        )


if __name__ == "__main__":
    for q in [
        "What is the range and charging time of the Nexon EV Long Range?",
        "Compare the Nexon EV and Punch EV on range and price.",
        "What is the resale value of a 2019 Nexon?",  # not in KB -> should refuse
    ]:
        print(f"\n=== Q: {q}")
        a = answer(q)
        print(f"[mode: {a.mode}]")
        print(a.text)
        if a.sources:
            print("Sources:", ", ".join(a.sources))

# 🚗 SalesSaathi

**Live demo: https://tatamotors-salesathi.streamlit.app/**

An AI co-pilot for **Tata Motors dealer sales executives** — built to demonstrate
hands-on AI product skills (RAG, agentic pipelines, MCP), not just talk about them.

**Phase 1 (this build):** a grounded **Product & Finance Q&A assistant**. A sales
exec asks a natural-language question; the assistant answers **only** from approved
product documents, **cites its sources**, and says *"I don't have that"* instead of
inventing specs or prices.

Companion to the [PRD](./PRD.md) — read that for the business framing (problem,
KPIs, rollout, risks). This README is about the working build.

---

## Why this matters (product framing)

Lead *generation* is already well served by Tata's digital marketing. The gap is at
the dealer front-end, where a sales exec has to recall every variant's range, price,
charging time, warranty, and the current finance scheme — accurately, live, in front
of a customer. Wrong or slow answers cost conversions, and a hallucinated price is a
compliance problem.

SalesSaathi gives **grounded, cited, instant answers**, and refuses to guess.

**Target KPIs:** faster response time per query, higher lead-to-test-drive rate,
>90% of product/finance queries answered without escalation.

---

## What it does (Phase 1)

- Ingests product/finance docs (Markdown, text, PDF) from `rag/data/`.
- Chunks, embeds locally (**MiniLM via ChromaDB's built-in ONNX** — no API key, no
  PyTorch), and stores them in a persistent ChromaDB index.
- Retrieves the most relevant knowledge for a question using
  **parent-document retrieval**: it matches at chunk granularity for precision, then
  returns the *whole fact sheet* so the model gets complete specs.
- Generates a grounded answer with **Groq (`openai/gpt-oss-120b`) via LangChain**,
  and **cites its sources**.
- **Refuses to guess:** if nothing clears a confidence threshold, it says so — and
  doesn't even call the LLM. No hallucinated pricing.
- **Streamlit chat UI** that shows the retrieved context for transparency.

---

## Architecture

```
rag/data/*.md ──► retriever.py  (chunk + embed + ChromaDB, parent-doc retrieval)
                        │
                   retrieve(query)  ── below confidence? → "I don't have that"
                        │
                  answerer.py  (grounded prompt + Groq via LangChain, cited)
                        │
              app.py (Streamlit)  /  cli.py ask "<q>"
```

Design choices worth noting (and defending in an interview):

- **Local embeddings (ONNX MiniLM):** free, offline after first download, tiny
  footprint — deploy-friendly and no per-query embedding cost.
- **Parent-document retrieval:** matches on small chunks (precise), returns the whole
  fact sheet (complete facts). Our docs are short, so the parent is the whole file;
  on large brochures the parent would instead be the enclosing section.
- **Confidence threshold → honest refusal:** the single most important behaviour for
  a customer-facing sales tool. Calibrated so off-topic questions return nothing.
- **Graceful degradation:** if the LLM key is missing or unreachable, retrieval still
  runs and the grounded source material is shown — the demo never hard-crashes.

---

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

Copy-Item .env.example .env         # then add your GROQ_API_KEY
```

Get a free Groq key at https://console.groq.com/keys and set it in `.env`:

```
GROQ_API_KEY=gsk_your_key_here
```

> On a corporate network with an SSL-inspecting proxy (e.g. Zscaler), the app calls
> `truststore.inject_into_ssl()` so Python trusts the OS certificate store. It's a
> safe no-op elsewhere.

---

## Run

Build the index and try the RAG engine from the CLI:

```powershell
python rag/retriever.py                       # (re)build the index + sample retrievals
python cli.py ask "Compare the Nexon EV and Punch EV on range and price."
```

Launch the app:

```powershell
streamlit run app.py
```

---

## Demo script (3 moments that make the point)

1. **Accurate, grounded answer**
   *"What is the range and charging time of the Nexon EV Long Range?"*
   → correct figures (≈465 km, ~40 min DC fast charge) with a source citation.

2. **Comparison across documents**
   *"Compare the Nexon EV and Punch EV on range and price."*
   → a side-by-side table pulled from two fact sheets, complete on both range and price.

3. **Honest "I don't know"** (the compliance moment)
   *"What is the resale value of a 2019 Nexon?"*
   → the assistant declines rather than inventing a number.

Then open the **Retrieved context** expander to show the grounding — proof it's real
RAG, not the model guessing.

---

## Deploy (share a live link with an interviewer)

> **Live instance:** https://tatamotors-salesathi.streamlit.app/

**Streamlit Community Cloud** (free, purpose-built for this):

1. Push this project to a GitHub repo (the `.gitignore` already excludes `.env` and
   `rag/chroma_store/`).
2. On https://share.streamlit.io, create an app pointing at `app.py`.
3. In the app's **Settings → Secrets**, add your key:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
4. Deploy. The first boot downloads the embedding model and builds the index once;
   after that it's fast. Anyone with the link can use it — nothing to install.

Tips for a shared demo:
- The app caps each session at 12 questions to protect the API quota.
- Open the app yourself a few minutes before sharing so it's warm (Community Cloud
  sleeps idle apps).

---

## Project layout

```
salessaathi/
├── PRD.md                     Product requirements doc (business framing)
├── app.py                     Streamlit demo UI
├── cli.py                     Demo CLI (leads / run / ask)
├── rag/
│   ├── retriever.py           Embedding index + parent-document retrieval
│   ├── answerer.py            Grounded answer generation (LangChain + Groq)
│   └── data/                  Product fact sheets, competitor note, finance scheme
├── agent/                     Agentic lead pipeline (Phase 2 — MCP + scoring + draft)
└── mcp_server/                MCP server + mock CRM/DMS data (Phase 2)
```

---

## Roadmap

- **Phase 2:** agentic lead pipeline over a real **MCP** tool layer (CRM lead lookup,
  DMS inventory, test-drive booking), with human-in-the-loop follow-up drafting.
- **Phase 3:** customer-facing assistant (WhatsApp / showroom kiosk), broader DMS
  integration, multi-language.

---

## Honesty notes for the demo

- All vehicle specs, prices, and finance schemes in `rag/data/` are **illustrative
  placeholders**, clearly marked in each file — not verified production figures.
- No live Tata Motors systems are connected. Say so plainly if asked.

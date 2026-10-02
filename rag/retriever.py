"""
SalesSaathi RAG engine
-----------------------
Embedding-based Retrieval-Augmented Generation over the local knowledge base
(product fact sheets, competitor notes, finance schemes) in ``rag/data``.

Responsibilities:
- Load product knowledge (Markdown / text / PDF) from the data folder.
- Split documents into overlapping chunks.
- Embed chunks locally (ChromaDB's built-in ONNX MiniLM -- no PyTorch, no API
  key, runs offline after the first model download).
- Store and query them in a persistent ChromaDB collection.
- Return the most relevant chunks with a similarity score, and expose a
  confidence check so the assistant can say "I don't have that" instead of
  answering from a weak, irrelevant match -- important for pricing/compliance.

This module is UI-agnostic so it can be reused by the CLI, the agent pipeline,
or a Streamlit front-end.

Swapping the embedding model or vector store later is contained to this file --
the ``retrieve()`` interface the rest of the pipeline depends on does not change.
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_store")
COLLECTION_NAME = "salessaathi_kb"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, runs locally
# The knowledge base is a set of short product fact sheets. A larger chunk keeps
# each fact sheet's specs (range, price, charging, warranty) together in one
# chunk, so a comparison query retrieves complete facts for a model instead of a
# fragment that happens to omit the numbers.
CHUNK_SIZE = 1200      # characters per chunk
CHUNK_OVERLAP = 150    # overlap keeps context across chunk boundaries

# Below this cosine similarity, a retrieved chunk is treated as "not a
# confident match". Lets the assistant refuse instead of grounding an answer
# in an irrelevant document.
#
# Calibrated against the local KB: genuine matches score ~0.53-0.67, while
# off-topic queries (e.g. "how tall is the CEO") only reach ~0.41-0.46 because
# they still share automotive vocabulary. 0.5 cleanly separates the two so the
# assistant refuses out-of-scope questions instead of grounding on noise.
MIN_CONFIDENCE = 0.5


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_pdf_file(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(knowledge_dir: str = KNOWLEDGE_DIR) -> list[tuple[str, str]]:
    """Return a list of (source_name, full_text) for every supported file."""
    documents: list[tuple[str, str]] = []
    patterns = ("*.md", "*.txt", "*.pdf")

    for pattern in patterns:
        for path in glob.glob(os.path.join(knowledge_dir, pattern)):
            name = os.path.basename(path)
            try:
                if path.lower().endswith(".pdf"):
                    text = _read_pdf_file(path)
                else:
                    text = _read_text_file(path)
            except Exception as exc:  # noqa: BLE001 - surface a readable message
                print(f"[rag] Skipping {name}: {exc}")
                continue

            if text.strip():
                documents.append((name, text))

    return documents


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character windows, preferring paragraph breaks."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        window = text[start:end]

        # Try to break on a paragraph or sentence boundary for cleaner chunks.
        if end < len(text):
            for sep in ("\n\n", "\n", ". "):
                idx = window.rfind(sep)
                if idx > chunk_size * 0.5:
                    end = start + idx + len(sep)
                    window = text[start:end]
                    break

        chunk = window.strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start < 0:
            start = 0

    return chunks


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

def _get_embedding_function():
    """Return an embedding function.

    Prefers ChromaDB's built-in ONNX MiniLM model (no PyTorch needed, works
    reliably on Windows). Falls back to sentence-transformers if available.
    """
    try:
        # Built-in, downloads a small ONNX model on first use. No torch required.
        return embedding_functions.ONNXMiniLM_L6_V2()
    except Exception:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )


def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


# Cache of {source_filename: full_document_text} for parent-document retrieval.
_DOC_CACHE: dict[str, str] = {}


def _load_doc_cache() -> dict[str, str]:
    """Load and memoize the full text of every knowledge base document."""
    global _DOC_CACHE
    if not _DOC_CACHE:
        _DOC_CACHE = {name: text for name, text in load_documents()}
    return _DOC_CACHE


def build_index(rebuild: bool = False) -> int:
    """Ingest all knowledge base documents into ChromaDB.

    Returns the number of chunks indexed. Safe to call repeatedly; when
    rebuild is False and the collection already has data, it is a no-op.
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = _get_collection()

    if not rebuild and collection.count() > 0:
        return collection.count()

    documents = load_documents()
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    for source, full_text in documents:
        for i, chunk in enumerate(chunk_text(full_text)):
            ids.append(f"{source}::chunk-{i}")
            texts.append(chunk)
            metadatas.append({"source": source, "chunk": i})

    if texts:
        collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

    return collection.count()


def retrieve(query: str, top_k: int = 4, min_confidence: float = MIN_CONFIDENCE) -> list[RetrievedChunk]:
    """Return the most relevant knowledge for a query, as parent documents.

    We match at chunk granularity (precise) but return each matched source's
    *whole fact sheet* -- the parent-document / small-to-big retrieval pattern.
    This keeps a model's specs together (range AND price AND charging) even when
    those sections landed in different chunks, which comparison queries need.
    Our fact sheets are short, so the parent is the whole file; if the docs grew
    into large brochures, the parent would instead be the enclosing section.

    Sources whose best chunk scores below ``min_confidence`` are dropped. An
    empty list means "no confident match" -- the caller should decline to answer
    rather than guess.
    """
    collection = _get_collection()
    if collection.count() == 0:
        build_index()

    # Over-fetch chunks so that, after collapsing to parent documents, several
    # distinct sources still surface for comparison-style queries.
    results = collection.query(query_texts=[query], n_results=top_k * 3)
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    # Keep the best chunk score per source.
    best_score: dict[str, float] = {}
    for meta, dist in zip(metas, dists):
        score = 1.0 - float(dist)  # cosine distance -> rough similarity
        if score < min_confidence:
            continue
        source = meta.get("source", "unknown")
        if score > best_score.get(source, -1.0):
            best_score[source] = score

    # Return each surviving source's full document, ranked by best score.
    doc_cache = _load_doc_cache()
    ranked_sources = sorted(best_score, key=best_score.get, reverse=True)[:top_k]
    return [
        RetrievedChunk(text=doc_cache.get(src, ""), source=src, score=best_score[src])
        for src in ranked_sources
    ]


if __name__ == "__main__":
    count = build_index(rebuild=True)
    print(f"Indexed {count} chunks from {KNOWLEDGE_DIR}\n")

    for q in [
        "Nexon EV vs Creta Electric for a range-conscious customer",
        "any finance scheme in Maharashtra this quarter",
        "how tall is the CEO of Tata Motors",  # should return nothing confident
    ]:
        print(f"Query: {q}")
        hits = retrieve(q)
        if not hits:
            print("  -> no confident match (would say 'I don't know')\n")
            continue
        for h in hits:
            print(f"  -> {h.source} (score={h.score:.3f})")
        print()

"""
rag_engine.py - RAG (Retrieval-Augmented Generation) pipeline.

Responsibilities
----------------
1. PDF ingestion  → PyMuPDFLoader
2. Chunking       → RecursiveCharacterTextSplitter
3. Embedding      → HuggingFace sentence-transformers (local, no API key)
4. Vector store   → Qdrant Cloud (Free Tier)
5. Retrieval      → similarity search, top-k chunks
6. Generation     → Groq LLM (llama-3.1-70b-versatile)
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import SystemMessage, HumanMessage

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

import config

logger = logging.getLogger(__name__)


# ── Singleton helpers (cached for the process lifetime) ───────────────────────

_embeddings: HuggingFaceEmbeddings | None = None
_qdrant_client: QdrantClient | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return (or lazily create) the shared HuggingFace embeddings object."""
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model '%s' …", config.EMBEDDING_MODEL)
        _embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _get_qdrant_client() -> QdrantClient:
    """Return (or lazily create) the shared Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        logger.info("Connecting to Qdrant Cloud …")
        _qdrant_client = QdrantClient(
            url=config.get_qdrant_url(),
            api_key=config.get_qdrant_api_key(),
        )
    return _qdrant_client


def _ensure_collection(client: QdrantClient, embedding_dim: int = 384) -> None:
    """Create the Qdrant collection if it doesn't already exist."""
    existing = [c.name for c in client.get_collections().collections]
    if config.QDRANT_COLLECTION not in existing:
        logger.info("Creating Qdrant collection '%s' …", config.QDRANT_COLLECTION)
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )


# ── Core RAG functions ────────────────────────────────────────────────────────

def load_and_chunk_pdf(file_bytes: bytes, filename: str) -> List[Document]:
    """
    Save uploaded bytes to a temp file, parse with PyMuPDF, then chunk.

    Chunking strategy
    -----------------
    RecursiveCharacterTextSplitter tries to split on paragraphs (\\n\\n),
    then sentences (\\n), then words (' '), before falling back to raw
    characters.  This keeps semantically related sentences together far
    better than a simple fixed-size splitter, which improves retrieval
    precision.  1 000-char chunks with a 200-char overlap ensure that
    sentences straddling a boundary are captured in both adjacent chunks.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        logger.info("Loading PDF '%s' …", filename)
        loader = PyMuPDFLoader(str(tmp_path))
        raw_docs: List[Document] = loader.load()

        logger.info("Chunking %d page(s) …", len(raw_docs))
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = splitter.split_documents(raw_docs)
        logger.info("Produced %d chunk(s) from '%s'.", len(chunks), filename)

        # Tag every chunk with the original filename for provenance
        for chunk in chunks:
            chunk.metadata["source_file"] = filename

        return chunks
    finally:
        tmp_path.unlink(missing_ok=True)


def index_documents(chunks: List[Document]) -> int:
    """
    Embed chunks and upsert them into Qdrant.
    Returns the number of chunks indexed.
    """
    if not chunks:
        return 0

    embeddings = _get_embeddings()
    client = _get_qdrant_client()
    _ensure_collection(client)

    logger.info("Indexing %d chunks into Qdrant …", len(chunks))
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=config.get_qdrant_url(),
        api_key=config.get_qdrant_api_key(),
        collection_name=config.QDRANT_COLLECTION,
        force_recreate=False,  # append to existing collection
    )
    logger.info("Indexing complete.")
    return len(chunks)


def retrieve_context(query: str) -> List[Document]:
    """Return the top-k most relevant chunks for a query."""
    embeddings = _get_embeddings()
    client = _get_qdrant_client()
    _ensure_collection(client)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.QDRANT_COLLECTION,
        embedding=embeddings,
    )
    results = vector_store.similarity_search(query, k=config.TOP_K_RESULTS)
    logger.info("Retrieved %d chunk(s) for query.", len(results))
    return results


def answer_question(question: str) -> dict:
    """
    Full RAG query:
      1. Retrieve relevant chunks.
      2. Build a grounded prompt (system + context + question).
      3. Call Groq LLM.
      4. Return answer + source metadata.
    """
    context_docs = retrieve_context(question)

    if not context_docs:
        return {
            "answer": "No documents have been indexed yet. Please upload a PDF first.",
            "sources": [],
        }

    # Concatenate chunk texts
    context_text = "\n\n---\n\n".join(
        f"[Chunk from '{doc.metadata.get('source_file', 'unknown')}', "
        f"page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in context_docs
    )

    # Build messages
    messages = [
        SystemMessage(content=config.SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Context:\n{context_text}\n\n"
                f"Question: {question}"
            )
        ),
    ]

    # Call Groq
    llm = ChatGroq(
        api_key=config.get_groq_api_key(),
        model=config.LLM_MODEL,
        temperature=0.2,
    )
    response = llm.invoke(messages)

    sources = [
        {
            "file": doc.metadata.get("source_file", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "snippet": doc.page_content[:200].replace("\n", " "),
        }
        for doc in context_docs
    ]

    return {"answer": response.content, "sources": sources}
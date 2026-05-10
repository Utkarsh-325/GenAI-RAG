"""
config.py - Centralized environment variable management.
Loads all secrets from a .env file so nothing is hardcoded.
"""

import os
from dotenv import load_dotenv

# Load variables from .env file into the environment
load_dotenv()


def get_groq_api_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")
    return key


def get_qdrant_url() -> str:
    url = os.getenv("QDRANT_URL")
    if not url:
        raise ValueError("QDRANT_URL is not set. Please add it to your .env file.")
    return url


def get_qdrant_api_key() -> str:
    key = os.getenv("QDRANT_API_KEY")
    if not key:
        raise ValueError("QDRANT_API_KEY is not set. Please add it to your .env file.")
    return key


# ── Model settings ────────────────────────────────────────────────────────────
LLM_MODEL         = "llama-3.3-70b-versatile"
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"

# ── RAG / chunking settings ───────────────────────────────────────────────────
CHUNK_SIZE        = 1000
CHUNK_OVERLAP     = 200
TOP_K_RESULTS     = 5

# ── Qdrant collection ─────────────────────────────────────────────────────────
# A single shared collection lets you "upload once, chat many times".
QDRANT_COLLECTION = "notebooklm_docs"

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the question ONLY using the provided context. "
    "If the answer is not in the context, say you don't know. "
    "Do not use outside knowledge."
)

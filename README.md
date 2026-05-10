# 📓 NotebookLM Clone — Free-Stack RAG App

A fully open-source, 100 % free-tier replica of Google NotebookLM built with:

| Layer | Technology |
|---|---|
| **LLM** | Groq · `llama-3.1-70b-versatile` |
| **Embeddings** | HuggingFace · `sentence-transformers/all-MiniLM-L6-v2` (local) |
| **Vector DB** | Qdrant Cloud (Free Tier) |
| **Backend** | FastAPI / Python |
| **Frontend** | Streamlit |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit UI (main.py)              │
│   ┌──────────────┐          ┌──────────────────────┐ │
│   │  PDF Upload  │          │     Chat Window      │ │
│   └──────┬───────┘          └──────────┬───────────┘ │
└──────────┼───────────────────────────  │  ───────────┘
           │                             │
           ▼                             ▼
┌──────────────────────────────────────────────────────┐
│                  rag_engine.py                       │
│                                                      │
│  load_and_chunk_pdf()    answer_question()           │
│       │                       │                     │
│       ▼                       ▼                     │
│  PyMuPDFLoader         retrieve_context()           │
│       │                  similarity_search          │
│       ▼                       │                     │
│  RecursiveChar         ┌──────┴──────┐              │
│  TextSplitter          │  Qdrant     │              │
│       │                │  Cloud      │              │
│       ▼                └──────┬──────┘              │
│  HuggingFace                  │ top-k chunks        │
│  Embeddings ──────────────────┘                     │
│                               │                     │
│                               ▼                     │
│                         Groq LLM                    │
│                    (grounded answer)                │
└──────────────────────────────────────────────────────┘
           │
           ▼
     config.py (env vars, constants)
```

---

## Chunking Strategy — Why `RecursiveCharacterTextSplitter`?

### The Problem with Naive Splitting
A simple **fixed-size character splitter** cuts text every N characters with no regard for sentence or paragraph boundaries.  This means a single sentence — or even a single fact — can be split across two chunks.  When those chunks are later embedded, neither carries the full semantic signal of the original idea, degrading retrieval quality.

### How `RecursiveCharacterTextSplitter` Works
LangChain's `RecursiveCharacterTextSplitter` applies a **cascade of separators** in priority order:

```
["\n\n", "\n", " ", ""]
```

1. It first tries to split on **blank lines** (`\n\n`) — preserving paragraph structure.
2. If a paragraph is still larger than `chunk_size`, it falls back to **single newlines** (`\n`) — keeping sentences on the same line together.
3. If still too large, it splits on **spaces** — keeping words intact.
4. Only as a last resort does it split on individual **characters**.

### Configuration Used
```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,    # ~200–250 tokens — fits well in the LLM context window
    chunk_overlap=200,  # 20 % overlap — facts straddling boundaries appear in both chunks
)
```

| Parameter | Value | Rationale |
|---|---|---|
| `chunk_size` | 1 000 chars | Large enough to contain a complete idea; small enough for precise retrieval |
| `chunk_overlap` | 200 chars | Ensures continuity at boundaries; reduces information loss |

### Why This Matters for Retrieval
| Splitter | Boundary behaviour | Retrieval quality |
|---|---|---|
| `CharacterTextSplitter` | Cuts mid-sentence | ❌ Fragments context |
| `RecursiveCharacterTextSplitter` | Respects semantic units | ✅ Keeps ideas intact |
| `TokenTextSplitter` | Token-accurate but language-model-specific | ⚠️ Overkill for retrieval |

### Visual Example
```
Original text (1 200 chars):
"The transformer architecture was introduced in …
… (paragraph 1, 600 chars) …

… (paragraph 2, 600 chars) …"

Fixed splitter → cuts at char 1 000 mid-sentence ✂️

Recursive splitter → splits cleanly at \n\n boundary ✅
  Chunk 1: paragraph 1 + 200-char overlap intro of paragraph 2
  Chunk 2: paragraph 2 (with 200-char tail of paragraph 1)
```

---

## File Structure

```
notebooklm-clone/
├── main.py            # Streamlit UI
├── rag_engine.py      # RAG pipeline (load → chunk → embed → retrieve → generate)
├── config.py          # Environment variable management & constants
├── requirements.txt   # Pinned dependencies
├── .env.example       # Template — copy to .env and fill in secrets
└── README.md          # This file
```

---

## Quick Start

### 1. Prerequisites
- Python 3.10 or 3.11
- A free [Groq API key](https://console.groq.com)
- A free [Qdrant Cloud](https://cloud.qdrant.io) cluster

### 2. Clone / Download & Install

```bash
# (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# CPU-only PyTorch first (saves ~700 MB vs the default CUDA build)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install everything else
pip install -r requirements.txt
```

### 3. Configure Secrets

```bash
cp .env.example .env
# Open .env in your editor and fill in the three values:
#   GROQ_API_KEY
#   QDRANT_URL
#   QDRANT_API_KEY
```

### 4. Run the App

```bash
streamlit run main.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 5. Usage
1. **Upload** one or more PDFs using the sidebar.
2. Click **⚡ Index Selected PDFs** — chunks are embedded and stored in Qdrant.
3. **Ask questions** in the chat box.  Answers are grounded exclusively in your documents.

---

## Deployment (Streamlit Community Cloud — free live link)

1. Push this folder to a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → point to `main.py`.
3. Under **Advanced settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY     = "gsk_..."
   QDRANT_URL       = "https://..."
   QDRANT_API_KEY   = "..."
   ```
4. Click **Deploy** — you'll get a shareable `https://yourapp.streamlit.app` URL.

> **Note:** `python-dotenv` reads from `.env` locally; on Streamlit Cloud the secrets are injected as environment variables automatically — no code change needed.

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| PDF loader | `PyMuPDFLoader` | Handles tables, multi-column, and scanned PDFs better than `PDFPlumber` or `pypdf` |
| Embeddings | `all-MiniLM-L6-v2` | 384-dim, runs on CPU in < 1 s per chunk, no API key required |
| Vector DB | Qdrant Cloud Free | Persistent across sessions, generous 1 GB free tier |
| LLM | Groq `llama-3.1-70b-versatile` | Fastest free inference available; 128 k context window |
| Prompt | Strict "context only" system prompt | Prevents hallucination and keeps answers grounded |
| Collection strategy | Single shared collection | "Upload once, chat many times" — new PDFs add to the same store |

---

## License
MIT — free to use, modify, and deploy.

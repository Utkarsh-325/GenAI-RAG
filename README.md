# 📓 NotebookLM Clone — Free-Stack Corrective RAG (CRAG) App

A fully open-source, 100% free-tier replica of Google NotebookLM upgraded to **Corrective RAG (CRAG)**. This app evaluates retrieved documents for relevance, reformulates queries, and automatically triggers keyless web searches to supplement or correct information when local documents are insufficient.

| Layer | Technology |
|---|---|
| **LLM** | Groq · `llama-3.3-70b-versatile` |
| **CRAG Pipeline** | Relevance Grading, Query Reformulation & Web Fallback |
| **Embeddings** | HuggingFace · `sentence-transformers/all-MiniLM-L6-v2` (local) |
| **Vector DB** | Qdrant Cloud (Free Tier) with **Automatic local `:memory:` fallback** |
| **Web Search** | DuckDuckGo (keyless, via `ddgs`) |
| **Frontend** | Streamlit (collapsible execution steps & web resource styling) |

---

## Corrective RAG (CRAG) Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit UI (main.py)              │
│   ┌──────────────┐          ┌──────────────────────┐ │
│   │  PDF Upload  │          │     Chat Window      │ │
│   └──────┬───────┘          └──────────┬───────────┘ │
└──────────┼─────────────────────────────┼─────────────┘
           │                             │
           ▼                             ▼
┌──────────────────────────────────────────────────────┐
│                  rag_engine.py                       │
│                                                      │
│  load_and_chunk_pdf()    answer_question()           │
│       │                       │                     │
│       ▼                       ▼                     │
│  PyMuPDFLoader         retrieve_context()           │
│       │                 (Qdrant Cloud / :memory:)   │
│       ▼                       │                     │
│  RecursiveChar                ├─────────────────┐   │
│  TextSplitter                 ▼                 │   │
│       │                Grade Chunks             │   │
│       ▼                (LLaMA Evaluator)        │   │
│  HuggingFace                  │                 │   │
│  Embeddings                   ├── [Irrelevant]  │   │
│                               ▼                 ▼   │
│                         Web Search         [All Ok] │
│                         (DuckDuckGo)            │   │
│                               │                 │   │
│                               ▼                 │   │
│                        Merged Context           │   │
│                               │                 │   │
│                               ▼                 │   │
│                           Groq LLM ◄────────────┘   │
│                        (final answer)               │
└──────────────────────────────────────────────────────┘
           │
           ▼
     config.py (env vars, constants)
```

### Key CRAG Pipeline Steps
1. **Retrieve**: Pulls the top-$K$ local document chunks from Qdrant.
2. **Grade**: Evaluates each chunk individually using the LLaMA model. If a chunk contains details directly relevant to the user question, it is marked as `RELEVANT`; otherwise, it is graded as `IRRELEVANT`.
3. **Correct/Supplement**: If any retrieved local chunks are graded as `IRRELEVANT` (or if no chunks are found at all), CRAG initiates search query reformulation and runs a web search using DuckDuckGo.
4. **Generate**: Synthesizes the final answer using the filtered relevant chunks plus web search snippets.
5. **Trace**: The execution steps and evaluation results are logged and displayed directly in the Streamlit frontend.

---

## Chunking Strategy — Why `RecursiveCharacterTextSplitter`?

### The Problem with Naive Splitting
A simple **fixed-size character splitter** cuts text every N characters with no regard for sentence or paragraph boundaries. This means a single sentence — or even a single fact — can be split across two chunks. When those chunks are later embedded, neither carries the full semantic signal of the original idea, degrading retrieval quality.

### How `RecursiveCharacterTextSplitter` Works
LangChain's `RecursiveCharacterTextSplitter` applies a **cascade of separators** in priority order:

```
["\n\n", "\n", " ", ""]
```

| Parameter | Value | Rationale |
|---|---|---|
| `chunk_size` | 1 000 chars | Large enough to contain a complete idea; small enough for precise retrieval |
| `chunk_overlap` | 200 chars | Ensures continuity at boundaries; reduces information loss |

---

## File Structure

```
notebooklm-clone/
├── main.py            # Streamlit UI (handles CRAG traces & dynamically formatted web sources)
├── rag_engine.py      # CRAG pipeline (Retrieval, document grading, search, & answer generation)
├── config.py          # Environment variable management, system prompts, & constants
├── requirements.txt   # Pinned dependencies (added ddgs package)
├── .env.example       # Template — copy to .env and fill in secrets
└── README.md          # This file
```

---

## Quick Start

### 1. Prerequisites
- Python 3.10 or 3.11
- A free [Groq API key](https://console.groq.com)
- (Optional) A free [Qdrant Cloud](https://cloud.qdrant.io) cluster. If your cluster is expired or not configured, the app will **automatically and gracefully fall back to a local, in-memory Qdrant database**.

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
# Open .env in your editor and fill in:
#   GROQ_API_KEY
#   QDRANT_URL     (Optional - if omitted or expired, local in-memory Qdrant is used)
#   QDRANT_API_KEY (Optional)
```

### 4. Run the App

```bash
streamlit run main.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 5. Usage
1. **Upload** one or more PDFs using the sidebar.
2. Click **⚡ Index Selected PDFs** — chunks are embedded and stored in the vector store.
3. **Ask questions** in the chat box:
   - If you ask a question answered in the document, the app answers using local document context and notes that the documents were graded as `RELEVANT`.
   - If you ask a question not present in the document (or if you haven't uploaded any documents yet), the evaluator flags the gap, reformulates the search query, triggers a DuckDuckGo web search, and answers using the web results!

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

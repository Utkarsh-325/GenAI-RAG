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
import json
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
qdrant_is_local: bool = False


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
    global _qdrant_client, qdrant_is_local
    if _qdrant_client is None:
        try:
            logger.info("Connecting to Qdrant Cloud …")
            _qdrant_client = QdrantClient(
                url=config.get_qdrant_url(),
                api_key=config.get_qdrant_api_key(),
                timeout=5.0,
            )
            # Test connection
            _qdrant_client.get_collections()
            qdrant_is_local = False
            logger.info("Successfully connected to Qdrant Cloud.")
        except Exception as e:
            logger.warning(
                "Failed to connect to Qdrant Cloud (%s). "
                "Falling back to local in-memory Qdrant database.",
                e
            )
            _qdrant_client = QdrantClient(path=":memory:")
            qdrant_is_local = True
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
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.QDRANT_COLLECTION,
        embedding=embeddings,
    )
    vector_store.add_documents(chunks)
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


def grade_document(llm: ChatGroq, question: str, document_text: str) -> bool:
    """
    Grades a retrieved document's relevance to the user's question.
    Returns True if relevant, False otherwise.
    """
    prompt = (
        "You are an expert evaluator grading the relevance of a retrieved document to a user question.\n\n"
        f"Here is the retrieved document:\n"
        f"<document>\n{document_text}\n</document>\n\n"
        f"Here is the user question:\n"
        f"{question}\n\n"
        "Evaluate if the document contains information that is directly relevant to or helps answer the user question.\n"
        "Provide your feedback as a JSON object with a single key 'score' which must be either 'yes' or 'no' to indicate whether the document is relevant.\n"
        "Do not explain or output anything else. Only output the raw JSON."
    )
    
    try:
        response = llm.invoke([
            SystemMessage(content="You are a strict document grader. Output JSON only."),
            HumanMessage(content=prompt)
        ])
        
        content = response.content.strip()
        # Clean any markdown block formatting if LLM generated it
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        data = json.loads(content)
        score = data.get("score", "no").strip().lower()
        return score == "yes"
    except Exception as e:
        logger.error(f"Error grading document: {e}. Fallback to string check.")
        text = response.content.lower() if 'response' in locals() else ""
        if '"score": "yes"' in text or '"score":"yes"' in text or "yes" in text:
            return True
        return False


def reformulate_query(llm: ChatGroq, question: str) -> str:
    """
    Optimizes/reformulates the user's question into a query suitable for web search.
    """
    prompt = (
        "You are an expert search query optimizer. The user asked a question, but some retrieved local documents were irrelevant or insufficient.\n"
        "Your task is to generate a single, highly effective search query that will be used to search the web for relevant information to answer the question.\n\n"
        f"Original user question: {question}\n\n"
        "Output only the optimized search query. Do not include any introduction, explanations, markdown formatting, or quotes."
    )
    try:
        response = llm.invoke([
            SystemMessage(content="You are a search query optimizer. Output ONLY the query string."),
            HumanMessage(content=prompt)
        ])
        query = response.content.strip().strip('"').strip("'")
        return query if query else question
    except Exception as e:
        logger.error(f"Error reformulating query: {e}")
        return question


def execute_web_search(query: str, max_results: int = 3) -> List[Document]:
    """
    Searches the web using DuckDuckGo and returns a list of LangChain Document objects.
    """
    from ddgs import DDGS
    
    logger.info("Executing DuckDuckGo web search for query: '%s' …", query)
    documents = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            for r in results:
                title = r.get("title", "Web Page")
                href = r.get("href", "#")
                body = r.get("body", "")
                
                # Combine title and body as the content
                content = f"Title: {title}\nURL: {href}\nContent: {body}"
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source_file": f"Web: {href}",
                        "page": "web",
                        "title": title,
                        "url": href
                    }
                )
                documents.append(doc)
    except Exception as e:
        logger.error(f"Error in web search: {e}")
    
    logger.info("Retrieved %d web search result(s).", len(documents))
    return documents


def answer_question(question: str, enable_web_search: bool = True) -> dict:
    """
    Corrective RAG (CRAG) flow:
      1. Retrieve relevant chunks from local vector store.
      2. Grade retrieved chunks for relevance to the question.
      3. If any chunks are irrelevant, reformulate the query and perform a web search.
      4. Synthesize final response using relevant local chunks + web search results.
      5. Return response, source list, and logs of CRAG steps.
    """
    crag_steps = []
    
    # 1. Retrieval
    crag_steps.append({
        "title": "🔍 Retrieving local documents",
        "status": "In progress...",
        "details": f"Querying Qdrant index for question: '{question}'"
    })
    
    context_docs = retrieve_context(question)
    
    # Grader/Reformulator LLM
    grader_llm = ChatGroq(
        api_key=config.get_groq_api_key(),
        model=config.LLM_MODEL,
        temperature=config.CRAG_GRADER_TEMPERATURE,
    )
    
    if not context_docs:
        # If no documents are in Qdrant at all, we fallback to web search entirely (if enabled)
        crag_steps[-1]["status"] = "No local documents found"
        
        if not enable_web_search:
            crag_steps[-1]["details"] = "Qdrant collection is empty or no documents returned. Web search is disabled by user configuration."
            return {
                "answer": "No local documents have been indexed yet, and web search fallback is disabled.",
                "sources": [],
                "crag_steps": crag_steps
            }
            
        crag_steps[-1]["details"] = "Qdrant collection is empty or no documents returned."
        
        web_query = reformulate_query(grader_llm, question)
        crag_steps.append({
            "title": "🌐 Web search fallback",
            "status": "Searching...",
            "details": f"No local documents found. Searching web for: '{web_query}'"
        })
        
        web_docs = execute_web_search(web_query, max_results=config.WEB_SEARCH_MAX_RESULTS)
        crag_steps[-1]["status"] = f"Found {len(web_docs)} web result(s)"
        crag_steps[-1]["details"] = "\n\n".join(f"- [{d.metadata.get('title', 'Link')}]({d.metadata.get('url', '#')})" for d in web_docs)
        
        if not web_docs:
            return {
                "answer": "No local documents are indexed, and web search did not return any results. Please upload a document or ask a search-friendly question.",
                "sources": [],
                "crag_steps": crag_steps
            }
            
        all_context_docs = web_docs
    else:
        crag_steps[-1]["status"] = f"Retrieved {len(context_docs)} document chunk(s)"
        crag_steps[-1]["details"] = "\n\n".join(
            f"- Chunk from `{doc.metadata.get('source_file', 'unknown')}` page {doc.metadata.get('page', '?')}"
            for doc in context_docs
        )
        
        # 2. Grading
        crag_steps.append({
            "title": "📋 Evaluating document relevance",
            "status": "In progress...",
            "details": "Grading each chunk using LLaMA..."
        })
        
        relevant_docs = []
        irrelevant_count = 0
        grading_details = []
        
        for idx, doc in enumerate(context_docs, 1):
            is_relevant = grade_document(grader_llm, question, doc.page_content)
            status_str = "RELEVANT" if is_relevant else "IRRELEVANT"
            grading_details.append(
                f"**Chunk {idx}** (from `{doc.metadata.get('source_file')}` p.{doc.metadata.get('page')}): **{status_str}**\n\n"
                f"*Snippet:* \"{doc.page_content[:150].strip()}...\""
            )
            if is_relevant:
                relevant_docs.append(doc)
            else:
                irrelevant_count += 1
                
        crag_steps[-1]["status"] = f"Evaluation complete: {len(relevant_docs)} relevant, {irrelevant_count} irrelevant"
        crag_steps[-1]["details"] = "\n\n---\n\n".join(grading_details)
        
        # 3. Action Decision
        run_web_search = (irrelevant_count > 0) and enable_web_search
        
        if not enable_web_search and irrelevant_count > 0:
            crag_steps.append({
                "title": "🌐 Web search skipped",
                "status": "Disabled by user",
                "details": f"{irrelevant_count} chunk(s) graded as irrelevant, but web search fallback is disabled."
            })
            
        if run_web_search:
            # We reformulate the query and run search to supplement/correct
            crag_steps.append({
                "title": "🧠 Reformulating search query",
                "status": "In progress...",
                "details": "Optimizing search query for web search..."
            })
            web_query = reformulate_query(grader_llm, question)
            crag_steps[-1]["status"] = f"Optimized query: '{web_query}'"
            crag_steps[-1]["details"] = f"Original question: '{question}'\n\nReformulated search query: **{web_query}**"
            
            crag_steps.append({
                "title": "🌐 Executing web search",
                "status": "In progress...",
                "details": f"Searching DuckDuckGo for: '{web_query}'"
            })
            web_docs = execute_web_search(web_query, max_results=config.WEB_SEARCH_MAX_RESULTS)
            crag_steps[-1]["status"] = f"Retrieved {len(web_docs)} web search result(s)"
            crag_steps[-1]["details"] = "\n\n".join(
                f"- [{d.metadata.get('title', 'Link')}]({d.metadata.get('url', '#')})" for d in web_docs
            )
            
            all_context_docs = relevant_docs + web_docs
        else:
            all_context_docs = relevant_docs
            
    # 4. Generation
    crag_steps.append({
        "title": "✍️ Synthesizing final answer",
        "status": "In progress...",
        "details": f"Generating answer using {len(all_context_docs)} total document(s) in context."
    })
    
    # Concatenate texts
    context_text = "\n\n---\n\n".join(
        f"[Source: '{doc.metadata.get('source_file', 'unknown')}', page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in all_context_docs
    )
    
    messages = [
        SystemMessage(content=config.CRAG_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Context:\n{context_text}\n\n"
                f"Question: {question}"
            )
        ),
    ]
    
    generator_llm = ChatGroq(
        api_key=config.get_groq_api_key(),
        model=config.LLM_MODEL,
        temperature=0.2,
    )
    response = generator_llm.invoke(messages)
    
    crag_steps[-1]["status"] = "Answer generated successfully"
    crag_steps[-1]["details"] = f"Synthesis complete. Context size: {len(all_context_docs)} items."
    
    sources = []
    for doc in all_context_docs:
        sources.append({
            "file": doc.metadata.get("source_file", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "snippet": doc.page_content[:200].replace("\n", " "),
            "url": doc.metadata.get("url", None)
        })
    
    return {
        "answer": response.content,
        "sources": sources,
        "crag_steps": crag_steps
    }
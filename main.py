"""
main.py - Streamlit frontend for the NotebookLM clone.

Run with:
    streamlit run main.py
"""

import logging
from urllib.parse import urlparse
import streamlit as st

import rag_engine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📓 NotebookLM Clone",
    page_icon="📓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Sidebar styling */
        [data-testid="stSidebar"] {background: #0f172a;}
        [data-testid="stSidebar"] * {color: #e2e8f0 !important;}

        /* Chat bubbles */
        .user-bubble {
            background: #1e40af;
            color: #fff;
            border-radius: 16px 16px 4px 16px;
            padding: 12px 16px;
            margin: 8px 0;
            max-width: 80%;
            margin-left: auto;
            word-wrap: break-word;
        }
        .assistant-bubble {
            background: #1e293b;
            color: #e2e8f0;
            border-radius: 16px 16px 16px 4px;
            padding: 12px 16px;
            margin: 8px 0;
            max-width: 80%;
            word-wrap: break-word;
        }
        .source-chip {
            display: inline-block;
            background: #334155;
            color: #94a3b8;
            border-radius: 12px;
            padding: 2px 10px;
            font-size: 0.75rem;
            margin: 2px 3px;
        }
        .hero-title {font-size: 2.4rem; font-weight: 800; color: #f1f5f9;}
        .hero-sub   {color: #64748b; margin-top: -12px; margin-bottom: 24px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # list of {"role": ..., "content": ..., "sources": [...]}
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []     # names of files already indexed

# ── Sidebar — PDF upload ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Upload Documents")
    st.markdown("_Upload PDFs to your notebook. Once indexed you can ask questions across all of them._")

    uploaded_files = st.file_uploader(
        "Choose PDF file(s)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    index_btn = st.button("⚡ Index Selected PDFs", use_container_width=True)

    if index_btn and uploaded_files:
        for uf in uploaded_files:
            if uf.name in st.session_state.indexed_files:
                st.sidebar.info(f"'{uf.name}' already indexed — skipping.")
                continue
            with st.spinner(f"Indexing **{uf.name}** …"):
                try:
                    chunks = rag_engine.load_and_chunk_pdf(uf.read(), uf.name)
                    n = rag_engine.index_documents(chunks)
                    st.session_state.indexed_files.append(uf.name)
                    st.success(f"✅ {uf.name} → {n} chunks indexed")
                except Exception as exc:
                    st.error(f"❌ Failed to index '{uf.name}': {exc}")

    if st.session_state.indexed_files:
        st.markdown("---")
        st.markdown("### 📚 Indexed Files")
        for name in st.session_state.indexed_files:
            st.markdown(f"- 📄 `{name}`")

    st.markdown("---")
    st.markdown(
        "<small>Powered by **Groq** · **LLaMA 3.1 70B** · **Qdrant** · **HuggingFace Embeddings**</small>",
        unsafe_allow_html=True,
    )

# ── Main area — chat ──────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">📓 NotebookLM Clone</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Ask anything — answers are grounded in your uploaded documents.</div>', unsafe_allow_html=True)

# Render conversation history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-bubble">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
        
        # Render CRAG Steps
        if msg.get("crag_steps"):
            with st.expander("🛠️ Corrective RAG (CRAG) Execution Trace", expanded=False):
                for step in msg["crag_steps"]:
                    st.markdown(f"### {step['title']}")
                    st.markdown(f"**Status:** `{step['status']}`")
                    if step.get("details"):
                        st.markdown(step["details"])
                    st.markdown("---")
                    
        if msg.get("sources"):
            chips = []
            for s in msg["sources"]:
                if s.get("url"):
                    domain = urlparse(s["url"]).netloc
                    chips.append(f'<span class="source-chip">🌐 <a href="{s["url"]}" target="_blank" style="color: inherit; text-decoration: none;">{domain}</a></span>')
                else:
                    chips.append(f'<span class="source-chip">📄 {s["file"]} · p.{s["page"]}</span>')
            
            st.markdown(f"<div style='margin-top:4px'>{''.join(chips)}</div>", unsafe_allow_html=True)
            with st.expander("View source snippets", expanded=False):
                for i, s in enumerate(msg["sources"], 1):
                    if s.get("url"):
                        st.markdown(f"**[{i}] 🌐 [{s['file']}]({s['url']})**")
                    else:
                        st.markdown(f"**[{i}] 📄 {s['file']} — page {s['page']}**")
                    st.caption(s["snippet"] + " …")

st.markdown("<br>", unsafe_allow_html=True)

# Input box
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([9, 1])
    with col1:
        user_input = st.text_input(
            "Ask a question",
            placeholder="e.g. What are the main findings of this paper?",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("➤", use_container_width=True)

if submitted and user_input.strip():
    question = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": question})

    if not st.session_state.indexed_files:
        answer_data = {
            "answer": "⚠️ No documents are indexed yet. Please upload and index a PDF using the sidebar first.",
            "sources": [],
        }
    else:
        with st.spinner("Thinking …"):
            try:
                answer_data = rag_engine.answer_question(question)
            except Exception as exc:
                answer_data = {"answer": f"❌ Error: {exc}", "sources": []}

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_data["answer"],
        "sources": answer_data["sources"],
        "crag_steps": answer_data.get("crag_steps", []),
    })
    st.rerun()

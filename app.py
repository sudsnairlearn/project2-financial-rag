"""
Streamlit chat UI for the Financial RAG pipeline.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import importlib
import time

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Apple 10-K RAG",
    page_icon="📄",
    layout="centered",
)

st.title("📄 Apple 10-K Q&A")
st.caption("Ask questions about Apple's annual report. Powered by Nebius Token Factory + FAISS.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    strategy = st.radio(
        "Chunking strategy",
        options=["semantic", "fixed"],
        index=0,
        help="Semantic: topic-aware chunks. Fixed: uniform 512-token chunks.",
    )
    use_reranker = st.toggle("Cross-encoder reranker", value=False,
                             help="Re-scores retrieved chunks with a local cross-encoder before answering.")
    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── Load pipeline (cached per strategy + reranker combo) ─────────────────────
@st.cache_resource(show_spinner="Loading pipeline…")
def get_pipeline(strategy: str, use_reranker: bool):
    mod = importlib.import_module("3_rag_pipeline")
    return mod.FinancialRAGPipeline(chunking_strategy=strategy, use_reranker=use_reranker)

pipeline = get_pipeline(strategy, use_reranker)

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            meta = msg["meta"]
            st.caption(f"⏱ {meta['latency']:.1f}s · {meta['strategy']} chunks · {'reranked' if meta['reranked'] else 'no rerank'}")

# ── Input ─────────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask a question about the Apple 10-K…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Run pipeline
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            t0 = time.time()
            result = pipeline.query(question)
            latency = time.time() - t0

        st.markdown(result["answer"])
        st.caption(
            f"⏱ {latency:.1f}s · {result['chunking_strategy']} chunks · "
            f"{'reranked' if result['use_reranker'] else 'no rerank'}"
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "meta": {
            "latency": latency,
            "strategy": result["chunking_strategy"],
            "reranked": result["use_reranker"],
        },
    })

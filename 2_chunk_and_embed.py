"""
Step 2 — Chunk the 10-K using two strategies and embed with Nebius Token Factory.

Strategy 1 — Fixed-size:
    Split every 512 tokens (≈2000 chars) with 50-token overlap.
    Fast and predictable. Breaks sentence/paragraph boundaries.

Strategy 2 — Semantic:
    LangChain SemanticChunker detects topic shifts by comparing adjacent
    sentence embeddings. Splits when cosine similarity drops below the
    configured percentile threshold. Produces coherent, meaning-preserving chunks.

Both use Nebius Token Factory for embeddings.
Results saved to two local FAISS index directories for side-by-side comparison.

Usage:
    python 2_chunk_and_embed.py
    python 2_chunk_and_embed.py --company Microsoft
"""
from __future__ import annotations

import argparse
import os
import shutil

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

import config


# ── Nebius embeddings (Token Factory, OpenAI-compatible) ─────────────────────

def get_embeddings() -> OpenAIEmbeddings:
    if not config.NEBIUS_API_KEY:
        raise EnvironmentError("NEBIUS_API_KEY not set. See README.")
    return OpenAIEmbeddings(
        model=config.NEBIUS_EMBED_MODEL,
        openai_api_base=config.NEBIUS_BASE_URL,
        openai_api_key=config.NEBIUS_API_KEY,
        check_embedding_ctx_length=False,  # Nebius wants raw strings, not pre-tokenized input
    )


# ── Load raw 10-K text ────────────────────────────────────────────────────────

def load_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages).strip()
    if not text:
        raise ValueError(f"Could not extract text from PDF: {path}")
    return text


def get_input_file_path(company: str) -> str:
    if config.USE_SAMPLE:
        return config.SAMPLE_DOC_PATH

    txt_path = os.path.join(config.DATA_DIR, f"{company.lower()}_10k.txt")
    pdf_path = os.path.join(config.DATA_DIR, f"{company.lower()}_10k.pdf")
    if os.path.exists(txt_path):
        return txt_path
    if os.path.exists(pdf_path):
        return pdf_path
    raise FileNotFoundError(
        f"No filing found. Place {txt_path} or {pdf_path} in the data folder."
    )


def load_text(company: str) -> str:
    path = get_input_file_path(company)
    if config.USE_SAMPLE:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Sample doc not found at {path}.")
        print(f"  (Using sample document: {path})")
        print("  To use a real 10-K, set USE_SAMPLE=False in config.py and place a .txt or .pdf file in data/.")

    if path.lower().endswith(".pdf"):
        print(f"  Loading PDF document: {path}")
        return load_pdf_text(path)

    with open(path, encoding="utf-8") as f:
        return f.read()


# ── Strategy 1: Fixed-size chunking ──────────────────────────────────────────

def fixed_chunks(text: str) -> list[Document]:
    """
    RecursiveCharacterTextSplitter respects paragraph and sentence boundaries
    where possible within the size budget, making it a better 'fixed' baseline
    than a naive character split.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.FIXED_CHUNK_SIZE * 4,    # chars ≈ tokens × 4
        chunk_overlap=config.FIXED_CHUNK_OVERLAP * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [
        Document(page_content=c, metadata={"strategy": "fixed", "chunk_index": i})
        for i, c in enumerate(chunks)
    ]


# ── Strategy 2: Semantic chunking ─────────────────────────────────────────────

def semantic_chunks(text: str, embeddings: OpenAIEmbeddings) -> list[Document]:
    """
    SemanticChunker splits at points where adjacent sentence embeddings
    diverge beyond the percentile threshold — capturing genuine topic shifts.
    Requires calling the embedding API during chunking (uses Nebius).
    """
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=config.SEMANTIC_BREAKPOINT,
        breakpoint_threshold_amount=config.SEMANTIC_THRESHOLD,
    )
    chunks = splitter.split_text(text)
    return [
        Document(page_content=c, metadata={"strategy": "semantic", "chunk_index": i})
        for i, c in enumerate(chunks)
    ]


# ── Embed and persist (FAISS) ─────────────────────────────────────────────────

def embed_and_store(
    docs: list[Document],
    embeddings: OpenAIEmbeddings,
    persist_dir: str,
) -> FAISS:
    # Wipe any existing index so switching documents doesn't mix old chunks in.
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print(f"  Cleared existing store at {persist_dir}")

    vectorstore = FAISS.from_documents(docs, embeddings)
    os.makedirs(persist_dir, exist_ok=True)
    vectorstore.save_local(persist_dir)
    return vectorstore


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--company", default=config.TARGET_COMPANY,
                        choices=list(config.COMPANY_CIKS.keys()))
    args = parser.parse_args()

    print(f"Loading 10-K text for {args.company}…")
    text = load_text(args.company)
    print(f"  {len(text):,} characters loaded.")

    embeddings = get_embeddings()
    print(f"  Embedding model: {config.NEBIUS_EMBED_MODEL} via Nebius Token Factory")

    # ── Fixed chunking ────────────────────────────────────────────────────────
    print("\n[Strategy 1] Fixed-size chunking…")
    fixed_docs = fixed_chunks(text)
    print(f"  {len(fixed_docs)} chunks created (size ≈{config.FIXED_CHUNK_SIZE} tokens, overlap {config.FIXED_CHUNK_OVERLAP})")
    avg_fixed = sum(len(d.page_content) for d in fixed_docs) // len(fixed_docs)
    print(f"  Average chunk length: {avg_fixed} chars")
    print("  Embedding and storing in FAISS…")
    embed_and_store(fixed_docs, embeddings, config.FAISS_DIR_FIXED)
    print(f"  Saved to {config.FAISS_DIR_FIXED}  ({len(fixed_docs)} vectors)")

    # ── Semantic chunking ─────────────────────────────────────────────────────
    print("\n[Strategy 2] Semantic chunking (calls Nebius for topic-shift detection)…")
    sem_docs = semantic_chunks(text, embeddings)
    print(f"  {len(sem_docs)} chunks created (breakpoint: {config.SEMANTIC_BREAKPOINT} @ {config.SEMANTIC_THRESHOLD}th percentile)")
    avg_sem = sum(len(d.page_content) for d in sem_docs) // len(sem_docs)
    print(f"  Average chunk length: {avg_sem} chars")
    print("  Embedding and storing in FAISS…")
    embed_and_store(sem_docs, embeddings, config.FAISS_DIR_SEMANTIC)
    print(f"  Saved to {config.FAISS_DIR_SEMANTIC}  ({len(sem_docs)} vectors)")

    print(f"""
Summary
  Fixed chunks    : {len(fixed_docs):>5}  avg {avg_fixed} chars
  Semantic chunks : {len(sem_docs):>5}  avg {avg_sem} chars
  (Semantic produces fewer, longer, more coherent chunks)

Next step: python 3_rag_pipeline.py
""")


if __name__ == "__main__":
    main()

"""
Centralised config for Project 2: Financial Document Intelligence Pipeline.
All credentials from environment variables — never hardcode keys.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Nebius Token Factory ─────────────────────────────────────────────────────
# Sign up at https://tokenfactory.nebius.com/ and set NEBIUS_API_KEY.
NEBIUS_API_KEY     = os.environ.get("NEBIUS_API_KEY", "")
NEBIUS_BASE_URL    = "https://api.tokenfactory.nebius.com/v1/"
NEBIUS_LLM_MODEL   = "meta-llama/Llama-3.3-70B-Instruct"
NEBIUS_EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"

# ── Company selection ─────────────────────────────────────────────────────────
# CIKs kept for reference; not used for downloading at runtime.
COMPANY_CIKS = {
    "Apple":     "0000320193",
    "Microsoft": "0000789019",
}
TARGET_COMPANY = "Apple"

# ── Data / Storage ────────────────────────────────────────────────────────────
DATA_DIR          = "./data"
FAISS_DIR_FIXED   = "./faiss_fixed"    # FAISS index for fixed-size chunks
FAISS_DIR_SEMANTIC = "./faiss_semantic" # FAISS index for semantic chunks

# ── Chunking ──────────────────────────────────────────────────────────────────
FIXED_CHUNK_SIZE    = 512   # tokens (approx characters / 4)
FIXED_CHUNK_OVERLAP = 50
SEMANTIC_BREAKPOINT = "percentile"  # SemanticChunker breakpoint type
SEMANTIC_THRESHOLD  = 95            # percentile for topic-shift detection

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RETRIEVE    = 10   # chunks retrieved before reranking
TOP_K_AFTER_RERANK = 3   # chunks passed to LLM after reranking
RERANKER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # local, no API key

# ── Output ────────────────────────────────────────────────────────────────────
RESULTS_FILE = "evaluation_results.json"
REPORT_FILE  = "comparison_report.md"

"""
RAG pipeline for financial documents.

Supports four query modes via two toggles:
  - chunking_strategy : "fixed" | "semantic"
  - use_reranker      : True | False

Query flow:
  1. Embed the question via Nebius Token Factory.
  2. Retrieve top-K chunks from FAISS (vector) + BM25 (keyword), fused via RRF.
  3. (Optional) Re-rank with a local cross-encoder; keep top-3.
  4. Send chunks + question to Nebius LLM; return grounded answer.

Usage:
    from rag_pipeline import FinancialRAGPipeline
    pipe = FinancialRAGPipeline(chunking_strategy="semantic", use_reranker=True)
    result = pipe.query("What was Apple's total revenue in FY2023?")
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import HumanMessage, SystemMessage
from sentence_transformers import CrossEncoder

import config


SYSTEM_PROMPT = """You are a precise financial analyst assistant.
Answer the user's question using ONLY the retrieved document excerpts below.
Cite specific figures, dates, and section names where available.
If the excerpts do not contain enough information to answer, say:
"The retrieved context does not contain enough information to answer this question."
Do not speculate or add information from outside the provided excerpts."""


class FinancialRAGPipeline:
    def __init__(
        self,
        chunking_strategy: str = "fixed",   # "fixed" or "semantic"
        use_reranker: bool = False,
    ) -> None:
        if not config.NEBIUS_API_KEY:
            raise EnvironmentError("NEBIUS_API_KEY not set. See README.")

        self.strategy     = chunking_strategy
        self.use_reranker = use_reranker

        # Nebius Token Factory — embeddings
        self.embeddings = OpenAIEmbeddings(
            model=config.NEBIUS_EMBED_MODEL,
            openai_api_base=config.NEBIUS_BASE_URL,
            openai_api_key=config.NEBIUS_API_KEY,
            check_embedding_ctx_length=False,  # Nebius wants raw strings, not pre-tokenized input
        )

        # Nebius Token Factory — generation
        self.llm = ChatOpenAI(
            model=config.NEBIUS_LLM_MODEL,
            openai_api_base=config.NEBIUS_BASE_URL,
            openai_api_key=config.NEBIUS_API_KEY,
            temperature=0,
        )

        # Load the right FAISS index
        faiss_dir = (
            config.FAISS_DIR_FIXED if chunking_strategy == "fixed"
            else config.FAISS_DIR_SEMANTIC
        )
        self.vectorstore = FAISS.load_local(
            faiss_dir,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        # Pull all stored texts for BM25 (done once at init, not per query)
        self._corpus: list[str] = [
            doc.page_content
            for doc in self.vectorstore.docstore._dict.values()
        ]

        # Cross-encoder reranker (local, no API key needed)
        self.reranker = CrossEncoder(config.RERANKER_MODEL) if use_reranker else None

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, question: str) -> list[str]:
        """Retrieve top-K chunks using hybrid vector similarity + BM25 (Reciprocal Rank Fusion)."""
        # ── Vector search via Nebius embeddings ──────────────────────────────
        vector_results = self.vectorstore.similarity_search_with_score(
            question, k=config.TOP_K_RETRIEVE
        )
        vector_docs = [doc.page_content for doc, _ in vector_results]

        # ── BM25 keyword search via rank_bm25 ────────────────────────────────
        tokenized_corpus = [re.findall(r"\b\w+\b", doc.lower()) for doc in self._corpus]
        tokenized_query  = re.findall(r"\b\w+\b", question.lower())

        bm25        = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_docs   = [
            doc for _, doc in sorted(zip(bm25_scores, self._corpus), reverse=True)
        ][: config.TOP_K_RETRIEVE * 2]

        # ── Reciprocal Rank Fusion ────────────────────────────────────────────
        combined: dict[str, float] = {}
        for i, doc in enumerate(vector_docs):
            combined[doc] = combined.get(doc, 0.0) + 1.0 / (i + 1)
        for i, doc in enumerate(bm25_docs):
            combined[doc] = combined.get(doc, 0.0) + 1.0 / (i + 1)

        return [
            doc for doc, _ in sorted(combined.items(), key=lambda x: x[1], reverse=True)
        ][: config.TOP_K_RETRIEVE]

    # ── Reranking ─────────────────────────────────────────────────────────────

    def rerank(self, question: str, chunks: list[str]) -> list[str]:
        """Score each (question, chunk) pair with the cross-encoder; return top-N."""
        pairs  = [(question, chunk) for chunk in chunks]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in ranked[: config.TOP_K_AFTER_RERANK]]

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(self, question: str, context_chunks: list[str]) -> str:
        context = "\n\n---\n\n".join(context_chunks)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"Document excerpts:\n\n{context}\n\nQuestion: {question}"
            ),
        ]
        response = self.llm.invoke(messages)
        return response.content.strip()

    # ── Public interface ──────────────────────────────────────────────────────

    def query(self, question: str, verbose: bool = False) -> dict:
        # 1. Retrieve
        retrieved = self.retrieve(question)

        # 2. Optionally rerank
        if self.use_reranker:
            final_chunks = self.rerank(question, retrieved)
            reranked = True
        else:
            final_chunks = retrieved[: config.TOP_K_AFTER_RERANK]
            reranked = False

        if verbose:
            print(f"  Retrieved {len(retrieved)} chunks → kept {len(final_chunks)} "
                  f"{'(after rerank)' if reranked else '(no rerank)'}")

        # 3. Generate
        answer = self.generate(question, final_chunks)

        return {
            "question":           question,
            "chunking_strategy":  self.strategy,
            "use_reranker":       reranked,
            "retrieved_count":    len(retrieved),
            "final_chunks":       final_chunks,
            "answer":             answer,
        }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for strategy in ("fixed", "semantic"):
        for rerank in (False, True):
            label = f"{strategy} / {'rerank' if rerank else 'no rerank'}"
            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"{'='*60}")
            pipe   = FinancialRAGPipeline(strategy, rerank)
            result = pipe.query(
                "What was Apple's total net revenue in FY2023?",
                verbose=True,
            )
            print(f"  Answer: {result['answer'][:300]}…")

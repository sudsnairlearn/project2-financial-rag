# Project 2: Financial Document Intelligence Pipeline

**Track:** LangChain (Code-heavy)  
**Corpus:** Apple 10-K annual filing (SEC EDGAR, public, free)  
**Nebius Token Factory:** Embeddings (`BAAI/bge-en-icl`) + Generation (`meta-llama/Meta-Llama-3.1-70B-Instruct`)  
**Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (runs locally, no API key)  

---

## What this builds

A RAG pipeline over Apple's 10-K filing evaluated across a **2 × 2 grid**:

| | No Reranking | With Reranking |
|---|---|---|
| **Fixed chunks** (512 tok) | Condition A | Condition B |
| **Semantic chunks** | Condition C | Condition D |

15 queries run against all four conditions. The output is a `comparison_report.md`
showing which approach wins for each query type and why.

Retrieval is now hybrid: it combines vector similarity with a lightweight BM25 keyword pass before the optional reranker step.

The main user experience is a Streamlit chat UI in `app.py`, where you can ask questions about the filing and compare the fixed vs. semantic retrieval strategies.

---

## Setup

The project now includes helper scripts to create a local virtual environment and install dependencies for you:
- `setup_env.ps1` for Windows PowerShell
- `setup_env.sh` for macOS/Linux
- `.gitignore` to keep `.venv/` and Python cache files out of version control

### 1. Create a Python virtual environment and install dependencies

On Windows PowerShell:
```powershell
.\setup_env.ps1
```

On macOS/Linux:
```bash
./setup_env.sh
```

If you prefer, you can also create the venv manually:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure credentials

Copy the example file and add your Nebius API key.

On macOS/Linux:
```bash
cp .env.example .env
```

On Windows PowerShell:
```powershell
Copy-Item .env.example .env
```

Then edit `.env` and set:
```dotenv
NEBIUS_API_KEY=v1.your_nebius_api_key_here
```

`config.py` loads `.env` automatically, so you do not need to export the key manually.

Get your **Nebius API key** at https://tokenfactory.nebius.com/

### 3. Place the 10-K text file in the data folder

Put the raw 10-K file in `data/` as one of:
- `data/sample_10k.txt` when using the bundled sample document
- `data/apple_10k.txt` or `data/apple_10k.pdf` for Apple
- `data/microsoft_10k.txt` or `data/microsoft_10k.pdf` for Microsoft

The pipeline assumes the file already exists and will not download it automatically.

> Note: PDF input must be text-extractable. Scanned image-only PDFs are not supported by the current pipeline.

### 4. Chunk and embed (creates both Chroma collections)

```bash
python 2_chunk_and_embed.py
```

This runs two chunking strategies and embeds both via Nebius Token Factory.
Expect ~3–5 minutes for a full 10-K (many embedding API calls for semantic chunking).

### 5. Launch the Streamlit UI

```bash
streamlit run app.py
```

This starts the interactive chat interface for querying the financial RAG pipeline.
If you are using the project virtual environment, activate it first or run the executable from `.venv`.

### 6. Run the full evaluation (optional)

```bash
python 4_run_evaluation.py
```

Outputs:
- `evaluation_results.json` — raw answers + scores for all 60 query-condition pairs
- `comparison_report.md` — your submission document

---

## The 15 Queries

| # | Category | Predicted winner |
|---|----------|-----------------|
| 1 | Numerical lookup — total revenue | Fixed |
| 2 | Segmented table — product line revenue | Fixed |
| 3 | Derived metric — Services gross margin % | Fixed |
| 4 | Risk factor narrative — supply chain | Semantic |
| 5 | Strategic narrative — competitive position | Semantic |
| 6 | Forward-looking statements — guidance | Semantic |
| 7 | Policy statement — capital return / buybacks | Semantic |
| 8 | Multi-year comparison — 3-year revenue | Fixed |
| 9 | Balance sheet — total assets & liabilities | Fixed |
| 10 | Expense breakdown — R&D and SG&A | Fixed |
| 11 | Debt schedule — maturity by year | Fixed |
| 12 | R&D strategy narrative | Semantic |
| 13 | Legal / regulatory disclosures | Semantic |
| 14 | Rerank showcase — ESG commitments | Semantic + Rerank |
| 15 | **Edge case** — generative AI policy (refusal test) | None (should refuse) |

---

## Chunking Strategy Summary

**Fixed (512 tokens, 50 overlap)**
- Consistent chunk size
- Keeps adjacent table rows together → wins on numerical queries
- Breaks mid-paragraph → loses on narrative queries

**Semantic (percentile breakpoint)**
- Variable chunk size, aligned to topic shifts
- Keeps full paragraphs together → wins on qualitative queries
- Can split inside tables at embedding similarity thresholds

**Reranking**
- Retrieves top-10 by vector similarity, re-scores with a cross-encoder
- Helps when the right chunk is not the closest embedding match
- Most effective on ambiguous queries and ESG / policy content

---

## Manually test a single query (optional)

```python
from dotenv import load_dotenv
load_dotenv()

from rag_pipeline import FinancialRAGPipeline

# Try all four conditions on one question
q = "What supply chain risks does Apple cite in its risk factors section?"
for strategy in ("fixed", "semantic"):
    for rerank in (False, True):
        pipe = FinancialRAGPipeline(strategy, rerank)
        r = pipe.query(q, verbose=True)
        label = f"{strategy}/{'rerank' if rerank else 'no rerank'}"
        print(f"\n[{label}]\n{r['answer']}\n")
```

---


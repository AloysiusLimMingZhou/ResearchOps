# ResearchOps RAG Pipeline

ResearchOps is a modular Retrieval-Augmented Generation (RAG) systems project focused on understanding, evaluating, and improving retrieval before hiding failures behind LLM generation.

The project starts deliberately small — transparent parsing, chunking, embeddings, vector search, and retrieval evaluation — then grows toward an **enterprise multimodal RAG architecture inspired by the NVIDIA RAG Blueprint**.

> ResearchOps is not an NVIDIA RAG Blueprint fork and does not require NVIDIA NIMs. The goal is to independently build and benchmark the same architectural capabilities with replaceable components, then optionally compare or integrate production-grade NVIDIA components later.

## Current Pipeline

```text
PDF
 ↓
PyPDF Parsing
 ↓
Text Cleaning
 ↓
LangChain Recursive Chunking
256 tokens / 64 overlap
 ↓
Sentence-Transformer Embeddings
 ↓
Qdrant
 ↓
Dense Retrieval
 ↓
Top-K Results
 ↓
Custom Retrieval Evaluation
```

Current priorities are still retrieval quality and evaluation. LLM generation is intentionally deferred until retrieval behavior is strong enough to measure and reason about.

---

## Architecture Direction

ResearchOps is being built toward the same broad separation of concerns used by enterprise RAG systems such as the NVIDIA RAG Blueprint:

```text
                           ┌──────────────────────────────┐
                           │        USER / CLIENT         │
                           └──────────────┬───────────────┘
                                          │
                                          ↓
                           ┌──────────────────────────────┐
                           │       Query Processing       │
                           │ filters / rewrite / decompo. │
                           └──────────────┬───────────────┘
                                          │
                         ┌────────────────┴────────────────┐
                         ↓                                 ↓
               ┌─────────────────┐               ┌─────────────────┐
               │ Dense Retrieval │               │ Sparse Retrieval│
               └────────┬────────┘               └────────┬────────┘
                        └────────────────┬─────────────────┘
                                         ↓
                               ┌──────────────────┐
                               │ Fusion / RRF     │
                               └────────┬─────────┘
                                        ↓
                               ┌──────────────────┐
                               │    Reranker      │
                               └────────┬─────────┘
                                        ↓
                               ┌──────────────────┐
                               │ Context Selection│
                               └────────┬─────────┘
                                        ↓
                               ┌──────────────────┐
                               │ LLM / VLM        │
                               │ + Citations      │
                               └──────────────────┘
```

The ingestion side will evolve separately:

```text
Enterprise Documents
PDF / Office / Images / Scans
          │
          ↓
┌─────────────────────────────┐
│ Document Extraction         │
│ text / layout / OCR         │
└──────────────┬──────────────┘
               │
               ├── Text / Sections ──→ Chunking ─────────────┐
               ├── Tables ───────────→ Structured Entity ────┤
               ├── Formulas ─────────→ Formula + Context ────┤
               ├── Charts ───────────→ Chart Representation ─┤
               └── Images ───────────→ VLM Representation ───┤
                                                             ↓
                                                  Embedding / Indexing
                                                             ↓
                                             Vector + Sparse Indexes
                                                             ↓
                                                   Metadata / Object Store
```

Cross-cutting concerns will eventually include:

```text
Evaluation
Observability
Latency / Throughput
Experiment Tracking
Caching
Guardrails
Deployment
```

This direction is inspired by:

- [NVIDIA RAG Blueprint](https://github.com/NVIDIA-AI-Blueprints/rag)
- [NVIDIA: Finding the Best Chunking Strategy for Accurate AI Responses](https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/)

---

## Current Features

- PDF parsing with page-level metadata
- Deterministic document IDs
- Text normalization and cleaning
- Fixed-token and LangChain recursive chunking
- Dense embeddings using Sentence Transformers
- Swappable embedding models
- Qdrant vector storage and cosine similarity search
- Top-K semantic retrieval
- Persistent document and chunk metadata
- CLI-based ingestion, search, and evaluation
- Custom retrieval evaluation using:
  - Hit@K
  - PagePrecision@K
  - PageRecall@K
  - Mean Reciprocal Rank (MRR)

---

## Project Structure

```text
researchops/
│
├── ingestion/
│   ├── parser.py
│   ├── cleaner.py
│   └── chunker.py
│
├── embeddings/
│   └── embedder.py
│
├── retrieval/
│   ├── vector_store.py
│   └── retriever.py
│
├── evaluation/
│   ├── dataset.json
│   └── retrieval_eval.py
│
├── model/
│   └── schema.py
│
├── papers/
│
└── main.py
```

### `model/schema.py`

Defines the core data models used throughout the pipeline.

Main objects include:

- `Document`
- `Page`
- `Chunk`
- `SearchResults`

Metadata such as document ID, filename, page number, chunk index, and chunk text is preserved throughout ingestion and retrieval.

### `ingestion/parser.py`

Current parser responsibilities:

- Read PDF files
- Extract text page by page
- Preserve page numbers
- Generate deterministic SHA-256 document IDs
- Extract basic document metadata

Current parser:

```text
PyPDF
```

Future document-intelligence work will compare this baseline against layout-aware extraction using PyMuPDF / PyMuPDF4LLM.

### `ingestion/cleaner.py`

Normalizes extracted PDF text before chunking.

Current cleaning includes:

- Whitespace normalization
- Line-ending normalization
- Removal of excessive blank lines
- Basic text cleanup

More advanced cleanup such as header/footer removal, OCR repair, section detection, table reconstruction, formula handling, and image extraction is intentionally deferred to the document-intelligence stage.

### `ingestion/chunker.py`

The current default is a token-aware LangChain recursive splitter:

```text
Strategy   : RecursiveCharacterTextSplitter
Tokenizer  : embedding-model Hugging Face tokenizer
Chunk Size : 256 tokens
Overlap    : 64 tokens
```

The splitter prefers natural boundaries:

```text
paragraph
   ↓
newline
   ↓
space
   ↓
hard split
```

ResearchOps also implemented and benchmarked:

- Fixed-token chunking
- Recursive chunking
- Semantic chunking
- Semantic → recursive chunking

The current conclusion is to **freeze recursive 256/64 as the working default** and revisit chunking when the corpus becomes larger and more heterogeneous.

Future chunking work should be driven by concrete document failures rather than additional tuning on a single paper. Examples include:

- Section-aware recursive chunking
- Keeping tables with captions
- Keeping formulas with surrounding explanations
- Preserving figure-caption relationships
- Document-type-specific chunking

### `embeddings/embedder.py`

Converts document chunks and queries into dense vectors.

Embedding models benchmarked so far:

- `BAAI/bge-small-en-v1.5`
- `BAAI/bge-large-en-v1.5`
- `BAAI/bge-m3`
- `Qwen/Qwen3-Embedding-0.6B`

Current working baseline:

```text
BAAI/bge-small-en-v1.5
```

`BAAI/bge-m3` is the strongest current challenger for rank quality, while BGE-small remains a useful lightweight baseline with strong Top-K coverage on the current evaluation set.

Embedding-model comparisons are intentionally performed with chunking held constant.

### `retrieval/vector_store.py`

Handles Qdrant integration.

Responsibilities include:

- Creating experiment-specific collections
- Storing dense vectors
- Storing metadata payloads
- Performing cosine similarity search
- Returning Top-K candidate chunks

Qdrant is intentionally treated as a replaceable backend. A later production version may compare or integrate other vector/search systems without changing retrieval evaluation semantics.

### `retrieval/retriever.py`

Provides the high-level retrieval interface.

```python
results = retriever.retrieve(
    "Why is multi-head attention useful?",
    top_k=5,
)
```

The retriever:

1. Embeds the query
2. Searches the current vector collection
3. Converts retrieved points into structured results
4. Returns ranked passages with similarity scores and metadata

### `evaluation/retrieval_eval.py`

Evaluates retrieval against manually labeled questions.

Current metrics:

- Hit@1 / Hit@3 / Hit@5
- PagePrecision@1 / PagePrecision@3 / PagePrecision@5
- PageRecall@1 / PageRecall@3 / PageRecall@5
- Mean Reciprocal Rank (MRR)

`PagePrecision@K` currently means the fraction of retrieved chunks whose page belongs to the manually labeled expected-page set. Multiple chunks from the same expected page can therefore each count as relevant.

`PageRecall@K` measures how many unique expected pages are represented in Top-K.

A later evaluator will move from page-level labels toward evidence spans and graded relevance.

---

## Example Usage

### Start Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

### Ingest a PDF

```bash
python main.py ingest papers/attention_is_all_you_need.pdf
```

### Search

```bash
python main.py search "Why is multi-head attention useful?"
```

### Evaluate

```bash
python main.py eval
```

---

## Experimental Method

ResearchOps changes **one variable at a time**.

Example:

```text
Experiment A
Chunker   : Recursive 256/64
Embedder  : BGE-small
Retriever : Dense cosine

Experiment B
Chunker   : Recursive 256/64
Embedder  : BGE-M3
Retriever : Dense cosine
```

Only the embedding model changes.

The same principle will later be used for:

- Dense vs sparse retrieval
- Hybrid fusion
- Candidate-set size
- Reranking
- MMR
- Parser/extractor comparisons
- Multimodal retrieval

This prevents improvements from being attributed to multiple simultaneous changes.

---

## Retrieval Failure Modes

ResearchOps intentionally exposes retrieval failures before generation is introduced.

```text
Retrieval Failure
Relevant evidence does not appear in the candidate set.

Ranking Failure
Relevant evidence is retrieved but ranked too low.

Redundancy Failure
Top-K is dominated by near-duplicate chunks.

Extraction Failure
The required evidence was damaged or lost during document parsing.

Representation Failure
The evidence exists but the embedding/index representation does not make it retrievable.
```

These distinctions guide later experiments.

---

# Roadmap Toward Enterprise Multimodal RAG

## V1 — Retrieval Foundation ✅

Goal: understand retrieval mechanics before abstraction.

- [x] PDF parsing
- [x] Text cleaning
- [x] Deterministic document/chunk identity
- [x] Fixed-token chunking
- [x] Recursive token-aware chunking
- [x] Semantic chunking experiments
- [x] Dense embeddings
- [x] Qdrant integration
- [x] Top-K semantic retrieval
- [x] Retrieval evaluation dataset
- [x] Hit@K / PagePrecision@K / PageRecall@K / MRR
- [x] Chunk-size benchmark
- [x] Chunking-strategy benchmark
- [x] Initial embedding-model benchmark

**Frozen working baseline**

```text
Parser      : PyPDF
Chunker     : LangChain Recursive
Chunk Size  : 256
Overlap     : 64
Embedder    : BGE-small-en-v1.5
Vector DB   : Qdrant
Retriever   : Dense cosine similarity
```

---

## V2 — Retrieval Quality

Goal: evolve from single dense retrieval into a strong candidate-retrieval and ranking pipeline.

- [ ] BM25 sparse retrieval
- [ ] Dense vs sparse comparison
- [ ] Hybrid dense + sparse retrieval
- [ ] Reciprocal Rank Fusion (RRF)
- [ ] Retrieve wider candidate sets such as Top-20
- [ ] Cross-encoder reranking
- [ ] Compare candidate recall before and after reranking
- [ ] Maximum Marginal Relevance (MMR)
- [ ] Duplicate/redundancy metrics
- [ ] Evidence-span relevance labels
- [ ] nDCG / graded relevance
- [ ] Metadata filtering
- [ ] Query rewriting / expansion experiments

Target architecture:

```text
Query
 ↓
┌───────────────┬───────────────┐
│ Dense Search  │ Sparse Search │
└───────┬───────┴───────┬───────┘
        └────── RRF ─────┘
               ↓
          Top-N Candidates
               ↓
             Reranker
               ↓
        Context Selection
```

This corresponds conceptually to the NVIDIA Blueprint's hybrid retrieval + reranking layer while remaining implementation-independent.

---

## V3 — Document Intelligence

Goal: stop treating PDFs as flat text and build an ingestion layer suitable for enterprise documents.

Baseline comparison:

```text
PyPDF
  vs
PyMuPDF / PyMuPDF4LLM
```

Planned capabilities:

- [ ] Layout-aware extraction
- [ ] Better reading-order preservation
- [ ] Section and heading metadata
- [ ] Header/footer removal
- [ ] Table detection and reconstruction
- [ ] Table-caption association
- [ ] Formula-region extraction
- [ ] Formula + nearby explanation preservation
- [ ] Figure and image extraction
- [ ] Figure-caption association
- [ ] Selective OCR for scanned or image-based pages
- [ ] Page-level retrieval baseline
- [ ] Structure-aware recursive chunking
- [ ] Multi-document ingestion
- [ ] Object-store representation for extracted visual assets

Target ingestion model:

```text
Document
 ↓
Extraction
 ├── Text / Sections
 ├── Tables
 ├── Formulas
 ├── Charts
 └── Images
 ↓
Normalized Document Elements
 ↓
Element-specific representation
 ↓
Indexing
```

This is the stage where ResearchOps begins to resemble NVIDIA NeMo Retriever Extraction conceptually, while using local/open-source components first.

---

## V4 — Multimodal Retrieval

Goal: retrieve evidence using both textual and visual representations.

- [ ] Text embeddings
- [ ] Image / VLM embeddings
- [ ] Table and chart retrieval
- [ ] Figure retrieval
- [ ] Caption-based retrieval baseline
- [ ] Native multimodal embedding experiment
- [ ] Multimodal reranking
- [ ] Compare text-only vs multimodal retrieval
- [ ] Multimodal evaluation dataset

Potential experiment:

```text
Image / Chart Query
        │
        ├── Caption → Text Embedding
        │
        ├── Native VLM Embedding
        │
        └── Hybrid
                ↓
        Multimodal Reranker
```

This maps closely to the NVIDIA Blueprint pattern of independently switchable VLM embedding and VLM reranking.

---

## V5 — Grounded Generation

Goal: add generation only after retrieval and ranking are measurable.

- [ ] LLM answer generation
- [ ] Pydantic structured outputs
- [ ] Evidence-grounded prompts
- [ ] Source citations
- [ ] Citation verification
- [ ] Answer correctness evaluation
- [ ] Faithfulness evaluation
- [ ] Context relevance evaluation
- [ ] Optional reflection / answer verification
- [ ] Multi-turn research sessions

Target:

```text
Retrieved + Reranked Evidence
            ↓
       Context Builder
            ↓
            LLM
            ↓
Structured Answer + Citations
```

---

## V6 — Production ResearchOps

Goal: evolve from a local experiment into a deployable RAG service.

- [ ] FastAPI / OpenAI-compatible API layer
- [ ] Separate ingestion and query workflows
- [ ] Persistent document registry
- [ ] Background ingestion jobs
- [ ] Redis caching / coordination where useful
- [ ] Object storage for original documents and extracted assets
- [ ] Retrieval experiment tracking
- [ ] OpenTelemetry tracing
- [ ] Metrics and dashboards
- [ ] p50 / p95 / p99 latency
- [ ] Throughput and concurrency tests
- [ ] Cost-per-query measurements
- [ ] Docker deployment
- [ ] AWS deployment
- [ ] Authentication / authorization
- [ ] Guardrails where required

A later deployment may decompose components into services similar to an enterprise RAG orchestrator + ingestor architecture.

---

## V7 — Enterprise / Agentic Extensions

Goal: add orchestration only after retrieval, document understanding, evaluation, and operations are strong.

- [ ] Multi-collection retrieval
- [ ] Query decomposition
- [ ] Dynamic metadata filtering
- [ ] Cross-document research queries
- [ ] Plan-and-execute research workflow
- [ ] Verification / reflection
- [ ] Streaming intermediate stages
- [ ] Kubernetes / Helm deployment experiments
- [ ] Horizontal scaling
- [ ] Performance benchmarking
- [ ] Optional NVIDIA NIM / NeMo component comparison

Agentic RAG is intentionally a late-stage capability rather than the foundation of the project.

---

## ResearchOps ↔ NVIDIA RAG Blueprint

ResearchOps uses the NVIDIA RAG Blueprint as a **reference architecture**, not as a required implementation.

| ResearchOps | NVIDIA Blueprint Concept | Status |
|---|---|---|
| PyPDF parser | Document extraction | ✅ Baseline |
| PyMuPDF4LLM / layout pipeline | NeMo Retriever Extraction / parsing | Planned V3 |
| Sentence Transformers | Retriever embedding service | ✅ Baseline |
| Qdrant | Pluggable vector database | ✅ Baseline |
| Dense retrieval | Dense retrieval | ✅ Baseline |
| BM25 + RRF | Hybrid dense + sparse search | Planned V2 |
| Cross-encoder reranker | Retriever reranking service | Planned V2 |
| Custom retrieval evaluator | RAG evaluation | ✅ / expanding |
| Text + visual representations | Multimodal retrieval | Planned V4 |
| Multimodal reranker | VLM reranker | Planned V4 |
| LLM + citations | Grounded answer generation | Planned V5 |
| FastAPI / jobs / object storage | RAG + ingestion services | Planned V6 |
| OpenTelemetry | Telemetry / observability | Planned V6 |
| Query decomposition | Query processing | Planned V7 |
| LangGraph-style research workflow | Agentic RAG | Planned V7 |
| Docker / AWS / Kubernetes | Enterprise deployment | Planned V6–V7 |

---

## Design Philosophy

### 1. Retrieval before generation

Do not use an LLM to hide weak retrieval.

### 2. One variable at a time

Chunking, embedding, retrieval, reranking, and parsing changes should be evaluated independently whenever possible.

### 3. Components should be replaceable

```text
Parser
 ↓
Cleaner
 ↓
Chunker
 ↓
Embedder
 ↓
Retriever
 ↓
Reranker
 ↓
Context Builder
 ↓
Generator
```

Each stage should expose a stable interface so implementations can change without rewriting the full pipeline.

### 4. Document intelligence belongs before chunking

Tables, formulas, OCR, charts, and images should be represented correctly during extraction rather than forcing the chunker to repair malformed document structure.

### 5. Candidate recall and final ranking are different problems

```text
Retriever:
Do not miss the evidence.

Reranker:
Put the best evidence first.
```

### 6. Multimodal RAG is more than image captioning

ResearchOps should eventually compare caption-based retrieval, native multimodal embeddings, multimodal reranking, and multimodal generation.

### 7. Evaluate quality and systems performance separately

Future evaluation will cover:

```text
Retrieval Quality
Hit@K
PageRecall@K
MRR
nDCG

Generation Quality
Answer correctness
Faithfulness
Citation correctness

System Performance
Latency
Throughput
Concurrency
Cost
Indexing time
```

---

## Current Status

ResearchOps has completed its initial retrieval-foundation phase.

The project can:

- ingest research-paper PDFs
- clean and chunk text
- generate dense embeddings
- maintain experiment-specific Qdrant collections
- retrieve semantically related evidence
- compare chunking and embedding configurations
- measure retrieval behavior directly

The next major focus is **V2 Retrieval Quality**, followed by **V3 Document Intelligence**.

---

## Tech Stack

### Current

- Python
- PyPDF
- Pydantic
- Hugging Face Transformers
- Sentence Transformers
- LangChain text splitters
- Qdrant
- Docker

### Planned / Experimental

- BM25 / sparse retrieval
- Reciprocal Rank Fusion
- Cross-encoder reranking
- PyMuPDF / PyMuPDF4LLM
- OCR
- Vision-language embeddings
- Multimodal reranking
- FastAPI
- Redis
- Object storage
- OpenTelemetry
- AWS
- Kubernetes / Helm
- Optional NVIDIA NIM / NeMo integrations

---

## Long-Term Goal

ResearchOps will evolve from a transparent dense-retrieval experiment into a modular enterprise-grade research RAG platform supporting:

- hybrid dense + sparse retrieval
- reranking and context optimization
- layout-aware document intelligence
- tables, formulas, charts, images, and OCR
- multimodal retrieval and generation
- evidence-grounded answers with citations
- multi-document research workflows
- retrieval and generation evaluation
- latency / throughput benchmarking
- observability and experiment tracking
- production deployment
- optional enterprise NVIDIA RAG Blueprint component integration

The project remains deliberately evaluation-first: architecture becomes more sophisticated only when a measurable problem justifies the additional complexity.

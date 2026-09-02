# ResearchOps RAG Pipeline

ResearchOps is a modular Retrieval-Augmented Generation (RAG) project focused on understanding, evaluating, and improving the retrieval layer before introducing LLM-based generation.

The current version implements the core retrieval pipeline:

```text
PDF
 ↓
Parsing
 ↓
Text Cleaning
 ↓
Token-Based Chunking
 ↓
Embeddings
 ↓
Qdrant Vector Database
 ↓
Dense Retrieval
 ↓
Top-K Results
```

The goal of this stage is to build a strong retrieval baseline and experimentally study how chunking, embeddings, ranking, and retrieval strategies affect RAG performance.

---

## Current Features

* PDF parsing with page-level metadata
* Text normalization and cleaning
* Token-based chunking with overlap
* Dense embeddings using Sentence Transformers
* Vector storage and similarity search with Qdrant
* Top-K semantic retrieval
* Persistent document and chunk metadata
* CLI-based document ingestion and search
* Foundation for retrieval evaluation using Hit@K, Recall@K, and MRR

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
├── models/
│   └── document.py
│
├── papers/
│
└── main.py
```

### `models/document.py`

Defines the core data models used throughout the pipeline.

Main objects include:

* `Document`
* `Page`
* `Chunk`
* `SearchResult`

Metadata such as document ID, filename, page number, and chunk index is preserved throughout ingestion and retrieval.

---

### `ingestion/parser.py`

Parses PDF files into structured documents.

Current responsibilities:

* Read PDF files
* Extract text page by page
* Preserve page numbers
* Generate deterministic document IDs
* Extract basic document metadata

---

### `ingestion/cleaner.py`

Normalizes extracted PDF text before chunking.

Current cleaning includes:

* Whitespace normalization
* Line-ending normalization
* Removal of excessive blank lines
* Basic text cleanup

More advanced PDF cleanup such as section detection, header/footer removal, and table handling is intentionally deferred.

---

### `ingestion/chunker.py`

Splits documents into token-based chunks.

Current configuration:

```text
Chunk size: 384 tokens
Overlap: 64 tokens
```

Each chunk preserves:

* Document ID
* Filename
* Page number
* Chunk index
* Text

The current fixed-token chunker serves as the retrieval baseline before experimenting with recursive, section-aware, or semantic chunking.

---

### `embeddings/embedder.py`

Converts document chunks and user queries into dense vector representations.

Current embedding model:

```text
BAAI/bge-small-en-v1.5
```

Documents and queries are encoded separately for semantic retrieval.

---

### `retrieval/vector_store.py`

Handles communication with Qdrant.

Responsibilities include:

* Creating the vector collection
* Storing embeddings
* Storing chunk metadata as payloads
* Performing cosine similarity search
* Returning top-K matching chunks

---

### `retrieval/retriever.py`

Provides the high-level retrieval interface.

Example:

```python
results = retriever.retrieve(
    "Why is multi-head attention useful?",
    top_k=5,
)
```

The retriever:

1. Embeds the query
2. Searches Qdrant
3. Converts retrieved points into structured search results
4. Returns ranked chunks with similarity scores and metadata

---

### `evaluation/retrieval_eval.py`

Evaluates retrieval quality against manually labeled questions.

Planned and current metrics include:

* Hit@1
* Hit@3
* Hit@5
* Recall@K
* Mean Reciprocal Rank (MRR)

The purpose of evaluation is to compare retrieval configurations experimentally instead of selecting parameters arbitrarily.

---

## Example Usage

### Start Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

---

### Ingest a PDF

```bash
python main.py ingest papers/attention_is_all_you_need.pdf
```

Example output:

```text
Parsed 15 pages
Cleaned Document

Page 1: 627 tokens
Page 2: 861 tokens
Page 3: 419 tokens
...

Created 36 chunks
Generated 36 embeddings
Stored 36 chunks into Qdrant
```

---

### Search

```bash
python main.py search "Why is multi-head attention useful?"
```

Example result:

```text
============================================================
#1 | Score: 0.7650
Page 5

Multi-head attention allows the model to jointly attend
to information from different representation subspaces...
```

---

## Current Retrieval Experiments

The current baseline uses:

```text
Parser        : PyPDF
Chunking      : Fixed token
Chunk Size    : 384
Overlap       : 64
Embedding     : BGE-small-en-v1.5
Retrieval     : Dense vector search
Similarity    : Cosine similarity
Top-K         : 5
Vector DB     : Qdrant
```

One of the main objectives is to study retrieval failure modes instead of immediately hiding them behind an LLM.

For example:

```text
Query:
"What optimizer did they use?"

Expected:
Adam optimizer

Observed:
Relevant optimizer chunk may appear below another
training-related chunk.
```

This demonstrates an important distinction between:

```text
Retrieval Failure
Relevant evidence is not retrieved at all.

vs.

Ranking Failure
Relevant evidence is retrieved, but ranked too low.
```

These cases will later be used to evaluate reranking and hybrid retrieval techniques.

---

## Planned Experiments

### Chunking

Compare:

```text
256 tokens
384 tokens
512 tokens
768 tokens
```

with different overlap values.

Future chunking strategies:

* Sentence-aware chunking
* Recursive chunking
* Section-aware chunking
* Semantic chunking

---

### Retrieval

Compare:

```text
Dense Retrieval
vs
BM25
vs
Dense + BM25
```

Future retrieval techniques include:

* Reciprocal Rank Fusion (RRF)
* Cross-encoder reranking
* Maximum Marginal Relevance (MMR)
* Metadata filtering
* Query rewriting
* Query expansion

---

### Evaluation

Build a manually labeled evaluation dataset containing:

```json
{
  "question": "What optimizer was used during training?",
  "expected_pages": [7]
}
```

Metrics will be used to measure whether retrieval changes actually improve performance.

---

## Roadmap

### R0 — Dense Retrieval Baseline

* [x] PDF parsing
* [x] Text cleaning
* [x] Token-based chunking
* [x] Dense embeddings
* [x] Qdrant integration
* [x] Top-K semantic search
* [ ] Retrieval evaluation dataset
* [ ] Hit@K / MRR evaluation

### R1 — Chunking Experiments

* [ ] Compare chunk sizes
* [ ] Compare overlap values
* [ ] Measure semantic dilution
* [ ] Explore section-aware chunking

### R2 — Hybrid Retrieval

* [ ] BM25
* [ ] Dense vs sparse comparison
* [ ] Reciprocal Rank Fusion

### R3 — Reranking

* [ ] Cross-encoder reranker
* [ ] Compare pre-rerank and post-rerank MRR
* [ ] Tune retrieval candidate size

### R4 — Retrieval Diversity

* [ ] Maximum Marginal Relevance
* [ ] Duplicate chunk reduction
* [ ] Context diversity experiments

### R5 — Generation

* [ ] LLM answer generation
* [ ] Pydantic structured outputs
* [ ] Source citations
* [ ] Grounded response validation

### R6 — Productionization

* [ ] API layer
* [ ] Persistent ingestion jobs
* [ ] Observability
* [ ] Retrieval experiment tracking
* [ ] Caching
* [ ] Multimodal document ingestion

---

## Design Philosophy

ResearchOps intentionally separates each RAG component:

```text
Parsing
   ↓
Cleaning
   ↓
Chunking
   ↓
Embedding
   ↓
Retrieval
   ↓
Reranking
   ↓
Generation
```

This makes individual components replaceable and independently measurable.

Instead of asking:

> "Does my RAG system work?"

ResearchOps aims to answer more specific engineering questions:

```text
Did the parser preserve useful information?

Did chunking preserve semantic boundaries?

Did retrieval find the relevant evidence?

Was the relevant evidence ranked highly enough?

Did hybrid retrieval improve recall?

Did reranking improve precision?

Did the final LLM answer remain grounded in retrieved evidence?
```

The project therefore treats RAG primarily as an information retrieval and systems engineering problem rather than only an LLM prompting problem.

---

## Current Status

ResearchOps is currently in the **retrieval baseline stage**.

The system can ingest research papers, generate dense embeddings, store chunks in Qdrant, and retrieve semantically related passages.

LLM generation is intentionally not yet included so retrieval behavior and failure modes can be inspected directly.

---

## Tech Stack

* Python
* PyPDF
* Pydantic
* Hugging Face Transformers
* Sentence Transformers
* BAAI BGE Embeddings
* Qdrant
* Docker

---

## Long-Term Goal

ResearchOps will evolve from a basic dense retrieval pipeline into an experimental research assistant platform supporting:

* High-quality document retrieval
* Hybrid search
* Reranking
* Retrieval evaluation
* Citation-grounded generation
* Multimodal RAG
* Research workflows
* Production-grade deployment and observability

## Retrieval Evaluation Experiments

> Benchmark scope: `attention_is_all_you_need.pdf`, 25 retrieval questions, Top-5 dense retrieval in Qdrant with cosine distance. Unless otherwise stated, unlabeled embedding runs use `BAAI/bge-small-en-v1.5`.

### Metric definitions

- **Hit@K** — fraction of queries with at least one retrieved chunk from an expected page in the Top-K.
- **PagePrecision@K** — renamed from the evaluator's current `Precision@K`: fraction of the Top-K retrieved **chunks** whose `page_number` is in the query's expected-page set. This is page-labelled chunk precision; duplicate chunks from the same expected page are counted separately.
- **PageRecall@K** — fraction of unique expected pages represented in the Top-K results.
- **MRR** — mean reciprocal rank of the first result whose page is in the expected-page set.

> **Metric caveat:** `PagePrecision@K` is not deduplicated by page. A future evidence-span evaluator should replace page-only relevance for stronger semantic evaluation.

### Overall results

| ID | Chunking | Chunk size | Overlap | Embedding | Hit@1 | PagePrecision@1 | PageRecall@1 | Hit@3 | PagePrecision@3 | PageRecall@3 | Hit@5 | PagePrecision@5 | PageRecall@5 | MRR |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E01 | Fixed | 384 | 64 | `BAAI/bge-small-en-v1.5` | 0.640 | 0.640 | 0.540 | 0.840 | 0.440 | 0.793 | 0.920 | 0.352 | 0.907 | 0.751 |
| E02 | Fixed | 512 | 64 | `BAAI/bge-small-en-v1.5` | 0.600 | 0.600 | 0.520 | 0.800 | 0.387 | 0.767 | 0.840 | 0.280 | 0.827 | 0.708 |
| E03 | Fixed | 128 | 64 | `BAAI/bge-small-en-v1.5` | 0.840 | 0.840 | 0.713 | 0.920 | 0.720 | 0.887 | 0.960 | 0.616 | 0.947 | 0.890 |
| E04 | Fixed | 256 | 64 | `BAAI/bge-small-en-v1.5` | 0.840 | 0.840 | 0.740 | 0.920 | 0.493 | 0.833 | 0.960 | 0.400 | 0.947 | 0.881 |
| E05 | Semantic (p90) | Variable | — | `BAAI/bge-small-en-v1.5` | 0.600 | 0.600 | 0.473 | 0.720 | 0.307 | 0.667 | 0.880 | 0.240 | 0.847 | 0.698 |
| E06 | Semantic → Recursive | 256 cap | 64 | `BAAI/bge-small-en-v1.5` | 0.760 | 0.760 | 0.633 | 0.840 | 0.560 | 0.787 | 0.920 | 0.424 | 0.907 | 0.820 |
| E07 | Recursive | 256 | 64 | `BAAI/bge-small-en-v1.5` | 0.800 | 0.800 | 0.700 | 0.920 | 0.547 | 0.853 | 0.960 | 0.432 | 0.947 | 0.863 |
| E08 | Recursive | 256 | 64 | `BAAI/bge-large-en-v1.5` | 0.640 | 0.640 | 0.553 | 0.880 | 0.560 | 0.813 | 0.920 | 0.400 | 0.887 | 0.761 |
| E09 | Recursive | 256 | 64 | `BAAI/bge-m3` | 0.840 | 0.840 | 0.713 | 0.920 | 0.613 | 0.893 | 0.920 | 0.432 | 0.893 | 0.873 |
| E10 | Recursive | 256 | 64 | `Qwen/Qwen3-Embedding-0.6B` | 0.800 | 0.800 | 0.693 | 0.920 | 0.520 | 0.847 | 0.920 | 0.408 | 0.867 | 0.853 |

### Current preliminary takeaways

- `Fixed 128/64` has the highest raw MRR (`0.890`) but also showed substantial duplicate-page redundancy in earlier diagnostics, so raw PagePrecision can overstate its practical advantage.
- `Fixed 256/64` remains the strongest simple fixed-size trade-off (`MRR 0.881`, `Hit@5 0.960`).
- `Recursive 256/64 + BGE-small` is the current practical default: near-fixed performance while respecting natural text boundaries (`MRR 0.863`, `Hit@5 0.960`).
- Among the recursive embedding runs, `BGE-M3` has the strongest early ranking (`Hit@1 0.840`, `PagePrecision@3 0.613`, `PageRecall@3 0.893`, `MRR 0.873`), while `BGE-small` has the strongest Top-5 coverage (`Hit@5 0.960`, `PageRecall@5 0.947`).
- Results are preliminary because the benchmark currently contains one document and 25 questions. Re-run chunk-size × embedding interactions only after expanding the corpus.
- Before declaring an embedding model a final winner, verify its model-specific query prompt/instruction contract and record it in the experiment config; the current comparison should be interpreted as performance under the exact `Embedder` implementation used for these runs.

### Query legend

| Q | Question |
|---:|---|
| Q1 | What architecture do the authors propose instead of recurrent and convolutional neural networks? |
| Q2 | Why are recurrent neural networks difficult to parallelize during training? |
| Q3 | How does the Transformer handle long-range dependencies without recurrence? |
| Q4 | How many layers are used in the Transformer encoder? |
| Q5 | How does the decoder prevent a position from attending to future output positions? |
| Q6 | How is scaled dot-product attention calculated? |
| Q7 | Why are the attention dot products divided by the square root of the key dimension? |
| Q8 | Why is multi-head attention useful? |
| Q9 | How many attention heads are used in the base Transformer model? |
| Q10 | What are the three ways multi-head attention is used inside the Transformer? |
| Q11 | What activation function is used inside the position-wise feed-forward network? |
| Q12 | What are the input and hidden dimensions of the position-wise feed-forward network? |
| Q13 | How does the Transformer represent the positions of tokens in the sequence? |
| Q14 | Why did the authors choose sinusoidal positional encodings? |
| Q15 | How did learned positional embeddings compare with sinusoidal positional encodings? |
| Q16 | Why can self-attention make learning long-range dependencies easier? |
| Q17 | What datasets were used to train the English-German and English-French translation models? |
| Q18 | What hardware was used to train the Transformer models? |
| Q19 | What optimizer did they use? |
| Q20 | What learning algorithm did they use to update the model parameters? |
| Q21 | How was the learning rate changed during training? |
| Q22 | How many warmup steps were used for the learning rate schedule? |
| Q23 | What regularization techniques were used when training the Transformer? |
| Q24 | What beam size and length penalty were used for machine translation inference? |
| Q25 | What happened when the number of attention heads was changed? |

### Per-query Top-5 relevance

`Correct` contains ranks whose retrieved chunk came from an expected page; `Incorrect` contains the remaining Top-5 ranks.

<details>
<summary><strong>E01 — Fixed — BAAI/bge-small-en-v1.5</strong></summary>

- **Q1:** Correct: `[R4, R5]`; Incorrect: `[R1, R2, R3]`
- **Q2:** Correct: `[R1, R5]`; Incorrect: `[R2, R3, R4]`
- **Q3:** Correct: `[R2, R4, R5]`; Incorrect: `[R1, R3]`
- **Q4:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q5:** Correct: `[R1, R2, R3, R5]`; Incorrect: `[R4]`
- **Q6:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q7:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q8:** Correct: `[R1, R2, R3]`; Incorrect: `[R4, R5]`
- **Q9:** Correct: `[R2]`; Incorrect: `[R1, R3, R4, R5]`
- **Q10:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q11:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q12:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q13:** Correct: `[R5]`; Incorrect: `[R1, R2, R3, R4]`
- **Q14:** Correct: `[R2, R4]`; Incorrect: `[R1, R3, R5]`
- **Q15:** Correct: `[R1, R2, R4]`; Incorrect: `[R3, R5]`
- **Q16:** Correct: `[R1, R3]`; Incorrect: `[R2, R4, R5]`
- **Q17:** Correct: `[R3]`; Incorrect: `[R1, R2, R4, R5]`
- **Q18:** Correct: `[]`; Incorrect: `[R1, R2, R3, R4, R5]`
- **Q19:** Correct: `[R2]`; Incorrect: `[R1, R3, R4, R5]`
- **Q20:** Correct: `[R1, R4]`; Incorrect: `[R2, R3, R5]`
- **Q21:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q22:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q23:** Correct: `[R1, R2, R4]`; Incorrect: `[R3, R5]`
- **Q24:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q25:** Correct: `[]`; Incorrect: `[R1, R2, R3, R4, R5]`

</details>

<details>
<summary><strong>E02 — Fixed — BAAI/bge-small-en-v1.5</strong></summary>

- **Q1:** Correct: `[R2]`; Incorrect: `[R1, R3, R4, R5]`
- **Q2:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q3:** Correct: `[R2, R3]`; Incorrect: `[R1, R4, R5]`
- **Q4:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q5:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q6:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q7:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q8:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q9:** Correct: `[R2]`; Incorrect: `[R1, R3, R4, R5]`
- **Q10:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q11:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q12:** Correct: `[R1, R5]`; Incorrect: `[R2, R3, R4]`
- **Q13:** Correct: `[R5]`; Incorrect: `[R1, R2, R3, R4]`
- **Q14:** Correct: `[R2, R3]`; Incorrect: `[R1, R4, R5]`
- **Q15:** Correct: `[R1, R2, R3]`; Incorrect: `[R4, R5]`
- **Q16:** Correct: `[R1, R2, R5]`; Incorrect: `[R3, R4]`
- **Q17:** Correct: `[]`; Incorrect: `[R1, R2, R3, R4, R5]`
- **Q18:** Correct: `[]`; Incorrect: `[R1, R2, R3, R4, R5]`
- **Q19:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q20:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q21:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q22:** Correct: `[R1, R4]`; Incorrect: `[R2, R3, R5]`
- **Q23:** Correct: `[R2, R4, R5]`; Incorrect: `[R1, R3]`
- **Q24:** Correct: `[]`; Incorrect: `[R1, R2, R3, R4, R5]`
- **Q25:** Correct: `[]`; Incorrect: `[R1, R2, R3, R4, R5]`

</details>

<details>
<summary><strong>E03 — Fixed — BAAI/bge-small-en-v1.5</strong></summary>

- **Q1:** Correct: `[R2, R3]`; Incorrect: `[R1, R4, R5]`
- **Q2:** Correct: `[R1, R2, R3]`; Incorrect: `[R4, R5]`
- **Q3:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q4:** Correct: `[R1, R2, R4, R5]`; Incorrect: `[R3]`
- **Q5:** Correct: `[R1, R2, R3, R4]`; Incorrect: `[R5]`
- **Q6:** Correct: `[R1, R2, R3, R4, R5]`; Incorrect: `[]`
- **Q7:** Correct: `[R1, R2, R3, R4, R5]`; Incorrect: `[]`
- **Q8:** Correct: `[R1, R2, R3, R4]`; Incorrect: `[R5]`
- **Q9:** Correct: `[R1, R2, R3]`; Incorrect: `[R4, R5]`
- **Q10:** Correct: `[R1, R2, R5]`; Incorrect: `[R3, R4]`
- **Q11:** Correct: `[R1, R3]`; Incorrect: `[R2, R4, R5]`
- **Q12:** Correct: `[R1, R3, R5]`; Incorrect: `[R2, R4]`
- **Q13:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q14:** Correct: `[R1, R2, R4, R5]`; Incorrect: `[R3]`
- **Q15:** Correct: `[R1, R2, R3, R4, R5]`; Incorrect: `[]`
- **Q16:** Correct: `[R1, R2, R5]`; Incorrect: `[R3, R4]`
- **Q17:** Correct: `[R4]`; Incorrect: `[R1, R2, R3, R5]`
- **Q18:** Correct: `[]`; Incorrect: `[R1, R2, R3, R4, R5]`
- **Q19:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q20:** Correct: `[R1, R3, R4]`; Incorrect: `[R2, R5]`
- **Q21:** Correct: `[R1, R2, R3, R4, R5]`; Incorrect: `[]`
- **Q22:** Correct: `[R1, R2, R3, R4, R5]`; Incorrect: `[]`
- **Q23:** Correct: `[R1, R2, R5]`; Incorrect: `[R3, R4]`
- **Q24:** Correct: `[R1, R3]`; Incorrect: `[R2, R4, R5]`
- **Q25:** Correct: `[R2, R4]`; Incorrect: `[R1, R3, R5]`

</details>

<details>
<summary><strong>E04 — Fixed — BAAI/bge-small-en-v1.5</strong></summary>

- **Q1:** Correct: `[R5]`; Incorrect: `[R1, R2, R3, R4]`
- **Q2:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q3:** Correct: `[R3, R4]`; Incorrect: `[R1, R2, R5]`
- **Q4:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q5:** Correct: `[R1, R3, R5]`; Incorrect: `[R2, R4]`
- **Q6:** Correct: `[R1, R2, R3]`; Incorrect: `[R4, R5]`
- **Q7:** Correct: `[R1, R2, R3]`; Incorrect: `[R4, R5]`
- **Q8:** Correct: `[R1, R2, R3, R4, R5]`; Incorrect: `[]`
- **Q9:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q10:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q11:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q12:** Correct: `[R1, R4]`; Incorrect: `[R2, R3, R5]`
- **Q13:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q14:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q15:** Correct: `[R1, R2, R4, R5]`; Incorrect: `[R3]`
- **Q16:** Correct: `[R1, R5]`; Incorrect: `[R2, R3, R4]`
- **Q17:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q18:** Correct: `[]`; Incorrect: `[R1, R2, R3, R4, R5]`
- **Q19:** Correct: `[R1, R4]`; Incorrect: `[R2, R3, R5]`
- **Q20:** Correct: `[R1, R4]`; Incorrect: `[R2, R3, R5]`
- **Q21:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q22:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q23:** Correct: `[R1, R2, R5]`; Incorrect: `[R3, R4]`
- **Q24:** Correct: `[R1, R4]`; Incorrect: `[R2, R3, R5]`
- **Q25:** Correct: `[R2]`; Incorrect: `[R1, R3, R4, R5]`

</details>

<details>
<summary><strong>E05 — Semantic (p90) — BAAI/bge-small-en-v1.5</strong></summary>

> Semantic threshold taken from experiment context; log header does not record it.

- **Q1:** Correct: `[R4]`; Incorrect: `[R1, R2, R3, R5]`
- **Q2:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q3:** Correct: `[R1, R3]`; Incorrect: `[R2, R4, R5]`
- **Q4:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q5:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q6:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q7:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q8:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q9:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q10:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q11:** Correct: `[]`; Incorrect: `[R1, R2, R3, R4, R5]`
- **Q12:** Correct: `[R4, R5]`; Incorrect: `[R1, R2, R3]`
- **Q13:** Correct: `[R5]`; Incorrect: `[R1, R2, R3, R4]`
- **Q14:** Correct: `[R2]`; Incorrect: `[R1, R3, R4, R5]`
- **Q15:** Correct: `[R1, R4]`; Incorrect: `[R2, R3, R5]`
- **Q16:** Correct: `[R1, R3, R5]`; Incorrect: `[R2, R4]`
- **Q17:** Correct: `[R4]`; Incorrect: `[R1, R2, R3, R5]`
- **Q18:** Correct: `[]`; Incorrect: `[R1, R2, R3, R4, R5]`
- **Q19:** Correct: `[R2]`; Incorrect: `[R1, R3, R4, R5]`
- **Q20:** Correct: `[R2]`; Incorrect: `[R1, R3, R4, R5]`
- **Q21:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q22:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q23:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q24:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q25:** Correct: `[]`; Incorrect: `[R1, R2, R3, R4, R5]`

</details>

<details>
<summary><strong>E06 — Semantic → Recursive — BAAI/bge-small-en-v1.5</strong></summary>

> Semantic breakpoint threshold was not recorded in the log header.

- **Q1:** Correct: `[R4, R5]`; Incorrect: `[R1, R2, R3]`
- **Q2:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q3:** Correct: `[R1, R3]`; Incorrect: `[R2, R4, R5]`
- **Q4:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q5:** Correct: `[R1, R3, R4, R5]`; Incorrect: `[R2]`
- **Q6:** Correct: `[R1, R2, R3]`; Incorrect: `[R4, R5]`
- **Q7:** Correct: `[R1, R2, R3]`; Incorrect: `[R4, R5]`
- **Q8:** Correct: `[R1, R2, R3, R4, R5]`; Incorrect: `[]`
- **Q9:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q10:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q11:** Correct: `[R2]`; Incorrect: `[R1, R3, R4, R5]`
- **Q12:** Correct: `[R2]`; Incorrect: `[R1, R3, R4, R5]`
- **Q13:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q14:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q15:** Correct: `[R1, R2, R3, R5]`; Incorrect: `[R4]`
- **Q16:** Correct: `[R1, R2, R5]`; Incorrect: `[R3, R4]`
- **Q17:** Correct: `[R4]`; Incorrect: `[R1, R2, R3, R5]`
- **Q18:** Correct: `[]`; Incorrect: `[R1, R2, R3, R4, R5]`
- **Q19:** Correct: `[R1, R5]`; Incorrect: `[R2, R3, R4]`
- **Q20:** Correct: `[R1, R3]`; Incorrect: `[R2, R4, R5]`
- **Q21:** Correct: `[R1, R2, R3]`; Incorrect: `[R4, R5]`
- **Q22:** Correct: `[R1, R2, R3]`; Incorrect: `[R4, R5]`
- **Q23:** Correct: `[R1, R4]`; Incorrect: `[R2, R3, R5]`
- **Q24:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q25:** Correct: `[]`; Incorrect: `[R1, R2, R3, R4, R5]`

</details>

<details>
<summary><strong>E07 — Recursive — BAAI/bge-small-en-v1.5</strong></summary>

> Embedding baseline for model comparison.

- **Q1:** Correct: `[R4, R5]`; Incorrect: `[R1, R2, R3]`
- **Q2:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q3:** Correct: `[R3, R4]`; Incorrect: `[R1, R2, R5]`
- **Q4:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q5:** Correct: `[R1, R2, R3]`; Incorrect: `[R4, R5]`
- **Q6:** Correct: `[R1, R2, R3]`; Incorrect: `[R4, R5]`
- **Q7:** Correct: `[R1, R2, R3]`; Incorrect: `[R4, R5]`
- **Q8:** Correct: `[R1, R2, R3, R4, R5]`; Incorrect: `[]`
- **Q9:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q10:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q11:** Correct: `[R2]`; Incorrect: `[R1, R3, R4, R5]`
- **Q12:** Correct: `[R1, R3]`; Incorrect: `[R2, R4, R5]`
- **Q13:** Correct: `[R1, R5]`; Incorrect: `[R2, R3, R4]`
- **Q14:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q15:** Correct: `[R1, R2, R3, R4, R5]`; Incorrect: `[]`
- **Q16:** Correct: `[R1, R2, R4]`; Incorrect: `[R3, R5]`
- **Q17:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q18:** Correct: `[]`; Incorrect: `[R1, R2, R3, R4, R5]`
- **Q19:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q20:** Correct: `[R1, R5]`; Incorrect: `[R2, R3, R4]`
- **Q21:** Correct: `[R1, R4]`; Incorrect: `[R2, R3, R5]`
- **Q22:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q23:** Correct: `[R1, R2, R4]`; Incorrect: `[R3, R5]`
- **Q24:** Correct: `[R1, R5]`; Incorrect: `[R2, R3, R4]`
- **Q25:** Correct: `[R2]`; Incorrect: `[R1, R3, R4, R5]`

</details>

<details>
<summary><strong>E08 — Recursive — BAAI/bge-large-en-v1.5</strong></summary>

- **Q1:** Correct: `[R5]`; Incorrect: `[R1, R2, R3, R4]`
- **Q2:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q3:** Correct: `[R1, R2, R3, R4]`; Incorrect: `[R5]`
- **Q4:** Correct: `[R1, R3]`; Incorrect: `[R2, R4, R5]`
- **Q5:** Correct: `[R1, R2, R5]`; Incorrect: `[R3, R4]`
- **Q6:** Correct: `[R1, R2, R3]`; Incorrect: `[R4, R5]`
- **Q7:** Correct: `[R1, R2, R3]`; Incorrect: `[R4, R5]`
- **Q8:** Correct: `[R1, R2, R3, R5]`; Incorrect: `[R4]`
- **Q9:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q10:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q11:** Correct: `[]`; Incorrect: `[R1, R2, R3, R4, R5]`
- **Q12:** Correct: `[R2, R3]`; Incorrect: `[R1, R4, R5]`
- **Q13:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q14:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q15:** Correct: `[R1, R2, R3]`; Incorrect: `[R4, R5]`
- **Q16:** Correct: `[R3, R4, R5]`; Incorrect: `[R1, R2]`
- **Q17:** Correct: `[R2]`; Incorrect: `[R1, R3, R4, R5]`
- **Q18:** Correct: `[]`; Incorrect: `[R1, R2, R3, R4, R5]`
- **Q19:** Correct: `[R2, R5]`; Incorrect: `[R1, R3, R4]`
- **Q20:** Correct: `[R2]`; Incorrect: `[R1, R3, R4, R5]`
- **Q21:** Correct: `[R1, R2, R4]`; Incorrect: `[R3, R5]`
- **Q22:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q23:** Correct: `[R2, R3]`; Incorrect: `[R1, R4, R5]`
- **Q24:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q25:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`

</details>

<details>
<summary><strong>E09 — Recursive — BAAI/bge-m3</strong></summary>

- **Q1:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q2:** Correct: `[R1, R2, R3]`; Incorrect: `[R4, R5]`
- **Q3:** Correct: `[R1, R2, R5]`; Incorrect: `[R3, R4]`
- **Q4:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q5:** Correct: `[R1, R2, R3, R4]`; Incorrect: `[R5]`
- **Q6:** Correct: `[R1, R2, R3]`; Incorrect: `[R4, R5]`
- **Q7:** Correct: `[R1, R2, R3]`; Incorrect: `[R4, R5]`
- **Q8:** Correct: `[R1, R2, R3]`; Incorrect: `[R4, R5]`
- **Q9:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q10:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q11:** Correct: `[R1, R5]`; Incorrect: `[R2, R3, R4]`
- **Q12:** Correct: `[R1, R4]`; Incorrect: `[R2, R3, R5]`
- **Q13:** Correct: `[]`; Incorrect: `[R1, R2, R3, R4, R5]`
- **Q14:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q15:** Correct: `[R1, R2, R3, R4]`; Incorrect: `[R5]`
- **Q16:** Correct: `[R1, R2, R3]`; Incorrect: `[R4, R5]`
- **Q17:** Correct: `[R2]`; Incorrect: `[R1, R3, R4, R5]`
- **Q18:** Correct: `[]`; Incorrect: `[R1, R2, R3, R4, R5]`
- **Q19:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q20:** Correct: `[R1, R3]`; Incorrect: `[R2, R4, R5]`
- **Q21:** Correct: `[R1, R5]`; Incorrect: `[R2, R3, R4]`
- **Q22:** Correct: `[R1, R3]`; Incorrect: `[R2, R4, R5]`
- **Q23:** Correct: `[R1, R2, R4, R5]`; Incorrect: `[R3]`
- **Q24:** Correct: `[R1, R3]`; Incorrect: `[R2, R4, R5]`
- **Q25:** Correct: `[R3]`; Incorrect: `[R1, R2, R4, R5]`

</details>

<details>
<summary><strong>E10 — Recursive — Qwen/Qwen3-Embedding-0.6B</strong></summary>

- **Q1:** Correct: `[R2]`; Incorrect: `[R1, R3, R4, R5]`
- **Q2:** Correct: `[R1, R4, R5]`; Incorrect: `[R2, R3]`
- **Q3:** Correct: `[R1, R2, R4]`; Incorrect: `[R3, R5]`
- **Q4:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q5:** Correct: `[R1, R2, R3]`; Incorrect: `[R4, R5]`
- **Q6:** Correct: `[R1, R2, R3]`; Incorrect: `[R4, R5]`
- **Q7:** Correct: `[R1, R2, R3]`; Incorrect: `[R4, R5]`
- **Q8:** Correct: `[R1, R2, R3, R4]`; Incorrect: `[R5]`
- **Q9:** Correct: `[R1, R5]`; Incorrect: `[R2, R3, R4]`
- **Q10:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q11:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q12:** Correct: `[R1, R4]`; Incorrect: `[R2, R3, R5]`
- **Q13:** Correct: `[]`; Incorrect: `[R1, R2, R3, R4, R5]`
- **Q14:** Correct: `[R1, R2, R5]`; Incorrect: `[R3, R4]`
- **Q15:** Correct: `[R1, R2, R3, R4, R5]`; Incorrect: `[]`
- **Q16:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q17:** Correct: `[R3, R5]`; Incorrect: `[R1, R2, R4]`
- **Q18:** Correct: `[]`; Incorrect: `[R1, R2, R3, R4, R5]`
- **Q19:** Correct: `[R1, R4]`; Incorrect: `[R2, R3, R5]`
- **Q20:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q21:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`
- **Q22:** Correct: `[R1, R2]`; Incorrect: `[R3, R4, R5]`
- **Q23:** Correct: `[R2]`; Incorrect: `[R1, R3, R4, R5]`
- **Q24:** Correct: `[R1, R2, R5]`; Incorrect: `[R3, R4]`
- **Q25:** Correct: `[R1]`; Incorrect: `[R2, R3, R4, R5]`

</details>
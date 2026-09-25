# 🔍 Section 9: Hybrid Search Strategies

## Verified notebook sequence

1. [Sparse, dense, and hybrid retrieval](1-densesparse.ipynb) — BM25, lossy
   dense LSA, branch evaluation, and reciprocal-rank fusion.
2. [Reranking](2-reranking.ipynb) — candidate recall, second-stage precision,
   candidate-depth experiments, and MRR.
3. [Maximal marginal relevance](3-mmr.ipynb) — relevance/diversity trade-offs,
   lambda sweeps, and redundancy checks.

These notebooks use offline, deterministic examples and are saved with verified
outputs. The older notes below remain as supplementary reference and may mention
optional packages or hosted models not required by the canonical notebooks.

---

## ⚡ TL;DR (30-Second Summary)

```
┌────────────────────────────────────────────────────────────────────────────┐
│  SPARSE (BM25)     →  Keyword matching      →  "Find exact words"          │
│  DENSE (FAISS)     →  Semantic embeddings   →  "Understand meaning"        │
│  HYBRID            →  Sparse + Dense        →  "Best of both"              │
│  RERANKING         →  Re-score top results  →  "Precision boost"           │
│  MMR               →  Diversity filter      →  "No redundant docs"         │
└────────────────────────────────────────────────────────────────────────────┘
```

**One-liner:** Combine retrievers (Hybrid) → Remove duplicates (MMR) → Rerank for precision (FlashRank)

---

## 🗺️ How It All Connects

```
                              USER QUERY
                                  │
                                  ▼
              ┌───────────────────┴───────────────────┐
              │                                       │
              ▼                                       ▼
        ┌──────────┐                           ┌──────────┐
        │  SPARSE  │                           │  DENSE   │
        │  (BM25)  │                           │ (FAISS)  │
        │ Keywords │                           │ Semantic │
        └────┬─────┘                           └────┬─────┘
              │                                       │
              └───────────────┬───────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ HYBRID/ENSEMBLE │  ← Combine with weights [0.7, 0.3]
                    │    Retriever    │
                    └────────┬────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │      MMR        │  ← Optional: Remove redundant docs
                    │   (Diversity)   │
                    └────────┬────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   RERANKING     │  ← Optional: Precision boost
                    │  (FlashRank)    │
                    └────────┬────────┘
                              │
                              ▼
                        TOP-K DOCS → LLM → ANSWER
```

---

## 📚 Table of Contents

1. [Dense vs Sparse Retrieval](#1-dense-vs-sparse-retrieval)
2. [Hybrid Retriever (Ensemble)](#2-hybrid-retriever-ensemble)
3. [Reranking Techniques](#3-reranking-techniques)
4. [MMR - Maximal Marginal Relevance](#4-mmr---maximal-marginal-relevance)
5. [Quick Reference Cheatsheet](#5-quick-reference-cheatsheet)
6. [Self-Test Questions](#6-self-test-questions)

---

## 1. Dense vs Sparse Retrieval

### 🎯 Core Concept

| Aspect         | **Sparse (BM25)**                    | **Dense (Embeddings)**                |
| -------------- | ------------------------------------ | ------------------------------------- |
| **How**        | Keyword/TF-IDF scoring               | Vector similarity                     |
| **Strengths**  | ✅ Exact matches, Fast, No ML needed | ✅ Semantic meaning, Handles synonyms |
| **Weaknesses** | ❌ Misses synonyms                   | ❌ May miss exact terms               |
| **Best For**   | Code, Technical docs                 | Conversational, Natural language      |

### 💡 Key Insight

> **Neither is perfect alone!** Sparse = exact terms, Dense = meaning. **Combine both.**

### 🧠 Memory Trick

> **"Dense = Deep meaning, Sparse = Surface keywords"**

---

## 2. Hybrid Retriever (Ensemble)

### 🎯 What Is It?

Combines Dense + Sparse retrievers with weighted scores.

### 📝 Code Pattern

```python
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings

# Dense
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
dense_retriever = FAISS.from_documents(docs, embedding_model).as_retriever(search_kwargs={"k": 3})

# Sparse
sparse_retriever = BM25Retriever.from_documents(docs)
sparse_retriever.k = 3

# Hybrid
hybrid = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.7, 0.3]  # Tune these!
)
```

### ⚖️ Weight Guidelines

| Use Case           | Dense | Sparse |
| ------------------ | ----- | ------ |
| Conversational/NLP | 0.7   | 0.3    |
| Balanced           | 0.5   | 0.5    |
| Code/Exact terms   | 0.3   | 0.7    |

---

## 3. Reranking Techniques

### 🎯 What Is It?

**Two-stage process:** Fast retrieval → Accurate re-scoring

```
Query → [Fast Retriever] → Top-K → [Reranker] → Best Docs
```

### 🔄 Why Rerank?

- Fast retrievers sacrifice accuracy for speed
- Vector similarity ≠ actual relevance
- Cross-encoders understand query-doc pairs better

---

### 📌 Method 1: LLM-Based Reranking

```python
prompt = PromptTemplate.from_template("""
Rank these documents by relevance to: "{question}"

Documents:
{documents}

Output: comma-separated indices (e.g., 2,1,3,0)
""")

chain = prompt | llm | StrOutputParser()
response = chain.invoke({"question": query, "documents": formatted_docs})

# Parse and reorder
indices = [int(x.strip()) - 1 for x in response.split(",") if x.strip().isdigit()]
reranked = [docs[i] for i in indices if 0 <= i < len(docs)]
```

**Pros:** Flexible | **Cons:** Slow, expensive

---

### 📌 Method 2: FlashRank ⭐ (Recommended)

Fast cross-encoder reranking without LLM costs.

```python
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever

compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

docs = reranking_retriever.invoke(query)
```

**Pros:** Fast, accurate, free | **Cons:** Model download needed

---

### 📋 Compressors Comparison

| Compressor          | Speed   | Accuracy | Cost |
| ------------------- | ------- | -------- | ---- |
| `FlashrankRerank`   | ⚡ Fast | ✅ High  | Free |
| `CohereRerank`      | Medium  | ✅ High  | API  |
| `LLMChainExtractor` | 🐢 Slow | High     | LLM  |
| `EmbeddingsFilter`  | ⚡ Fast | Medium   | Free |

### 🧠 Memory Trick

> **"Retrieve broad, Rerank narrow"**

---

## 4. MMR - Maximal Marginal Relevance

### 🎯 What Is It?

Balances **relevance** + **diversity** to avoid redundant results.

### 🎨 The Problem

```
Without MMR:                          With MMR:
1. Python is a language...            1. Python is a language...
2. Python is a programming lang...    2. Python has ML libraries...
3. Python language was created...     3. Python uses indentation...
   ↑ REDUNDANT!                          ↑ DIVERSE!
```

### 📝 Implementation

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",           # 🔑 Magic switch
    search_kwargs={
        "k": 5,                  # Final docs to return
        "fetch_k": 20,           # Candidates to consider
        "lambda_mult": 0.5       # 0=diversity, 1=relevance
    }
)
```

### 🎚️ Lambda Guide

```
lambda_mult = 1.0  →  Pure relevance (no diversity)
lambda_mult = 0.5  →  Balanced ✅ (default)
lambda_mult = 0.0  →  Max diversity (may hurt relevance)
```

### 🧠 Memory Trick

> **"MMR = Maximum info, Minimum repetition"**

---

## 5. Quick Reference Cheatsheet

### 🚀 Decision Tree

```
Need exact keywords?
  └─ YES → BM25 (Sparse)
  └─ NO  → Need semantic?
            └─ YES → Dense (FAISS)
            └─ BOTH → Hybrid (EnsembleRetriever)

Results redundant?
  └─ YES → Add MMR (search_type="mmr")

Need precision?
  └─ YES → Add Reranking (FlashRank)
```

### 📦 All Imports

```python
# Dense
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Sparse
from langchain_community.retrievers import BM25Retriever

# Hybrid
from langchain_classic.retrievers import EnsembleRetriever

# Reranking
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
```

### 🔑 Copy-Paste Snippets

**Hybrid:**

```python
EnsembleRetriever(retrievers=[dense, sparse], weights=[0.7, 0.3])
```

**MMR:**

```python
vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
```

**Reranking:**

```python
ContextualCompressionRetriever(base_compressor=FlashrankRerank(), base_retriever=retriever)
```

---

## 🎓 Key Takeaways

| Technique  | Purpose             | Key Code                                |
| ---------- | ------------------- | --------------------------------------- |
| **Sparse** | Keyword matching    | `BM25Retriever.from_documents()`        |
| **Dense**  | Semantic similarity | `FAISS.from_documents()`                |
| **Hybrid** | Combine both        | `EnsembleRetriever(weights=[0.7, 0.3])` |
| **MMR**    | Reduce redundancy   | `search_type="mmr"`                     |
| **Rerank** | Precision boost     | `FlashrankRerank()`                     |

---

## 6. Self-Test Questions

### 📝 Quick Recall (Answer in your head!)

<details>
<summary><b>1. What's the difference between Dense and Sparse retrieval?</b></summary>

**Sparse (BM25):** Keyword/term matching using TF-IDF style scoring. Great for exact matches.

**Dense (Embeddings):** Semantic similarity in vector space. Understands meaning & synonyms.

</details>

<details>
<summary><b>2. Why combine Dense + Sparse in a Hybrid retriever?</b></summary>

Neither is perfect alone! Sparse catches exact keywords, Dense handles paraphrases and semantic meaning. Together they improve both precision and recall.

</details>

<details>
<summary><b>3. What does `weights=[0.7, 0.3]` mean in EnsembleRetriever?</b></summary>

Dense retriever gets 70% weight, Sparse gets 30%. Higher weight = more influence on final ranking. Use higher dense weight for conversational queries, higher sparse for technical/code search.

</details>

<details>
<summary><b>4. What is reranking and why use it?</b></summary>

Two-stage process: Fast retriever gets top-k docs → Reranker re-scores them for precision. Vector similarity isn't always actual relevance. Cross-encoders (like FlashRank) understand query-document pairs better.

</details>

<details>
<summary><b>5. FlashRank vs LLM reranking - which is better?</b></summary>

**FlashRank:** Faster, cheaper (no API costs), uses cross-encoder models.

**LLM:** More flexible reasoning, but slower and expensive.

**Recommendation:** FlashRank for most cases.

</details>

<details>
<summary><b>6. What problem does MMR solve?</b></summary>

Reduces **redundancy** in retrieved documents. Without MMR, you might get 5 docs saying the same thing. MMR balances relevance with diversity to get complementary information.

</details>

<details>
<summary><b>7. What does `lambda_mult` control in MMR?</b></summary>

Balance between relevance (1.0) and diversity (0.0).

- `1.0` = Pure relevance, no diversity
- `0.5` = Balanced (recommended)
- `0.0` = Maximum diversity

</details>

<details>
<summary><b>8. What's the typical RAG pipeline order with these techniques?</b></summary>

```
Query → Hybrid Retriever → MMR (optional) → Reranking (optional) → Top-K Docs → LLM
```

</details>

---

### 🎯 Code Challenge

**Try writing this from memory:**

> Create a hybrid retriever that combines FAISS and BM25, then add MMR for diversity.

<details>
<summary><b>Solution</b></summary>

```python
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings

# Dense
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
dense = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})  # MMR here!

# Sparse
sparse = BM25Retriever.from_documents(docs)
sparse.k = 3

# Hybrid
hybrid = EnsembleRetriever(
    retrievers=[dense, sparse],
    weights=[0.7, 0.3]
)
```

</details>

---

## 📁 Files in This Section

| File                        | Description                    |
| --------------------------- | ------------------------------ |
| `1-densesparse.ipynb`       | Hybrid retriever: BM25 + FAISS |
| `2-reranking.ipynb`         | LLM and FlashRank reranking    |
| `3-mmr.ipynb`               | Maximal Marginal Relevance     |
| `langchain_sample.txt`      | Sample data for reranking      |
| `langchain_rag_dataset.txt` | Sample data for MMR            |

---

> 💡 **Golden Rule:** Start simple → Add hybrid if missing keywords → Add MMR if redundant → Add reranking if need precision!

# 📄 Section 8: Advanced Chunking & Preprocessing

---

## ⚡ TL;DR (30-Second Summary)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TRADITIONAL CHUNKING  →  Fixed size splits     →  "Cuts mid-thought" 😕    │
│  SEMANTIC CHUNKING     →  Meaning-based splits  →  "Preserves context" ✅   │
└─────────────────────────────────────────────────────────────────────────────┘

How it works:
  1. Split text into sentences
  2. Embed each sentence
  3. Group similar sentences (cosine similarity ≥ threshold)
  4. Each group = one chunk
```

**One-liner:** Split by **meaning**, not by character count.

---

## 🗺️ How It Works (Visual)

```
                         INPUT TEXT
                              │
                              ▼
                    ┌─────────────────┐
                    │  Split into     │
                    │   Sentences     │
                    └────────┬────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
     ┌─────────┐        ┌─────────┐        ┌─────────┐
     │ Sent 1  │        │ Sent 2  │        │ Sent 3  │
     └────┬────┘        └────┬────┘        └────┬────┘
          │                   │                   │
          ▼                   ▼                   ▼
     ┌─────────┐        ┌─────────┐        ┌─────────┐
     │Embed [..]│       │Embed [..]│       │Embed [..]│
     └────┬────┘        └────┬────┘        └────┬────┘
          │                   │                   │
          └─────────┬─────────┴─────────┬─────────┘
                    │                   │
                    ▼                   ▼
            ┌──────────────┐    ┌──────────────┐
            │ Similarity   │    │ Similarity   │
            │   ≥ 0.7?     │    │   ≥ 0.7?     │
            └──────┬───────┘    └──────┬───────┘
                   │                   │
            ┌──────┴──────┐     ┌──────┴──────┐
            │ YES: Same   │     │ NO: New     │
            │   Chunk     │     │   Chunk     │
            └─────────────┘     └─────────────┘
```

---

## 📚 Table of Contents

1. [Traditional vs Semantic Chunking](#1-traditional-vs-semantic-chunking)
2. [How Semantic Chunking Works](#2-how-semantic-chunking-works)
3. [Custom Semantic Chunker](#3-custom-semantic-chunker-from-scratch)
4. [LangChain SemanticChunker](#4-langchain-semanticchunker)
5. [RAG Pipeline Integration](#5-rag-pipeline-integration)
6. [Quick Reference Cheatsheet](#6-quick-reference-cheatsheet)
7. [Self-Test Questions](#7-self-test-questions)

---

## 1. Traditional vs Semantic Chunking

### 🎯 The Problem with Traditional Chunking

```
Traditional (Character-based):
┌─────────────────────────────────────────┐
│ "LangChain is a framework for building  │ ← Chunk 1
│ applications with LLMs. Langchain prov" │ ← CUT MID-WORD! 😱
├─────────────────────────────────────────┤
│ "ides modular abstractions to combine"  │ ← Chunk 2
└─────────────────────────────────────────┘
```

```
Semantic (Meaning-based):
┌─────────────────────────────────────────┐
│ "LangChain is a framework for building  │
│ applications with LLMs. Langchain       │ ← Chunk 1 (complete thought)
│ provides modular abstractions..."       │
├─────────────────────────────────────────┤
│ "The Eiffel Tower is located in Paris.  │ ← Chunk 2 (different topic)
│ France is a popular tourist dest..."    │
└─────────────────────────────────────────┘
```

### 📊 Comparison Table

| Aspect           | Traditional Chunking        | Semantic Chunking              |
| ---------------- | --------------------------- | ------------------------------ |
| **Split Method** | Fixed character/token count | Embedding similarity           |
| **Context**      | ❌ May cut mid-sentence     | ✅ Preserves complete thoughts |
| **Speed**        | ⚡ Very fast                | 🐢 Slower (needs embeddings)   |
| **Quality**      | Lower retrieval quality     | Higher retrieval quality       |
| **Use Case**     | Simple docs, speed-critical | Complex docs, quality-critical |

### 💡 Key Insight

> **Traditional chunking is blind to meaning.** It's like cutting a book into 500-word pieces without checking if you're mid-sentence. Semantic chunking **understands** where thoughts end.

---

## 2. How Semantic Chunking Works

### 🔧 The Algorithm

```
1. Split text into sentences
2. Embed each sentence using a model (e.g., all-MiniLM-L6-v2)
3. Compare adjacent sentence embeddings (cosine similarity)
4. If similarity ≥ threshold → Same chunk
5. If similarity < threshold → Start new chunk
```

### 📝 Core Code Pattern

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize
model = SentenceTransformer('all-MiniLM-L6-v2')
threshold = 0.7  # Tune this!

# Split into sentences
sentences = [s.strip() for s in text.split('.') if s.strip()]

# Embed all sentences
embeddings = model.encode(sentences)

# Group by similarity
chunks = []
current_chunk = [sentences[0]]

for i in range(1, len(sentences)):
    sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]

    if sim >= threshold:
        current_chunk.append(sentences[i])  # Same topic
    else:
        chunks.append(". ".join(current_chunk) + ".")  # New topic
        current_chunk = [sentences[i]]

chunks.append(". ".join(current_chunk) + ".")  # Don't forget last chunk!
```

### 🎚️ Threshold Guide

```
threshold = 0.9  →  Very strict (smaller, tighter chunks)
threshold = 0.7  →  Balanced ✅ (recommended)
threshold = 0.5  →  Loose (larger chunks, more content together)
```

### 🧠 Memory Trick

> **"High threshold = High standards = Smaller chunks"**

---

## 3. Custom Semantic Chunker (From Scratch)

### 📝 Reusable Class

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document

class ThresholdSemanticChunker:
    def __init__(self, model_name="all-MiniLM-L6-v2", threshold=0.7):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def split(self, text: str) -> list[str]:
        """Split raw text into semantic chunks."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        embeddings = self.model.encode(sentences)

        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
            if sim >= self.threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append(". ".join(current_chunk) + ".")
                current_chunk = [sentences[i]]

        chunks.append(". ".join(current_chunk) + ".")
        return chunks

    def split_documents(self, docs: list[Document]) -> list[Document]:
        """Split LangChain Documents into semantic chunks."""
        result = []
        for doc in docs:
            for chunk in self.split(doc.page_content):
                result.append(Document(page_content=chunk, metadata=doc.metadata))
        return result
```

### 🚀 Usage

```python
# Initialize
chunker = ThresholdSemanticChunker(threshold=0.7)

# From raw text
chunks = chunker.split("Your long text here...")

# From LangChain Documents
doc = Document(page_content="Your text here...")
chunk_docs = chunker.split_documents([doc])
```

---

## 4. LangChain SemanticChunker

### 🎯 Built-in Solution

LangChain provides `SemanticChunker` in the experimental module.

### 📝 Implementation

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader

# Load documents
loader = TextLoader("your_file.txt")
docs = loader.load()

# Initialize embedding model
embedding = OpenAIEmbeddings()

# Create semantic chunker
chunker = SemanticChunker(embedding)

# Split documents
chunks = chunker.split_documents(docs)

# View results
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:\n{chunk.page_content}")
```

### ⚖️ Custom vs LangChain

| Aspect           | Custom Chunker              | LangChain SemanticChunker |
| ---------------- | --------------------------- | ------------------------- |
| **Control**      | Full control over algorithm | Less customizable         |
| **Dependencies** | sentence-transformers       | langchain-experimental    |
| **Embedding**    | Any SentenceTransformer     | Any LangChain Embeddings  |
| **Use When**     | Need custom logic/threshold | Quick prototyping         |

---

## 5. RAG Pipeline Integration

### 🔧 Complete Pipeline

```python
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1️⃣ Load & Chunk (Semantic)
chunker = ThresholdSemanticChunker(threshold=0.7)
chunks = chunker.split_documents(docs)

# 2️⃣ Create Vector Store
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3️⃣ Prompt Template
template = """Answer based on the context:

Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 4️⃣ LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# 5️⃣ RAG Chain (LCEL)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6️⃣ Query
result = rag_chain.invoke("What is LangChain used for?")
```

### 🗺️ Pipeline Flow

```
Document → Semantic Chunker → Vector Store → Retriever → RAG Chain → Answer
              (split by         (FAISS/       (top-k)    (prompt +
               meaning)          Chroma)                   LLM)
```

---

## 6. Quick Reference Cheatsheet

### 📦 All Imports

```python
# Custom Semantic Chunking
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# LangChain Semantic Chunking
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# Document Handling
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader

# Vector Stores
from langchain_community.vectorstores import FAISS, Chroma
```

### 🔑 Copy-Paste Snippets

**Custom Chunker (Quick):**

```python
chunker = ThresholdSemanticChunker(threshold=0.7)
chunks = chunker.split_documents(docs)
```

**LangChain Chunker (Quick):**

```python
chunker = SemanticChunker(OpenAIEmbeddings())
chunks = chunker.split_documents(docs)
```

**Cosine Similarity:**

```python
sim = cosine_similarity([embedding1], [embedding2])[0][0]
```

### 🚀 Decision Tree

```
Need fast chunking?
  └─ YES → RecursiveCharacterTextSplitter (traditional)
  └─ NO  → Need semantic coherence?
            └─ YES → SemanticChunker
            └─ CUSTOM → ThresholdSemanticChunker
```

---

## 🎓 Key Takeaways

| Concept         | Purpose                   | Key Code                        |
| --------------- | ------------------------- | ------------------------------- |
| **Traditional** | Fast, simple splits       | `CharacterTextSplitter()`       |
| **Semantic**    | Meaning-preserving splits | `SemanticChunker(embedding)`    |
| **Threshold**   | Control chunk granularity | `threshold=0.7` (tune it!)      |
| **Similarity**  | Compare sentence meanings | `cosine_similarity(emb1, emb2)` |

---

## 7. Self-Test Questions

### 📝 Quick Recall (Click to reveal!)

<details>
<summary><b>1. What's the main problem with traditional character-based chunking?</b></summary>

It **cuts text blindly** without considering meaning. You might split mid-sentence, mid-word, or separate related information into different chunks. This hurts retrieval quality because context is lost.

</details>

<details>
<summary><b>2. How does semantic chunking decide where to split?</b></summary>

1. Split text into sentences
2. Create embeddings for each sentence
3. Calculate cosine similarity between adjacent sentences
4. If similarity ≥ threshold → keep in same chunk
5. If similarity < threshold → start new chunk

</details>

<details>
<summary><b>3. What does the threshold parameter control?</b></summary>

How similar sentences must be to stay in the same chunk:

- **Higher threshold (0.9)** = Stricter, smaller chunks, only very similar sentences grouped
- **Lower threshold (0.5)** = Looser, larger chunks, more content together
- **Recommended: 0.7** = Balanced

</details>

<details>
<summary><b>4. What embedding model is commonly used for semantic chunking?</b></summary>

`all-MiniLM-L6-v2` from SentenceTransformers - it's fast, lightweight, and produces good quality embeddings for similarity comparisons.

</details>

<details>
<summary><b>5. What's cosine similarity and why use it?</b></summary>

Cosine similarity measures the angle between two vectors (embeddings). Range: -1 to 1.

- **1.0** = Identical direction (same meaning)
- **0.0** = Perpendicular (unrelated)
- **-1.0** = Opposite

We use it because it's **scale-invariant** - it measures direction, not magnitude, which is better for comparing text meanings.

</details>

<details>
<summary><b>6. When should you use traditional chunking vs semantic chunking?</b></summary>

**Traditional:**

- Speed is critical
- Simple documents
- Prototyping quickly

**Semantic:**

- Quality matters
- Complex documents with multiple topics
- When retrieval accuracy is important

</details>

<details>
<summary><b>7. What's the difference between custom ThresholdSemanticChunker and LangChain's SemanticChunker?</b></summary>

**Custom ThresholdSemanticChunker:**

- Full control over threshold and algorithm
- Uses SentenceTransformers directly
- More customizable

**LangChain SemanticChunker:**

- Easy to use, less code
- Integrates with any LangChain embedding model
- Less customizable but good for quick prototyping

</details>

---

### 🎯 Code Challenge

**Try writing this from memory:**

> Create a semantic chunker that splits text based on embedding similarity, then use it in a simple RAG pipeline.

<details>
<summary><b>Solution</b></summary>

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Custom Semantic Chunker
class SemanticChunker:
    def __init__(self, threshold=0.7):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = threshold

    def split(self, text):
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        embeddings = self.model.encode(sentences)
        chunks, current = [], [sentences[0]]

        for i in range(1, len(sentences)):
            sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
            if sim >= self.threshold:
                current.append(sentences[i])
            else:
                chunks.append(". ".join(current) + ".")
                current = [sentences[i]]
        chunks.append(". ".join(current) + ".")
        return [Document(page_content=c) for c in chunks]

# RAG Pipeline
chunker = SemanticChunker(threshold=0.7)
chunks = chunker.split("Your long document text here...")

vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_template("Context: {context}\n\nQuestion: {question}")
llm = ChatOpenAI(model="gpt-4o-mini")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

answer = chain.invoke("Your question here")
```

</details>

---

## 📁 Files in This Section

| File                     | Description                       |
| ------------------------ | --------------------------------- |
| `semanti_chunking.ipynb` | Semantic chunking implementations |
| `langchain_intro.txt`    | Sample text for testing chunkers  |

---

> 💡 **Golden Rule:** Use **semantic chunking** when retrieval quality matters. The extra computation cost pays off in better RAG answers!

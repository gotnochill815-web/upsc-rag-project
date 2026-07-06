#  UPSC GS-II Curriculum Intelligence Engine

An end-to-end Retrieval-Augmented Generation (RAG) system for answering UPSC GS-II Mains questions using hybrid retrieval, neural reranking, and Large Language Models.

The system retrieves relevant constitutional provisions, previous year questions (PYQs), and curated notes before generating a structured UPSC-style answer.

---

##  Demo

**Live Demo**

 https://huggingface.co/spaces/prakhya15/upsc-gs2-rag

---

##  Features

- Hybrid Retrieval
  - BM25 lexical search
  - Dense semantic retrieval using BGE embeddings
-  CrossEncoder reranking for improved document relevance
-  Answer generation using Qwen 2.5 Instruct
-  Knowledge base containing
  - Constitution Articles
  - UPSC GS-II PYQs
  - Curated GS-II Notes
- Interactive Streamlit interface
-  Deployed on Hugging Face Spaces

---

# Architecture

```
                    User Question
                           │
                           ▼
                 Hybrid Retrieval Pipeline
          ┌─────────────────────────────────┐
          │                                 │
      BM25 Search                 Dense Retrieval
          │                                 │
          └──────────────┬──────────────────┘
                         ▼
                 Hybrid Score Fusion
                         ▼
               CrossEncoder Reranker
                         ▼
               Top Relevant Documents
                         ▼
                 Qwen 2.5 Instruct
                         ▼
           Structured UPSC GS-II Answer
```

---

# Tech Stack

| Component | Technology |
|------------|------------|
| Frontend | Streamlit |
| Retrieval | BM25 |
| Semantic Search | Sentence Transformers (BGE) |
| Vector Search | FAISS |
| Reranker | CrossEncoder MiniLM |
| LLM | Qwen 2.5 |
| Language | Python |
| Deployment | Hugging Face Spaces |

---

# Project Structure

```
.
├── app/
│   └── app.py
├── retrieval/
│   ├── bm25.py
│   ├── dense.py
│   ├── hybrid.py
│   └── reranker.py
├── generator/
│   └── answer_generator.py
├── data/
│   └── processed/
├── knowledge_base/
├── requirements.txt
└── README.md
```

---

# Retrieval Pipeline

### 1. BM25 Retrieval

Performs lexical keyword search over the UPSC knowledge base.

### 2. Dense Retrieval

Uses BGE embeddings with FAISS to retrieve semantically similar documents.

### 3. Hybrid Retrieval

Combines BM25 and Dense Retrieval results.

### 4. CrossEncoder Reranking

Ranks retrieved documents based on semantic relevance to the query.

### 5. Answer Generation

Qwen 2.5 generates a structured UPSC Mains answer grounded in the retrieved context.

---

# Knowledge Sources

- Constitution of India
- UPSC GS-II Previous Year Questions
- Curated GS-II Notes
- Constitutional Articles
- Governance Topics

---

# Example Query

```
Evaluate the role of the Election Commission of India in ensuring free and fair elections. Discuss the constitutional provisions, challenges, and reforms required.
```

---

# Example Output

- Retrieves relevant Constitution Articles
- Retrieves related PYQs
- Retrieves GS-II Notes
- Generates a structured answer including

- Introduction
- Constitutional Provisions
- Analysis
- Way Forward
- Conclusion

---

# Installation

Clone the repository

```bash
git clone https://github.com/gotnochill815-web/upsc-rag-project.git
cd upsc-rag-project
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
streamlit run app/app.py
```

---

# Future Improvements

- Source citations inside generated answers
- RAG evaluation using RAGAS
- Better prompt engineering
- Support for multiple UPSC papers
- Multi-turn conversational QA
- Knowledge graph integration

---

# Author

**Prakhya Khandelwal**

AI/ML | Retrieval-Augmented Generation | NLP | Large Language Models

GitHub:
https://github.com/gotnochill815-web

---

# License

MIT License
<img width="1919" height="1100" alt="image" src="https://github.com/user-attachments/assets/9eed6e9d-ebf9-4199-8fec-b307265a75fe" />
<img width="1918" height="1100" alt="image" src="https://github.com/user-attachments/assets/d35b171f-285b-4628-b270-316b1b0096ef" />

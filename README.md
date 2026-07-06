#  UPSC GS-II Curriculum Intelligence Engine

An AI-powered Retrieval-Augmented Generation (RAG) system for UPSC GS-II preparation that combines Constitutional Articles, Previous Year Questions (PYQs), structured knowledge notes, hybrid retrieval, reranking, and LLM-based answer generation.

---

## Overview

Preparing for UPSC Mains requires connecting constitutional provisions, governance concepts, and previous year questions. This project builds a structured knowledge base and uses Retrieval-Augmented Generation (RAG) to answer GS-II questions using relevant constitutional articles and PYQs.

The system combines lexical search, semantic search, reranking, and a language model to generate grounded answers.

---

## Features

- Constitution Article parser
- UPSC GS-II PYQ parser (2013–2022)
- Structured knowledge base generation
- Knowledge graph construction
- Hybrid Retrieval
  - BM25 lexical retrieval
  - Dense semantic retrieval (BGE embeddings)
  - Reciprocal Rank Fusion (RRF)
- CrossEncoder reranking
- Qwen-based answer generation
- Interactive Streamlit application
- Retrieval evaluation pipeline

---

## Architecture

```
                User Question
                      │
                      ▼
          Hybrid Retrieval (BM25 + Dense)
                      │
                      ▼
          Reciprocal Rank Fusion (RRF)
                      │
                      ▼
            CrossEncoder Reranker
                      │
                      ▼
           Top Relevant Documents
                      │
                      ▼
        Qwen 2.5 Answer Generator
                      │
                      ▼
              Final UPSC Answer
```

---

## Project Structure

```
upsc_rag_project/

├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── retrieval/
│   ├── bm25.py
│   ├── dense.py
│   ├── hybrid.py
│   └── reranker.py
│
├── knowledge_builder/
│   ├── collect_documents.py
│   ├── build_notes.py
│   ├── build_structured_notes.py
│   ├── build_graph.py
│   ├── keyword_extractor.py
│   ├── theme_extractor.py
│   └── related_topics.py
│
├── generator/
│   └── answer_generator.py
│
├── evaluation/
│   └── evaluate_retrieval.py
│
├── knowledge_base/
├── knowledge_base_structured/
├── knowledge_graph.html
├── requirements.txt
└── README.md
```

---

## Retrieval Pipeline

The retrieval system combines multiple retrieval techniques:

- BM25 lexical retrieval
- Dense semantic retrieval using BAAI BGE embeddings
- Reciprocal Rank Fusion (RRF)
- CrossEncoder reranking

This enables the system to retrieve relevant constitutional articles, knowledge notes, and previous year questions before generating answers.

---

## Models Used

| Component | Model |
|-----------|-------|
| Embeddings | BAAI/bge-base-en-v1.5 |
| Reranker | BAAI/bge-reranker-base |
| Generator | Qwen2.5-0.5B-Instruct |

---

## Dataset

The project includes:

- Indian Constitution Articles
- UPSC GS-II Previous Year Questions (2013–2022)
- Structured knowledge notes
- Knowledge graph

---

## Evaluation

The retrieval system was evaluated using manually annotated relevance labels.

### Retrieval Performance

| Metric | Score |
|---------|-------|
| Recall@5 | 0.590 |
| Recall@10 | 0.735 |
| Precision@5 | 0.590 |
| MRR | 1.000 |

---

## Demo

The Streamlit application allows users to:

- Ask UPSC GS-II questions
- Retrieve relevant constitutional articles
- View supporting PYQs
- Generate AI-assisted UPSC-style answers

---

## Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/upsc-rag-project.git

cd upsc-rag-project
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app/app.py
```

---

## Future Work

- Better citation-aware answer generation
- Multi-hop retrieval
- Graph-based retrieval
- Topic-wise analytics
- Answer evaluation
- Support for additional GS papers

---

## Author

**Prakhya Khandelwal**

AI/ML | Retrieval-Augmented Generation | NLP | Knowledge Graphs

GitHub: https://github.com/<your-username>

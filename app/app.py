import os
import sys
import pandas as pd
import streamlit as st

# =====================================================
# Project Root
# =====================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.insert(0, PROJECT_ROOT)

# =====================================================
# Imports
# =====================================================

from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from generator.answer_generator import AnswerGenerator

# =====================================================
# Streamlit Config
# =====================================================

st.set_page_config(
    page_title="UPSC GS-II Curriculum Intelligence Engine",
    page_icon="UPSC",
    layout="wide"
)

st.title(" UPSC GS-II Curriculum Intelligence Engine")

st.markdown("""
Hybrid Retrieval + Dense Retrieval + CrossEncoder + Qwen 2.5

Ask any UPSC GS-II question.
""")

# =====================================================
# Load Pipeline
# =====================================================

@st.cache_resource(show_spinner=False)
def load_pipeline():

    docs_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "documents.csv"
    )

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"documents.csv not found:\n{docs_path}"
        )

    docs = pd.read_csv(docs_path)

    bm25 = BM25Retriever()
    bm25.build(docs)

    dense = DenseRetriever()
    dense.build(docs)

    hybrid = HybridRetriever(
        bm25=bm25,
        dense=dense
    )

    reranker = CrossEncoderReranker()

    generator = AnswerGenerator()

    return docs, hybrid, reranker, generator


with st.spinner("Loading retrieval pipeline... This may take a minute on first launch."):

    try:

        documents, hybrid, reranker, generator = load_pipeline()

    except Exception as e:

        st.error("Failed to load the application.")

        st.exception(e)

        st.stop()

# =====================================================
# Input
# =====================================================

query = st.text_area(
    "Ask a GS-II Question",
    height=150,
    placeholder="Example: What are the discretionary powers of the Governor?"
)

# =====================================================
# Generate
# =====================================================

if st.button("Generate Answer", type="primary"):

    if not query.strip():

        st.warning("Please enter a question.")

        st.stop()

    with st.spinner("Retrieving relevant documents..."):

        retrieved = hybrid.search(
            query=query,
            top_k=50
        )

        ranked = reranker.rerank(
            query=query,
            documents=retrieved,
            top_k=10
        )

    st.success(f"Retrieved {len(ranked)} supporting documents.")

    # =====================================================
    # Retrieved Documents
    # =====================================================

    st.subheader(" Retrieved Documents")

    for row in ranked.itertuples():

        with st.expander(f"{row.title} ({row.source})"):

            st.markdown(f"**Document ID:** `{row.doc_id}`")

            if pd.notna(row.text):
                st.write(row.text)

    # =====================================================
    # Answer
    # =====================================================

    st.subheader(" AI Generated Answer")

    with st.spinner("Generating answer..."):

        answer = generator.generate(
            query=query,
            documents=ranked
        )

    st.markdown(answer)

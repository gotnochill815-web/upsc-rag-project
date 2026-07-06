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
    page_icon="",
    layout="wide"
)

st.title(" UPSC GS-II Curriculum Intelligence Engine")

st.markdown(
    """
Hybrid Retrieval + Dense Retrieval + CrossEncoder + Qwen 2.5

Ask any UPSC GS-II question.
"""
)

# =====================================================
# Load Pipeline
# =====================================================

@st.cache_resource
def load_pipeline():

    docs = pd.read_csv(
        os.path.join(
            PROJECT_ROOT,
            "data",
            "processed",
            "documents.csv"
        )
    )

    st.write("Building BM25...")

    bm25 = BM25Retriever()
    bm25.build(docs)

    st.write("Building Dense Index...")

    dense = DenseRetriever()
    dense.build(docs)

    hybrid = HybridRetriever(
        bm25,
        dense
    )

    reranker = CrossEncoderReranker()

    generator = AnswerGenerator()

    return docs, hybrid, reranker, generator


documents, hybrid, reranker, generator = load_pipeline()

# =====================================================
# Question Input
# =====================================================

query = st.text_area(
    "Ask a GS-II Question",
    height=150,
    placeholder="Example: Discuss the discretionary powers of the Governor."
)

# =====================================================
# Run Pipeline
# =====================================================

if st.button("Generate Answer"):

    if query.strip() == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("Retrieving relevant documents..."):

            retrieved = hybrid.search(
                query,
                top_k=50
            )

            ranked = reranker.rerank(
                query=query,
                documents=retrieved,
                top_k= 10
            )

        st.success(f"Retrieved {len(ranked)} documents.")

        # ==========================================
        # Retrieved Documents
        # ==========================================

        st.subheader(" Retrieved Documents")

        for row in ranked.itertuples():

            with st.expander(f"{row.title} ({row.source})"):

                st.markdown(f"**Document ID:** `{row.doc_id}`")

                if pd.notna(row.text):
                    st.write(row.text)

        # ==========================================
        # Answer
        # ==========================================

        st.subheader(" AI Generated Answer")

        with st.spinner("Generating answer..."):

            answer = generator.generate(
                query,
                ranked
            )

        st.markdown(answer)
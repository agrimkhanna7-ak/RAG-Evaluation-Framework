import streamlit as st


st.set_page_config(
    page_title="RAG Evaluation",
    page_icon="🤖",
    layout="wide",
)


st.title("🤖 RAG Evaluation Framework")

st.markdown(
    """
    ### Evaluate and explore a Retrieval-Augmented Generation system

    This application provides two interfaces:

    - **💬 Ask Question** — Ask questions against the NIST documents
      and receive an AI-generated answer with sources.

    - **📊 Evaluation Dashboard** — Compare Semantic, Hybrid, and MMR
      retrieval strategies using Recall@K and MRR, along with
      generation-quality metrics.
    """
)


st.info(
    "Use the sidebar to navigate between the "
    "Question Answering and Evaluation Dashboard pages."
)


st.markdown("---")

st.subheader("Retrieval Strategies Evaluated")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔎 Semantic Search")
    st.write(
        "Retrieves documents based on semantic similarity "
        "between the question and document chunks."
    )

with col2:
    st.markdown("### 🔀 Hybrid Search")
    st.write(
        "Combines semantic search with keyword-based BM25 "
        "retrieval."
    )

with col3:
    st.markdown("### 🧩 MMR Search")
    st.write(
        "Balances relevance and diversity when selecting "
        "retrieved chunks."
    )


st.markdown("---")

st.caption(
    "RAG Evaluation Framework | NIST AI Risk Management Documents"
)
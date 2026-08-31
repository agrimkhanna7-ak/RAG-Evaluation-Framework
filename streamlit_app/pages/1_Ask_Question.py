import sys
from pathlib import Path

import streamlit as st


# ---------------------------------------------------------
# ADD PROJECT ROOT TO PYTHON PATH
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# IMPORT RAG COMPONENTS
# ---------------------------------------------------------

from src.retrieval.vector_retriever import retrieve_documents
from src.generation.rag_generator import generate_answer

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Ask Question",
    page_icon="💬",
    layout="wide",
)


# ---------------------------------------------------------
# PAGE TITLE
# ---------------------------------------------------------

st.title("💬 Ask a Question")

st.markdown(
    """
    Ask a question about the NIST AI documents.

    The application uses **Semantic Search** to retrieve the
    most relevant document chunks and then generates an answer
    using the RAG pipeline.
    """
)


# ---------------------------------------------------------
# RETRIEVAL STRATEGY
# ---------------------------------------------------------

st.info(
    """
    🔎 **Retrieval Strategy: Semantic Search**

    Semantic Search is currently selected as the default
    strategy because it achieved the best retrieval performance
    on our evaluation dataset.
    """
)


# ---------------------------------------------------------
# QUESTION INPUT
# ---------------------------------------------------------

question = st.text_area(
    "Enter your question:",
    placeholder=(
        "Example: What are the four functions "
        "of the AI Risk Management Framework?"
    ),
    height=100,
)


# ---------------------------------------------------------
# ASK BUTTON
# ---------------------------------------------------------

ask_button = st.button(
    "🔍 Ask Question",
    type="primary",
)


# ---------------------------------------------------------
# PROCESS QUESTION
# ---------------------------------------------------------

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question before clicking "
            "'Ask Question'."
        )

    else:

        # -------------------------------------------------
        # RETRIEVE DOCUMENTS
        # -------------------------------------------------

        with st.spinner("Searching the documents..."):

            retrieved = retrieve_documents(
                query=question,
                k=5,
            )


        # -------------------------------------------------
        # EXTRACT DOCUMENTS
        # -------------------------------------------------

        retrieved_documents = [
            document
            for document, score in retrieved
        ]


        # -------------------------------------------------
        # GENERATE ANSWER
        # -------------------------------------------------

        with st.spinner("Generating answer..."):

            answer = generate_answer(
                query=question,
                retrieved_documents=retrieved_documents,
            )


        # -------------------------------------------------
        # DISPLAY ANSWER
        # -------------------------------------------------

        st.markdown("---")

        st.header("🤖 Generated Answer")

        st.write(answer)


        # -------------------------------------------------
        # DISPLAY RETRIEVED SOURCES
        # -------------------------------------------------

        st.markdown("---")

        st.header("📚 Retrieved Sources")

        st.write(
            "The following document chunks were retrieved "
            "by Semantic Search:"
        )


        for rank, (document, score) in enumerate(
            retrieved,
            start=1,
        ):

            source = document.metadata.get(
                "source",
                "Unknown",
            )

            page = document.metadata.get(
                "page",
                "Unknown",
            )


            with st.expander(
                f"Rank {rank} — {source} — Page {page}"
            ):

                st.write(
                    f"**Source:** {source}"
                )

                st.write(
                    f"**Page:** {page}"
                )

                st.write(
                    f"**Retrieval Score:** "
                    f"{score:.4f}"
                )

                st.write(
                    document.page_content
                )
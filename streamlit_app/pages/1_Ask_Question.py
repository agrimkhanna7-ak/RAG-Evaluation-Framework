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
    Ask questions about the NIST AI documents using the RAG copilot.
    Choose a retrieval strategy to compare the sources used to answer.
    """
)


# ---------------------------------------------------------
# RETRIEVAL STRATEGY
# ---------------------------------------------------------

strategy = st.selectbox(
    "Retrieval strategy",
    ["Semantic Search", "Hybrid Search", "MMR Search"],
    help=(
        "Semantic finds meaning matches, Hybrid combines meaning and keywords, "
        "and MMR balances relevance with source diversity."
    ),
)


# ---------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def prepare_document_index():
    from src.chunking.simple_chunker import chunk_documents
    from src.ingestion.pdf_loader import load_all_pdfs
    from src.vectorstore.chroma_store import ensure_vectorstore

    return ensure_vectorstore(
        documents=chunk_documents(load_all_pdfs()),
    )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Ask a question about the NIST AI documents")


# ---------------------------------------------------------
# RETRIEVE AND ANSWER
# ---------------------------------------------------------

if question:
    from src.generation.rag_generator import generate_answer

    with st.spinner("Preparing the document index (first run only)..."):
        prepare_document_index()

    if strategy == "Semantic Search":
        from src.retrieval.vector_retriever import retrieve_documents

        retrieved = retrieve_documents(question, k=5)
    elif strategy == "Hybrid Search":
        from src.retrieval.hybrid_retriever import hybrid_search

        retrieved = hybrid_search(question, k=5)
    else:
        from src.retrieval.mmr_retriever import retrieve_mmr_documents

        retrieved = [
            (document, None)
            for document in retrieve_mmr_documents(question, k=5)
        ]

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    retrieved_documents = [document for document, _ in retrieved]
    conversation_history = st.session_state.messages[:-1][-6:]
    with st.chat_message("assistant"):
        with st.spinner("Generating an answer from the documents..."):
            answer = generate_answer(
                query=question,
                retrieved_documents=retrieved_documents,
                chat_history=conversation_history,
            )
        st.markdown(answer)

        if retrieved:
            with st.expander(f"Sources from {strategy}"):
                for rank, (document, score) in enumerate(retrieved, start=1):
                    source = document.metadata.get("source", "Unknown")
                    page = document.metadata.get("page", "Unknown")
                    score_text = (
                        f" · Score: {score:.4f}"
                        if score is not None
                        else ""
                    )
                    st.markdown(
                        f"**{rank}. {source} · Page {page}{score_text}**"
                    )
                    st.write(document.page_content)
        else:
            st.warning("No document passages were retrieved.")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

if st.session_state.messages and st.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()
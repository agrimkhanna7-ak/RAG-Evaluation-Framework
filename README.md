# RAG Evaluation Framework

A Streamlit application for comparing semantic, hybrid, and MMR retrieval over the included NIST AI PDFs, with a Gemini-powered question-answering copilot.

## Windows setup

Use Python 3.13 or 3.12. From the repository root in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY` to your own Google AI Studio API key. Do not commit `.env`.

## Run the app

```powershell
streamlit run streamlit_app/app.py
```

Open the local URL printed by Streamlit. The first question builds the local Chroma index from `data/documents`; the embedding model is downloaded the first time it is needed. Both are cached locally for subsequent runs. The Ask Question page provides a chat interface and lets you choose Semantic, Hybrid, or MMR retrieval. The Evaluation Dashboard shows the saved comparison metrics.

## Verify the project data

```powershell
python -m src.evaluation.test_dataset
```

Files named `test_*.py` in this repository are executable experiments, not isolated pytest tests; some require the built vector index or a Gemini API key.

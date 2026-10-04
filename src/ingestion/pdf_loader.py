from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


def load_pdf(file_path: Path) -> list[Document]:
    """
    Load one PDF and convert each page into a LangChain Document.
    """

    reader = PdfReader(file_path)#it opens the pdf file and reads it

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):#goes through each page of the pdf file 
        text = page.extract_text() or ""#extracts the text from it

        if not text.strip():
            continue

        document = Document(#creates a new LangChain Document for each page
            page_content=text,
            metadata={
                "source": file_path.name,
                "page": page_number,
            },
        )

        documents.append(document)

    return documents


def load_all_pdfs() -> list[Document]:
    """
    Load all PDFs from the documents directory.
    """

    all_documents = []

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    for pdf_file in pdf_files:
        documents = load_pdf(pdf_file)
        all_documents.extend(documents)

    return all_documents
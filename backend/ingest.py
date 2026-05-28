from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import CHROMA_DB_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
import os

def ingest_document(file_path: str) -> dict:
    """
    Ingestion pipeline:
    1. Load PDF
    2. Split into chunks
    3. Embed each chunk
    4. Store in ChromaDB
    """

    # Step 1: Load PDF
    print(f"Loading document: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    # Step 2: Split into chunks
    # Why chunk? LLMs have context limits — we can't send a whole doc
    # RecursiveCharacterTextSplitter tries to split on paragraphs, then sentences
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP  # overlap so context isn't lost at boundaries
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Step 3 + 4: Embed and store in ChromaDB
    # HuggingFaceEmbeddings runs locally — no API call, no cost
    print("Embedding and storing in ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )

    print("Ingestion complete!")
    return {
        "status": "success",
        "pages": len(documents),
        "chunks": len(chunks)
    }
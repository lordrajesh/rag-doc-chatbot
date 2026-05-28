from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import sys

# So backend files can import each other
sys.path.append(os.path.dirname(__file__))

from ingest import ingest_document
from query import query_document

app = FastAPI(title="RAG Document Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def health_check():
    return {"status": "running"}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Receive PDF, save it, run ingestion pipeline"""
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = ingest_document(file_path)
    return result

@app.post("/query")
async def query(payload: dict):
    """Receive question, run RAG query, return answer"""
    question = payload.get("question", "")
    if not question:
        return {"error": "No question provided"}

    result = query_document(question)
    return result
# RAG Document Chatbot 📄

A production-inspired AI application that lets you upload a PDF and ask questions about it using **Retrieval-Augmented Generation (RAG)**.

## How It Works

**Ingestion Pipeline** (once per document):
1. Upload PDF via UI
2. Extract text using PyPDF
3. Split into overlapping chunks (500 tokens, 50 overlap)
4. Embed each chunk using HuggingFace model
5. Store embeddings in ChromaDB vector store

**Query Pipeline** (every question):
1. Embed the user question
2. Semantic search in ChromaDB (Top-3 chunks)
3. Build prompt with retrieved context + question
4. Send to Groq LLM
5. Return answer with source chunks

## Tech Stack

| Component | Free (This Project) | Production Equivalent |
|---|---|---|
| Vector Store | ChromaDB (local) | Pinecone / Azure AI Search |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Azure OpenAI text-embedding-ada-002 |
| LLM | Groq llama-3.3-70b-versatile | Azure OpenAI GPT-4 / AWS Bedrock |
| UI | Streamlit | React / Enterprise Portal |
| Backend | FastAPI | AWS Lambda / Azure Functions |
| File Storage | Local | AWS S3 / Azure Blob Storage |

## Project Structure

rag-doc-chatbot/

├── backend/
│   ├── main.py        # FastAPI app + endpoints
│   ├── ingest.py      # PDF ingestion + embedding pipeline
│   ├── query.py       # RAG query pipeline
│   └── config.py      # Settings and environment variables
├── frontend/
│   └── app.py         # Streamlit UI
├── data/              # Uploaded PDFs (gitignored)
├── .env               # API keys (gitignored)
└── requirements.txt

## Run Locally

### Prerequisites
- Python 3.10+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Setup
```bash
git clone https://github.com/YOUR_USERNAME/rag-doc-chatbot.git
cd rag-doc-chatbot
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

### Configure
Create a `.env` file:

GROQ_API_KEY=your_groq_api_key_here

### Start
Terminal 1 (Backend):
```bash
cd backend
uvicorn main:app --reload
```

Terminal 2 (Frontend):
```bash
cd frontend
streamlit run app.py
```

Visit `http://localhost:8501`

## Key Concepts Demonstrated
- **RAG (Retrieval-Augmented Generation)** — grounding LLM answers in real documents
- **Vector embeddings** — semantic search beyond keyword matching
- **LLM orchestration** — LangChain for pipeline management
- **REST API design** — FastAPI with clean separation of concerns
- **Production mapping** — every free component maps to an enterprise equivalent
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from config import CHROMA_DB_PATH, EMBEDDING_MODEL, GROQ_API_KEY, GROQ_MODEL, TOP_K_RESULTS

def query_document(question: str) -> dict:
    """
    RAG Query pipeline:
    1. Embed the question
    2. Search ChromaDB for top-K relevant chunks
    3. Build prompt with context + question
    4. Send to LLM (Groq)
    5. Return answer
    """

    # Step 1 + 2: Embed question and search ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )

    print(f"Searching for: {question}")
    relevant_chunks = vectorstore.similarity_search(question, k=TOP_K_RESULTS)
    print(f"Found {len(relevant_chunks)} relevant chunks")

    # Step 3: Build context from retrieved chunks
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

    # Step 4: Send to Groq LLM
    # This is the RAG prompt pattern — always used in production too
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name=GROQ_MODEL)

    messages = [
        SystemMessage(content="""You are a helpful assistant that answers questions 
        based on the provided document context. If the answer is not in the context, 
        say so honestly. Do not make up information."""),
        HumanMessage(content=f"""Context from document:
{context}

Question: {question}

Answer based only on the context above:""")
    ]

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": [
            {
                "page": chunk.metadata.get("page", "unknown"),
                "snippet": chunk.page_content[:200]
            }
            for chunk in relevant_chunks
        ]
    }
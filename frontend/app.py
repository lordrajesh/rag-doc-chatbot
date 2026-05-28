import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Document Chatbot", page_icon="📄")
st.title("📄 RAG Document Chatbot")
st.caption("Upload a PDF and ask questions about it")

# --- Section 1: Upload ---
st.header("1. Upload Document")
uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if uploaded_file:
    if st.button("Ingest Document"):
        with st.spinner("Ingesting document... (first time may take a minute)"):
            response = requests.post(
                f"{BACKEND_URL}/ingest",
                files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            )
            result = response.json()

        if result.get("status") == "success":
            st.success(f"✅ Ingested! {result['pages']} pages → {result['chunks']} chunks stored in vector DB")
        else:
            st.error("Something went wrong during ingestion")

# --- Section 2: Ask Questions ---
st.header("2. Ask a Question")
question = st.text_input("What do you want to know?")

if st.button("Ask") and question:
    with st.spinner("Searching document and generating answer..."):
        response = requests.post(
            f"{BACKEND_URL}/query",
            json={"question": question}
        )
        result = response.json()

    st.subheader("Answer")
    st.write(result.get("answer", "No answer returned"))

    with st.expander("View Source Chunks Retrieved from Vector DB"):
        for i, source in enumerate(result.get("sources", [])):
            st.markdown(f"**Chunk {i+1} — Page {source['page']}**")
            st.caption(source["snippet"])
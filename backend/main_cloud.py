from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pypdf
from docx import Document as DocxDocument
import uvicorn
import requests
import json

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Hugging Face Configuration
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")  # Set this as environment variable

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

try:
    collection = chroma_client.get_collection("documents")
except:
    collection = chroma_client.create_collection("documents")


class ChatRequest(BaseModel):
    message: str
    use_rag: bool = False


def query_huggingface(prompt: str, max_tokens: int = 500) -> str:
    """Query Hugging Face Inference API"""
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.95,
            "return_full_text": False
        }
    }

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "Sorry, I couldn't generate a response.")
        return "Sorry, I couldn't generate a response."
    except Exception as e:
        print(f"Hugging Face API error: {e}")
        return f"Error: Unable to connect to AI model. Please check your API token."


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """Simple text splitter"""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap

    return chunks


def extract_text_from_file(file_path: str, filename: str) -> str:
    """Extract text from various file formats"""
    try:
        if filename.endswith('.pdf'):
            with open(file_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text

        elif filename.endswith('.docx'):
            doc = DocxDocument(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text

        elif filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

        else:
            raise ValueError(f"Unsupported file type: {filename}")

    except Exception as e:
        raise Exception(f"Error extracting text: {str(e)}")


@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "RAG Chatbot API (Cloud Version)",
        "model": "Mistral-7B via Hugging Face",
        "endpoints": {
            "chat": "/chat",
            "upload": "/upload",
            "documents": "/documents"
        }
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        if request.use_rag:
            # RAG mode - search documents
            query_embedding = embedding_model.encode(request.message).tolist()

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )

            if results['documents'] and results['documents'][0]:
                context = "\n\n".join(results['documents'][0])
                sources = results['metadatas'][0] if results['metadatas'] else []

                prompt = f"""Based on the following context, answer the question. If the answer is not in the context, say so.

Context:
{context}

Question: {request.message}

Answer:"""

                response = query_huggingface(prompt)

                return {
                    "response": response,
                    "sources": sources,
                    "mode": "rag"
                }
            else:
                return {
                    "response": "No relevant documents found. Please upload documents first or disable RAG mode.",
                    "sources": [],
                    "mode": "rag"
                }
        else:
            # Normal chat mode
            response = query_huggingface(request.message)
            return {
                "response": response,
                "mode": "normal"
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        text = extract_text_from_file(file_path, file.filename)
        chunks = split_text(text)

        for i, chunk in enumerate(chunks):
            embedding = embedding_model.encode(chunk).tolist()

            collection.add(
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "source": file.filename,
                    "chunk": i,
                    "total_chunks": len(chunks)
                }],
                ids=[f"{file.filename}_{i}"]
            )

        return {
            "message": f"Successfully processed {file.filename}",
            "chunks": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    try:
        all_docs = collection.get()

        if not all_docs['metadatas']:
            return {"documents": []}

        unique_sources = set()
        for metadata in all_docs['metadatas']:
            if 'source' in metadata:
                unique_sources.add(metadata['source'])

        return {"documents": list(unique_sources)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents")
async def clear_documents():
    try:
        global collection
        chroma_client.delete_collection("documents")
        collection = chroma_client.create_collection("documents")

        for file in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        return {"message": "All documents cleared successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting cloud server on port {port}...")
    print(f"Using Hugging Face Mistral-7B model")
    print(f"HF_API_TOKEN configured: {'Yes' if HF_API_TOKEN else 'No - Please set it!'}")
    uvicorn.run(app, host="0.0.0.0", port=port)
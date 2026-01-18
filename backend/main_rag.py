from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import json
from typing import AsyncGenerator, List, Optional
import chromadb
from sentence_transformers import SentenceTransformer
import pypdf
import docx
import os
from pathlib import Path
import re

app = FastAPI(title="AI Chatbot with RAG")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models
class ChatRequest(BaseModel):
    message: str
    use_rag: bool = False


class ChatResponse(BaseModel):
    response: str


# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize RAG components
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()

# Create or get collection
try:
    collection = chroma_client.get_collection(name="documents")
except:
    collection = chroma_client.create_collection(name="documents")


# Simple text splitter function
def split_text(text, chunk_size=500, chunk_overlap=50):
    """Split text into chunks"""
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        # Try to break at sentence end
        if end < text_len:
            # Look for sentence endings
            for i in range(end, max(start + chunk_overlap, end - 100), -1):
                if text[i] in '.!?\n':
                    end = i + 1
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap

    return chunks


# Helper functions
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF"""
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = pypdf.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX"""
    doc = docx.Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])


def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def process_document(file_path: str, filename: str) -> int:
    """Process document and add to vector store"""
    # Extract text based on file type
    ext = Path(filename).suffix.lower()

    if ext == '.pdf':
        text = extract_text_from_pdf(file_path)
    elif ext == '.docx':
        text = extract_text_from_docx(file_path)
    elif ext == '.txt':
        text = extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Split text into chunks using our simple splitter
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)

    # Generate embeddings
    embeddings = embedding_model.encode(chunks).tolist()

    # Add to ChromaDB
    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk": i} for i in range(len(chunks))]

    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )

    return len(chunks)


def retrieve_context(query: str, n_results: int = 3) -> tuple[str, List[dict]]:
    """Retrieve relevant context from vector store"""
    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    if not results['documents'] or not results['documents'][0]:
        return "", []

    # Combine retrieved chunks
    context = "\n\n".join(results['documents'][0])

    # Get sources
    sources = []
    for i, metadata in enumerate(results['metadatas'][0]):
        sources.append({
            "source": metadata['source'],
            "chunk": metadata['chunk'],
            "text": results['documents'][0][i][:200] + "..."
        })

    return context, sources


# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "AI Chatbot API with RAG",
        "status": "running",
        "model": MODEL_NAME,
        "documents": collection.count()
    }


@app.get("/health")
async def health_check():
    """Check if Ollama is running"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                OLLAMA_URL,
                json={"model": MODEL_NAME, "prompt": "test", "stream": False},
                timeout=10.0
            )
            return {
                "status": "healthy",
                "model": MODEL_NAME,
                "documents": collection.count()
            }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Save file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Process document
        chunks_count = process_document(str(file_path), file.filename)

        return {
            "message": "Document uploaded successfully",
            "filename": file.filename,
            "chunks": chunks_count,
            "total_documents": collection.count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    try:
        all_data = collection.get()

        # Get unique sources
        sources = set()
        if all_data['metadatas']:
            sources = {meta['source'] for meta in all_data['metadatas']}

        return {
            "total_chunks": collection.count(),
            "documents": list(sources)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents")
async def clear_documents():
    """Clear all documents from vector store"""
    global collection
    try:
        chroma_client.delete_collection(name="documents")
        collection = chroma_client.create_collection(name="documents")
        return {"message": "All documents cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def generate_stream(prompt: str, context: str = "") -> AsyncGenerator[str, None]:
    """Generator function for streaming responses"""
    try:
        # Add context to prompt if provided
        if context:
            full_prompt = f"""Based on the following context, answer the question. If the answer is not in the context, say so.

Context:
{context}

Question: {prompt}

Answer:"""
        else:
            full_prompt = prompt

        async with httpx.AsyncClient() as client:
            async with client.stream(
                    "POST",
                    OLLAMA_URL,
                    json={
                        "model": MODEL_NAME,
                        "prompt": full_prompt,
                        "stream": True
                    },
                    timeout=120.0
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                chunk = {
                                    "content": data["response"],
                                    "done": data.get("done", False)
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"

                                if data.get("done", False):
                                    yield "data: [DONE]\n\n"
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        error_msg = {"error": str(e)}
        yield f"data: {json.dumps(error_msg)}\n\n"


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint with optional RAG"""
    context = ""
    sources = []

    # Retrieve context if RAG is enabled
    if request.use_rag and collection.count() > 0:
        context, sources = retrieve_context(request.message)

        # Send sources first
        if sources:
            sources_msg = {"sources": sources}

            async def stream_with_sources():
                yield f"data: {json.dumps(sources_msg)}\n\n"
                async for chunk in generate_stream(request.message, context):
                    yield chunk

            return StreamingResponse(
                stream_with_sources(),
                media_type="text/event-stream"
            )

    return StreamingResponse(
        generate_stream(request.message, context),
        media_type="text/event-stream"
    )


@app.post("/chat")
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint"""
    try:
        context = ""
        if request.use_rag and collection.count() > 0:
            context, _ = retrieve_context(request.message)

        full_prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {request.message}

Answer:""" if context else request.message

        async with httpx.AsyncClient() as client:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": full_prompt,
                    "stream": False
                },
                timeout=60.0
            )

            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Ollama API error")

            result = response.json()
            return ChatResponse(response=result.get("response", ""))

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    print(f"Starting server with RAG enabled...")
    print(f"Upload directory: {UPLOAD_DIR.absolute()}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
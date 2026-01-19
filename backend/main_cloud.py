from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import os
import chromadb
from sentence_transformers import SentenceTransformer
import pypdf
from docx import Document as DocxDocument
import uvicorn
import json
from dotenv import load_dotenv
from pathlib import Path
from groq import Groq

# Load environment variables from .env file in parent directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

app = FastAPI()

# CORS - Allow all origins (fine for public chatbot)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = "uploads"
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# Groq Configuration (FREE and FAST!)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Initialize embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedding model loaded")

# Initialize ChromaDB
print(f"Initializing ChromaDB at {CHROMA_DB_PATH}...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

try:
    collection = chroma_client.get_collection("documents")
    print(f"✅ ChromaDB loaded with {collection.count()} documents")
except:
    collection = chroma_client.create_collection("documents")
    print("✅ ChromaDB collection created")


class ChatRequest(BaseModel):
    message: str
    use_rag: bool = False


def query_groq(prompt: str, stream: bool = False):
    """Query Groq API (FREE Llama 3 models!)"""
    if not groq_client:
        raise Exception("GROQ_API_KEY not configured")

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # FREE Llama 3.3 70B!
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
            stream=stream
        )

        if stream:
            return completion
        else:
            return completion.choices[0].message.content
    except Exception as e:
        print(f"Groq API error: {e}")
        raise Exception(f"Error: {str(e)}")


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


# Serve frontend HTML
@app.get("/")
async def serve_frontend():
    """Serve the chatbot interface"""
    html_path = Path(__file__).parent / "chatbot_rag.html"
    if html_path.exists():
        return FileResponse(html_path)
    else:
        return {
            "status": "online",
            "message": "RAG Chatbot API (Groq Version)",
            "model": "Llama 3.3 70B via Groq",
            "api_configured": "Yes" if GROQ_API_KEY else "No - Please set GROQ_API_KEY",
            "endpoints": {
                "chat": "/chat",
                "chat_stream": "/chat/stream",
                "upload": "/upload",
                "documents": "/documents",
                "health": "/health"
            }
        }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "groq_configured": bool(GROQ_API_KEY),
        "chromadb_initialized": collection is not None,
        "documents_count": collection.count() if collection else 0,
        "model": "llama-3.3-70b-versatile"
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

                response = query_groq(prompt)

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
            response = query_groq(request.message)
            return {
                "response": response,
                "mode": "normal"
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming endpoint for real-time responses"""

    async def generate():
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

                    # Send sources first
                    yield f"data: {json.dumps({'sources': sources})}\n\n"

                    prompt = f"""Based on the following context, answer the question. If the answer is not in the context, say so.

Context:
{context}

Question: {request.message}

Answer:"""

                    # Stream from Groq
                    stream = query_groq(prompt, stream=True)

                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            yield f"data: {json.dumps({'content': content})}\n\n"

                    yield f"data: [DONE]\n\n"
                else:
                    error_msg = "No relevant documents found. Please upload documents first or disable RAG mode."
                    yield f"data: {json.dumps({'content': error_msg})}\n\n"
                    yield f"data: [DONE]\n\n"
            else:
                # Normal chat mode - stream from Groq
                stream = query_groq(request.message, stream=True)

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        yield f"data: {json.dumps({'content': content})}\n\n"

                yield f"data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'content': f'Error: {str(e)}'})}\n\n"
            yield f"data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


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
            "filename": file.filename,
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
    print(f"\n🚀 Starting RAG Chatbot on port {port}...")
    print(f"🤖 Model: Llama 3.3 70B via Groq (FREE & FAST!)")
    print(f"🔑 GROQ_API_KEY configured: {'Yes ✅' if GROQ_API_KEY else 'No ❌ - Please set it!'}")
    print(f"💾 ChromaDB path: {CHROMA_DB_PATH}")
    print(f"📊 Documents loaded: {collection.count()}")
    uvicorn.run(app, host="0.0.0.0", port=port)
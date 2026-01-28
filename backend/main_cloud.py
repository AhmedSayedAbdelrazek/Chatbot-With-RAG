from fastapi import FastAPI, UploadFile, File, HTTPException , BackgroundTasks
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
from collections import defaultdict, deque
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
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = str(BASE_DIR / "uploads")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma_db"))
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# Groq Configuration (FREE and FAST!)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
MAX_TURNS = int(os.getenv("MAX_TURNS", "10"))  # keep last 10 user+assistant messages
# Lazy-load embedding model (important for Render: avoids port-scan timeout)
embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("Loading embedding model (lazy)...")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Embedding model loaded")
    return embedding_model

# session_id -> deque of messages [{"role": "...", "content": "..."}]
SESSION_HISTORY = defaultdict(lambda: deque(maxlen=MAX_TURNS * 2))

def get_history(session_id: Optional[str]):
    if not session_id:
        return []
    return list(SESSION_HISTORY[session_id])

def add_to_history(session_id: Optional[str], role: str, content: str):
    if not session_id:
        return
    SESSION_HISTORY[session_id].append({"role": role, "content": content})

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
    session_id: Optional[str] = None

def query_groq(messages, stream: bool = False):
    """Query Groq API (FREE Llama 3 models!)"""
    if not groq_client:
        raise Exception("GROQ_API_KEY not configured")

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # FREE Llama 3.3 70B!
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            stream=stream
        )
        return completion
    except Exception as e:
        print(f"Groq API error: {e}")
        raise Exception(f"Error: {str(e)}")


def split_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 100):
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

def process_and_index_file(file_path: str, filename: str):
    try:
        print(f"BG: start processing {filename}", flush=True)

        text = extract_text_from_file(file_path, filename)
        chunks = split_text(text, chunk_size=1200, chunk_overlap=100)

        # safety limit (Render free can die on huge docs)
        MAX_CHUNKS = 80
        chunks = chunks[:MAX_CHUNKS]

        model = get_embedding_model()

        for i, chunk in enumerate(chunks):
            emb = model.encode(chunk).tolist()
            collection.add(
                embeddings=[emb],
                documents=[chunk],
                metadatas=[{"source": filename, "chunk": i}],
                ids=[f"{filename}_{i}"],
            )

        print(f"BG: done {filename} chunks={len(chunks)}", flush=True)

    except Exception as e:
        print(f"BG ERROR {filename}: {repr(e)}", flush=True)

# Serve frontend HTML
@app.get("/")
async def serve_frontend():
    # ✅ absolute path in Render container
    html_path = Path("/opt/render/project/src/backend/chatbot_rag.html")

    # ✅ fallback to local relative path (when running locally)
    if not html_path.exists():
        html_path = Path(__file__).parent / "chatbot_rag.html"

    return FileResponse(html_path)

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
            query_embedding = get_embedding_model().encode(request.message).tolist()

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
                add_to_history(request.session_id, "user", request.message)

                messages = [
                    {"role": "system",
                     "content": "You are a helpful assistant. Use ONLY the provided context. If not found, say you don't know."},
                    {"role": "system", "content": f"Context:\n{context}"},
                    *get_history(request.session_id),
                    {"role": "user", "content": request.message},
                ]

                completion = query_groq(messages, stream=False)
                response = completion.choices[0].message.content

                add_to_history(request.session_id, "assistant", response)

            else:
                return {
                    "response": "No relevant documents found. Please upload documents first or disable RAG mode.",
                    "sources": [],
                    "mode": "rag"
                }
        else:
            # Normal chat mode
            add_to_history(request.session_id, "user", request.message)

            messages = [
                {"role": "system", "content": "You are a helpful assistant. Keep answers concise and clear."},
                *get_history(request.session_id),
                {"role": "user", "content": request.message},
            ]

            completion = query_groq(messages, stream=False)
            response = completion.choices[0].message.content

            add_to_history(request.session_id, "assistant", response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming endpoint for real-time responses"""

    async def generate():
        try:
            if request.use_rag:
                # RAG mode - search documents
                query_embedding = get_embedding_model().encode(request.message).tolist()

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
                # ✅ Add user message to memory
                add_to_history(request.session_id, "user", request.message)

                # ✅ Build messages with memory
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Keep answers concise and clear."},
                    *get_history(request.session_id),
                    {"role": "user", "content": request.message},
                ]

                # ✅ Stream from Groq
                stream = query_groq(messages, stream=True)

                assistant_text = ""

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        assistant_text += content
                        yield f"data: {json.dumps({'content': content})}\n\n"

                # ✅ Save assistant response in memory
                add_to_history(request.session_id, "assistant", assistant_text)

                yield f"data: [DONE]\n\n"


        except Exception as e:
            yield f"data: {json.dumps({'content': f'Error: {str(e)}'})}\n\n"
            yield f"data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        print("UPLOAD: received:", file.filename, flush=True)

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        print("UPLOAD: saved:", file_path, flush=True)

        # ✅ do heavy work in background
        background_tasks.add_task(process_and_index_file, file_path, file.filename)

        # ✅ respond immediately so no 502
        return {
            "filename": file.filename,
            "message": "Upload received. Processing started in background."
        }

    except Exception as e:
        print("UPLOAD ERROR:", repr(e), flush=True)
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
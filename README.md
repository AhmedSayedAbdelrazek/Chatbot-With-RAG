# 🤖 RAG-Powered Chatbot with Llama 3 (Groq) – Dockerized

A **production-ready AI chatbot** with **Retrieval-Augmented Generation (RAG)** built using **FastAPI**, **ChromaDB**, and **Llama 3.3 (70B) via Groq**, fully containerized with Docker and persistent storage.

## ✨ Features

- 💬 **Real-time Chat** - Streaming responses with Llama 3.2
- 🧠 **Conversation Memory** –  Session-based conversation memory (multi-turn)
- 📚 **RAG Support** - Upload and query your documents (PDF, DOCX, TXT)
- 🔍 **Semantic Search** - Find relevant information using vector embeddings
- 🎨 **Modern UI** - Clean, responsive interface
- 🐳 **Docker & Docker Compose** - Support
- ❤️ Health check endpoint for production monitoring
- ⚡ Ultra-fast inference via Groq (Llama 3.3 70B)
## 🏗️ Architecture


```
Client
|
v
FastAPI Backend
├── Groq (Llama 3.3 70B)
├── Sentence-Transformers
├── ChromaDB (Vector Store)
└── Docker Volumes (Persistence)

```

---
## 🛠️ Tech Stack

**Backend:**
- FastAPI - Modern Python web framework
- Ollama - Local LLM runtime
- ChromaDB - Vector database
- Sentence Transformers - Text embeddings
- Groq SDK
- httpx

**AI Model**
- Llama 3.3 70B (Groq)

**Infrastructure**
- Docker
- Docker Compose

**Frontend:**
- Pure HTML/CSS/JavaScript
- No framework dependencies
---

## 📋 Prerequisites

- Docker
- Docker Compose
- Groq API Key (https://console.groq.com)
- Python 3.8 or higher
- Ollama installed ([Download here](https://ollama.ai))
- Git

---
## 🚀 Run with Docker

### 1. Clone the Repository

```bash
git clone https://github.com/AhmedSayedAbdelrazek/Chatbot-With-RAG.git
cd chatbot-project
```

### 2. Create .env

```env
GROQ_API_KEY=your_groq_api_key_here
```
### 3. Build & run

```bash
docker compose up --build
```
- The server will start on `http://localhost:8000`
- Swagger Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### Health Check
```http
GET /health
```
Example response:
``` json
{
  "status": "healthy",
  "groq_configured": true,
  "chromadb_initialized": true,
  "documents_count": 0,
  "model": "llama-3.3-70b-versatile"
}
```
### 📄 RAG Flow
- Upload documents (PDF / DOCX / TXT)
- Text is chunked and embedded
- Stored in ChromaDB
- User query → semantic search
- Context injected into LLM
- Grounded response generated

### 🧠 Conversation Memory
- Session-based memory per user
- Stores last N turns
- Automatically injected into prompts
- Works in chat & RAG mode

Example:
```sql
User: My name is Ahmed
Assistant: Nice to meet you, Ahmed
User: What is my name?
Assistant: Your name is Ahmed

```

## 📁 Project Structure

```bash
chatbot-project/
├── backend/
│   ├── main_cloud.py          # FastAPI server with RAG
│   ├── chatbot_rag.html       # Frontend with RAG support
│   ├── chroma_db/
│   └── uploads/               # Uploaded documents (auto-created)
├── .gitignore
├── Dockerfile
├── .env
├── requirements.txt           # Python dependencies
└── README.md
```

## 💡 How to Use

### Normal Chat Mode

1. Open the frontend in your browser
2. Type your message and press Enter
3. Get instant responses from Llama 3.2

### RAG Mode (Document Q&A)

1. Click "📄 Upload Document" in the sidebar
2. Select a PDF, DOCX, or TXT file
3. Wait for processing confirmation
4. Toggle "Use RAG Mode" ON
5. Ask questions about your documents!

### Example Queries

**After uploading a research paper:**
- "What is the main methodology used?"
- "Summarize the key findings"
- "What are the limitations mentioned?"

**After uploading meeting notes:**
- "What were the action items?"
- "Who is responsible for the marketing campaign?"
- "When is the next deadline?"
## 🧠 Conversation Memory

The chatbot supports **session-based conversation memory**, allowing it to remember previous messages and respond with proper context during multi-turn conversations.

### How it works
- Each browser session is assigned a unique `session_id`
- The backend stores the last **N user–assistant messages** per session
- Conversation history is injected into the LLM prompt automatically
- Memory works in both **Normal Chat** and **RAG Mode**

### Example
- User: My name is Ahmed 
- Assistant: Nice to meet you, Ahmed
- User: What is my name?
- Assistant: Your name is Ahmed
## 🔧 Configuration

### Change the Model

Edit `main_rag.py` and modify:

```python
OLLAMA_MODEL = "llama3.2"  # Change to any Ollama model
```

### Adjust Chunk Size

In `main_cloud.py`, modify the text splitting parameters:

```python
def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
```

## 🐛 Troubleshooting

**Ollama not responding?**
```bash
# Check if Ollama is running:
ollama list

# Restart Ollama if needed
```

**Port 8000 already in use?**
```bash
# Change the port in main.py:
uvicorn.run(app, host="0.0.0.0", port=8001)
```

**Module not found errors?**
```bash
# Reinstall dependencies:
pip install -r requirements.txt --force-reinstall
```

## 📝 API Endpoints

- `POST /chat` - Send chat messages
- `POST /upload` - Upload documents
- `GET /documents` - List uploaded documents
- `DELETE /documents` - Clear all documents
- `GET /` - Health check

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai) for local LLM runtime
- [ChromaDB](https://www.trychroma.com/) for vector database
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [Sentence Transformers](https://www.sbert.net/) for embeddings

## 📧 Contact

Project Link: [https://github.com/AhmedSayedAbdelrazek/Chatbot-With-RAG](https://github.com/AhmedSayedAbdelrazek/Chatbot-With-RAG)

---

Made with ❤️ and AI

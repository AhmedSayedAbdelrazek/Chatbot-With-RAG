# 🤖 RAG-Powered Chatbot with Llama 3.2

A full-stack AI chatbot application with Retrieval-Augmented Generation (RAG) capabilities, built with FastAPI, Ollama, and ChromaDB.

## ✨ Features

- 💬 **Real-time Chat** - Streaming responses with Llama 3.2
- 📚 **RAG Support** - Upload and query your documents (PDF, DOCX, TXT)
- 🔍 **Semantic Search** - Find relevant information using vector embeddings
- 🎨 **Modern UI** - Clean, responsive interface
- 🚀 **100% Local** - No API costs, runs entirely on your machine
- 📄 **Multi-format Support** - PDF, Word documents, and text files

## 🛠️ Tech Stack

**Backend:**
- FastAPI - Modern Python web framework
- Ollama - Local LLM runtime
- ChromaDB - Vector database
- Sentence Transformers - Text embeddings

**Frontend:**
- Pure HTML/CSS/JavaScript
- No framework dependencies

**AI Model:**
- Llama 3.2 (via Ollama)

## 📋 Prerequisites

- Python 3.8 or higher
- Ollama installed ([Download here](https://ollama.ai))
- Git

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AhmedSayedAbdelrazek/Chatbot-With-RAG.git
cd chatbot-project
```

### 2. Install Ollama and Pull Llama 3.2

```bash
# Install Ollama from https://ollama.ai
# Then pull the model:
ollama pull llama3.2
```

### 3. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

## 🎯 Usage

### Start the Backend Server

```bash
cd backend
python main.py
```

The server will start on `http://localhost:8000`

### Open the Frontend

Simply open `chatbot_rag.html` in your web browser.

## 📁 Project Structure

```
chatbot-project/
├── backend/
│   ├── main_rag.py              # FastAPI server with RAG
│   ├── requirements.txt     # Python dependencies
│   └── uploads/             # Uploaded documents (auto-created)
├── chatbot_rag.html         # Frontend with RAG support
├── .gitignore
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

## 🔧 Configuration

### Change the Model

Edit `main_rag.py` and modify:

```python
OLLAMA_MODEL = "llama3.2"  # Change to any Ollama model
```

### Adjust Chunk Size

In `main_rag.py`, modify the text splitting parameters:

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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai) for local LLM runtime
- [ChromaDB](https://www.trychroma.com/) for vector database
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [Sentence Transformers](https://www.sbert.net/) for embeddings

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/AhmedSayedAbdelrazek/Chatbot-With-RAG](https://github.com/AhmedSayedAbdelrazek/Chatbot-With-RAG)

---

Made with ❤️ and AI

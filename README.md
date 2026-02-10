# 🤖 RAG Research Assistant

> A privacy-first, locally-run Retrieval-Augmented Generation (RAG) system for intelligent conversations with your documents.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Overview

RAG Research Assistant enables **private, context-aware conversations** with your documents using state-of-the-art retrieval and generation techniques. Everything runs locally on your machine—no data leaves your system.

### What is RAG?

**Retrieval-Augmented Generation** combines:

- 🔍 **Vector Search**: Find relevant document chunks using semantic similarity
- 🤖 **LLM Generation**: Generate accurate answers grounded in your documents
- 💬 **Conversation Memory**: Maintain context across multi-turn conversations

---

## ✨ Key Features

| Feature | Description |
| --------- | ------------- |
| 🔒 **100% Private** | All processing happens locally—no cloud APIs, no data sharing |
| 🚀 **Production-Ready** | FastAPI backend with async support and proper error handling |
| 🎨 **Modern UI** | Clean Gradio interface for document management and chat |
| 📚 **Multi-Format** | Support for PDF and text files with intelligent chunking |
| 🧠 **Smart Context** | Automatic conversation windowing and summarization |
| 🔄 **Persistent Storage** | ChromaDB vector store and SQLite metadata database |
| 🧪 **Well-Tested** | Comprehensive test suite with pytest |

---

## 🏗️ Architecture

```bash
┌─────────────────────────────────────┐
│      Gradio UI / REST API           │
│   (User Interface Layer)            │
└─────────────────────────────────────┘
                ↓
┌─────────────────────────────────────┐
│     OrchestratorService             │
│   (Business Logic Layer)            │
└─────────────────────────────────────┘
                ↓
┌──────────────┬────────────────┬────────┐
│ DocumentSvc  │ ConversationMgr│   ...  │
│ (Services)   │  (Services)    │        │
└──────────────┴────────────────┴────────┘
                ↓
┌──────────────┬──────────────┬────────┐
│  ChromaDB    │   Ollama     │ SQLite │
│ (VectorStore)│   (LLM)      │  (DB)  │
└──────────────┴──────────────┴────────┘
```

### Project Structure

```bash
rag-research-assistant/
├── src/
│   ├── api/              # FastAPI routes and schemas
│   ├── config/           # Configuration and settings
│   ├── conversation/     # Chat history and context management
│   ├── documents/        # Document ingestion and chunking
│   ├── infra/
│   │   ├── db/          # SQLite session management
│   │   ├── embeddings/  # Sentence Transformers integration
│   │   ├── llm/         # Ollama LLM client
│   │   └── vectorstore/ # ChromaDB wrapper
│   ├── orchestrator/    # Main business logic coordinator
│   └── utils/           # Logging and utilities
├── ui/
│   ├── gradio_app.py    # Production UI (use this)
│   └── main.py          # UI experimentation sandbox
├── tests/               # Pytest test suite
├── data/                # Auto-created: stores DB and vectors
├── requirements.txt
└── README.md
```

---

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.11 or higher** ([Download](https://www.python.org/downloads/))
- **Ollama** installed and running ([Installation Guide](https://ollama.ai/download))
- **Git** for cloning the repository

### Verify Ollama Installation

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Should return a JSON response with available models
```

---

## 🚀 Quick Start

### 1. Clone and Install Dependencies

```bash
# Clone the repository
git clone https://github.com/dsbarik/rag-research-assistant.git
cd rag-research-assistant

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download the LLM Model

```bash
# Pull the default model (llama3.2)
ollama pull llama3.2

# Verify the model is available
ollama list
```

### 3. Start the Backend API

Open a terminal and run:

```bash
fastapi dev src/api/main.py
```

✅ API will be available at: **<http://localhost:8000>**  
📚 Interactive docs at: **<http://localhost:8000/docs>**

### 4. Start the Gradio UI

Open a **new terminal** (keep the API running) and run:

```bash
python ui/gradio_app.py
```

✅ Web interface will open at: **<http://localhost:7860>**

---

## 💡 Usage Guide

### Using the Gradio UI

1. **Upload Documents**
   - Navigate to the "📂 Knowledge Base" tab
   - Click "Select PDF Files" and choose your documents
   - Click "🚀 Ingest Document" to process them

2. **Chat with Your Documents**
   - Switch to the "💬 Chat" tab
   - Ask questions about your uploaded documents
   - The system will retrieve relevant context and generate answers

3. **Manage Documents**
   - View all uploaded documents in the table
   - Delete documents you no longer need
   - Refresh the list to see updates

### Using the REST API

#### Upload a Document

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf"
```

**Response:**

```json
{
  "status": "success",
  "document_id": 1,
  "chunks_processed": 42
}
```

#### Chat with Context

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the main findings?",
    "conversation_id": null
  }'
```

**Response:**

```json
{
  "conversation_id": 1,
  "response": "Based on the documents, the main findings are...",
  "sources": [
    {"filename": "document.pdf", "chunk_id": "doc_1_chunk_3"}
  ]
}
```

#### List All Documents

```bash
curl -X GET "http://localhost:8000/api/v1/documents"
```

#### Delete a Document

```bash
curl -X DELETE "http://localhost:8000/api/v1/documents/1"
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root (optional):

```env
# LLM Configuration
LLM_MODEL_NAME=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

# Paths (defaults to ./data/)
DATA_DIR=./data
DOCUMENTS_DIR=./data/documents
VECTOR_STORE_DIR=./data/vector_store
DATABASE_PATH=./data/documents.db
```

### Changing the LLM Model

Edit `src/config/settings.py`:

```python
LLM_MODEL_NAME = "llama3.2"  # Change to any Ollama model
```

Available models: `llama3.2`, `mistral`, `codellama`, etc. ([Full list](https://ollama.ai/library))

---

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_orchestrator.py -v
```

---

## 🔧 Troubleshooting

### Issue: "Connection refused" when starting API

**Solution:** Ensure Ollama is running:

```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# If not running, start Ollama
ollama serve
```

### Issue: "Module not found: sentence_transformers"

**Solution:** Install the missing dependency:

```bash
pip install sentence-transformers
```

### Issue: Documents not being ingested

**Possible causes:**

- File is a duplicate (check hash in database)
- Unsupported file format (only PDF and TXT supported)
- File is corrupted or empty

**Debug:**

```bash
# Check API logs for detailed error messages
# Check data/documents/ directory for uploaded files
```

### Issue: Slow response times

**Solutions:**

- Use a smaller/faster LLM model (e.g., `llama3.2:1b`)
- Reduce chunk retrieval limit in `orchestrator/service.py`
- Ensure Ollama has sufficient RAM allocated

---

## 🗺️ Roadmap

This is an **ongoing project** with planned enhancements:

- [ ] **Multi-Agent System** (`agents/`) - Specialized agents for different tasks
- [ ] **Knowledge Graph Integration** (`kg/`) - Entity extraction and graph-based verification
- [ ] **Advanced Retrieval** (`retrieval/`) - Hybrid search, re-ranking, query expansion
- [ ] **Explainability** (`explanation/`) - Visualize retrieval and generation process
- [ ] **FAISS Support** - Alternative vector store for larger datasets
- [ ] **More File Formats** - DOCX, Markdown, HTML support
- [ ] **Streaming Responses** - Real-time token streaming in UI
- [ ] **Multi-User Support** - User authentication and document isolation

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Ollama](https://ollama.ai/)** - Local LLM inference
- **[ChromaDB](https://www.trychroma.com/)** - Vector database
- **[Sentence Transformers](https://www.sbert.net/)** - Embedding models
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern web framework
- **[Gradio](https://gradio.app/)** - UI framework

---

## 📞 Support

- 📧 Email: <your-email@example.com>
- 🐛 Issues: [GitHub Issues](https://github.com/dsbarik/rag-research-assistant/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/dsbarik/rag-research-assistant/discussions)

---

**Built with ❤️ for privacy-conscious researchers and developers**

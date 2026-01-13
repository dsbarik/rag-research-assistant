
# 🤖 RAG Research Assistant

RAG Research Assistant is a modular, high-performance Retrieval-Augmented Generation (RAG) system. It enables users to have private, context-aware conversations with their own documents by leveraging local Large Language Models (LLMs) and vector search.

## 🌟 Key Features

* **100% Local & Private**: Runs entirely on your machine using **Ollama** for LLM inference and **Sentence Transformers** for local embeddings.
* **Intelligent Retrieval**: Utilizes **ChromaDB** for persistent vector storage and similarity search.
* **Modular Architecture**: Clean separation of concerns between API, infrastructure, document processing, and orchestration.
* **Smart Conversation Memory**: Automatically maintains context windows and supports conversation summarization for long-running chats.
* **Multi-Format Support**: Ingest and process PDF and Text files.
* **Dual Interface**: Access the system via a modern **Gradio UI** or a production-ready **FastAPI** REST interface.

## 🏗️ Project Structure

```text
rag-research-assistant/
├── src/
│   ├── api/             # FastAPI endpoints, routers, and Pydantic schemas
│   ├── config/          # Centralized app settings and directory management
│   ├── conversation/    # Message history, context windowing, and session management
│   ├── documents/       # File loading, text chunking, and hashing
│   ├── infra/           # Ollama LLM integration, ChromaDB, and Embedding functions
│   └── orchestrator/    # Business logic orchestration
├── ui/
│   └── gradio_app.py    # Gradio web interface with Knowledge Base management
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation

```

## 📋 Prerequisites

* **Python 3.11+**
* **Ollama**: Must be installed and running on your system.
* **Model**: The default configuration uses `llama3.2`.

## 🚀 Setup and Installation

### 1. Clone & Install

```bash
git clone https://github.com/dsbarik/rag-research-assistant
cd rag-research-assistant
pip install -r requirements.txt

```

### 2. Prepare the LLM

Ensure Ollama is running and pull the required model:

```bash
ollama pull llama3.2

```

### 3. Configuration

Application paths and model names are managed in `src/config/settings.py`. By default, it creates a `data/` directory in the project root for your database and vector store.

## 🛠️ Usage

Since the services are decoupled, you must start the backend API and the frontend UI in separate terminal sessions.

### 1. Start the FastAPI Backend

The API handles document ingestion, vector search, and LLM orchestration. Run it in development mode for hot-reloading:

```bash
fastapi dev src/api/main.py

```

* **Interactive API Docs**: `http://localhost:8000/docs`

### 2. Start the Gradio Web UI

The UI provides a user-friendly interface for chatting and managing the Knowledge Base.

```bash
python ui/gradio_app.py

```

* **Web Interface**: `http://localhost:7860`

## 🔌 API Usage Example

**Chat with Context:**

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "What are the main findings of the document?", "conversation_id": null}'

```

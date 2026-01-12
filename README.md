# RAG Research Assistant

RAG Research Assistant is a Retrieval-Augmented Generation (RAG) system designed to help users interact with their documents using local Large Language Models (LLMs). It features a modular architecture with a FastAPI backend and a Gradio-based user interface.

## Features

- **Local LLM Integration**: Uses Ollama to run models like `llama3` locally for privacy and performance.
- **Vector Database**: Utilizes ChromaDB for efficient document storage and retrieval.
- **Modular Architecture**: Separates concerns into API, infrastructure, document processing, and conversation management.
- **Dual Interface**: Supports both a RESTful API and a user-friendly Gradio web interface.
- **Document Management**: Handles document loading, chunking, and embedding generation using Sentence Transformers.

## Project Structure (Current Implemetentation)

```text
rag-research-assistant/
├── run.py                 # Main entry point to launch API and UI
├── src/
│   ├── api/               # FastAPI implementation
│   ├── config/             # Application settings and environment variables
│   ├── conversation/      # Context and message management
│   ├── documents/         # Document loading and processing logic
│   ├── infra/             # Infrastructure: DB, LLM (Ollama), Vectorstore
│   └── orchestrator/      # Business logic orchestration
└── ui/
    └── gradio_app.py      # Gradio web interface

```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) (Required for local LLM inference)
- SQLite (for conversation history and metadata)
- PyMuPdf (for extracting text from pdf files)
- Chroma (for vector database)
- FastAPI (for API backend)
- Gradio (for UI)

## Setup and Installation

1. **Clone the Repository**:

```bash
git clone https://github.com/darkdevil18/rag-research-assistant
cd rag-research-assistant
```

2.**Environment Configuration**:
Configure your environment variables in `src/config/settings.py`:

- `OLLAMA_BASE_URL`: Defaults to `http://localhost:11434`.
- `MODEL_NAME`: Defaults to `llama3`.
- `DATA_DIR`: Directory for storing uploaded files and vector data.

3.**Ensure Ollama is Running**:
Make sure Ollama is installed and the model is pulled:

```bash
ollama pull llama3
```

## Usage

You can run the entire application (both the API and the UI) using the provided entry point script:

```bash
python run.py
```

- **API**: The FastAPI backend will typically start on `http://0.0.0.0:8000`.
- **Web UI**: The Gradio interface will launch, allowing you to upload documents and chat with the RAG system.

## Core Components

- **Document Processor**: Splits documents into manageable chunks for the vector database.
- **Embedding Service**: Uses `sentence-transformers` to convert text chunks into vector embeddings.
- **LLM Service**: Connects to an `Ollama` instance to generate responses based on retrieved context.
- **Orchestrator**: Coordinates between the vector store, the LLM, and the database to manage the RAG workflow.

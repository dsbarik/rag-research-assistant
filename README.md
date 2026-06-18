# RAG Research Assistant

A privacy-centric, locally deployed Retrieval-Augmented Generation (RAG) system designed for secure and intelligent conversations with personal or organizational documents.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/) [![Ollama](https://img.shields.io/badge/Ollama-LLM-orange.svg)](https://ollama.ai/) [![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-blueviolet.svg)](https://www.trychroma.com/) [![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg)](https://gradio.app/) [![Streamlit](https://img.shields.io/badge/Streamlit-UI-ff4b4b.svg)](https://streamlit.io/) [![Pytest](https://img.shields.io/badge/Pytest-Tests-green.svg)](https://docs.pytest.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

The RAG Research Assistant provides a secure framework for context-aware conversations with documents using state-of-the-art retrieval and generation techniques. The system is designed for complete local execution, ensuring that no data leaves the host environment.

### Core Concepts: Retrieval-Augmented Generation

The system integrates three primary components to ensure accuracy and relevance:

- **Vector Search**: Employs semantic similarity to retrieve the most relevant document segments.
- **LLM Generation**: Produces grounded responses based on the retrieved context.
- **Conversation Memory**: Manages state and context across multi-turn interactions.

---

## Key Features

| Feature | Description |
| --------- | ------------- |
| **Data Privacy** | All processing is performed locally without the use of cloud APIs. |
| **Production Architecture** | FastAPI backend with asynchronous support and robust error handling. |
| **Flexible Interfaces** | Support for both Gradio and Streamlit user interfaces. |
| **Multi-Format Ingestion** | Support for PDF and text files with intelligent chunking strategies. |
| **Context Management** | Automatic conversation windowing and summarization. |
| **Persistent Storage** | Integrated ChromaDB vector store and SQLite metadata management. |
| **Verified Quality** | Comprehensive test suite implemented with pytest. |

---

## Architecture

### System Diagram

```bash
┌─────────────────────────────────────┐
│  Gradio / Streamlit UI / REST API     │
│   (User Interface Layer)              │
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
│   ├── gradio_app.py    # Gradio-based UI
│   └── streamlit_app.py # Streamlit-based UI
├── tests/               # Pytest test suite
├── data/                # Local storage for databases and vectors
├── requirements.txt
└── README.md
```

---

## Prerequisites

The following components are required for installation and operation:

- **Python 3.11+**: [Download](https://www.python.org/downloads/)
- **Ollama**: Installed and running. [Installation Guide](https://ollama.ai/download)
- **Git**: For repository cloning.

### Verify Ollama Service

```bash
# Check if the Ollama service is responsive
curl http://localhost:11434/api/tags
```

---

## Installation and Setup

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/dsbarik/rag-research-assistant.git
cd rag-research-assistant

# Initialize virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Deployment

```bash
# Pull the required LLM model
ollama pull llama3.2

# Verify model availability
ollama list
```

### 3. Service Execution

**Backend API:**
```bash
fastapi dev src/api/main.py
```
- API Endpoint: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

**Gradio Interface:**
```bash
python ui/gradio_app.py
```
- Interface URL: `http://localhost:7860`

**Streamlit Interface:**
```bash
streamlit run ui/streamlit_app.py
```
- Interface URL: `http://localhost:8501`

---

## Usage Guide

### Document Management
- **Ingestion**: Upload PDF or text files via the Knowledge Base tab. The system processes these into semantic chunks for the vector store.
- **Administration**: View uploaded documents and remove unnecessary files through the management interface.

### Conversational Interface
- **Interaction**: Use the chat interface to query the ingested knowledge base.
- **Grounding**: The system retrieves relevant context and generates responses grounded in the provided documents.
- **Session Management**: Use the "New Chat" feature to reset conversation history.

### REST API Reference

#### Document Ingestion
`POST /api/v1/ingest`
```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf"
```

#### Contextual Chat
`POST /api/v1/chat`
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the main findings?",
    "conversation_id": null
  }'
```

#### Document Metadata
`GET /api/v1/documents`
`DELETE /api/v1/documents/{id}`

---

## Configuration

### Environment Variables

Optional configuration can be provided via a `.env` file in the project root:

```env
# LLM Configuration
LLM_MODEL_NAME=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

# Storage Paths
DATA_DIR=./data
DOCUMENTS_DIR=./data/documents
VECTOR_STORE_DIR=./data/vector_store
DATABASE_PATH=./data/documents.db
```

### Model Selection

To change the default LLM, modify `src/config/settings.py`:

```python
LLM_MODEL_NAME = "llama3.2"  # Supports any model available via Ollama
```

---

## Testing

```bash
# Execute full test suite
pytest

# Execute tests with coverage reporting
pytest --cov=src tests/

# Execute specific module tests
pytest tests/test_orchestrator.py -v
```

---

## Troubleshooting

| Issue | Resolution |
| --- | --- |
| **Connection Refused** | Ensure the Ollama service is running via `ollama serve`. |
| **Missing Dependencies** | Run `pip install sentence-transformers` if the embedding module is missing. |
| **Ingestion Failure** | Verify file format (PDF/TXT), check for duplicates, or inspect API logs. |
| **Latency** | Consider a smaller model (e.g., `llama3.2:1b`) or optimize the chunk retrieval limit in `orchestrator/service.py`. |

---

## Development Roadmap

The project is actively evolving with the following planned enhancements:

- **Multi-Agent Orchestration**: Specialized agents for distinct research tasks.
- **Knowledge Graph Integration**: Entity extraction and graph-based verification.
- **Advanced Retrieval**: Implementation of hybrid search and re-ranking.
- **Explainability Layer**: Visualizations of the retrieval and generation pipeline.
- **Expanded Format Support**: Integration of DOCX, Markdown, and HTML.
- **Real-time Streaming**: Token streaming for the user interfaces.
- **Multi-tenancy**: User authentication and isolated document namespaces.

---

## Contributing

Contributions are welcome. Please follow these steps:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature-name`.
3. Commit changes: `git commit -m 'Add descriptive commit message'`.
4. Push to the branch and open a Pull Request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Ollama**: Local LLM inference engine.
- **ChromaDB**: Vector database for semantic storage.
- **Sentence Transformers**: High-quality embedding models.
- **FastAPI**: High-performance web framework.
- **Gradio & Streamlit**: Rapid UI prototyping frameworks.

---

## Contact and Support

- **Email**: <barikdibyasampad@gmail.com>
- **Issues**: [GitHub Issues](https://github.com/dsbarik/rag-research-assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dsbarik/rag-research-assistant/discussions)

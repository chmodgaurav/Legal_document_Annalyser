# Legal Document Analyzer

An AI-powered legal document analysis tool built with Streamlit, LangChain, and Ollama. Upload PDFs, generate summaries, semantically search across your document store, and ask questions with source-grounded answers — all running on local LLMs.

## Features

- **Document ingestion** — load PDFs (and text/DOCX via LangChain loaders) and split them into overlapping chunks for retrieval
- **Vector search** — persistent Chroma database with Ollama embeddings for semantic similarity search
- **Summarization** — generate concise summaries of uploaded documents
- **Q&A with threshold gating** — questions are answered via retrieval-augmented generation only when similarity exceeds a configurable threshold, falling back otherwise
- **Fully local** — Ollama serves both the embedding model and the chat model, so no documents leave your machine

## Architecture

```
PDF/TXT/DOCX
     │
     ▼
Document Loader (PyPDFLoader / TextLoader / Docx2txtLoader)
     │
     ▼
RecursiveCharacterTextSplitter (chunk_size=1000, overlap=150)
     │
     ▼
OllamaEmbeddings (nomic-embed-text) → Chroma vector store (./database)
     │
     ▼
Query → similarity_search_with_score → ChatOllama (llama3:8b) → answer
```

## Project Structure

| Path | Purpose |
|---|---|
| `streamlit_app.py` | Main Streamlit application (upload, summarize, search, ask) |
| `chroma.py` | Batch ingestion script — loads `./data`, splits, embeds, persists to `./database` |
| `main.py` | Minimal CLI query script against the existing vector store |
| `utils.py` | Directory helpers and document-info extraction utilities |
| `requirements.txt` | Python dependencies |
| `data/` | PDF files to ingest |
| `database/` | Persistent Chroma vector store (created on first run) |

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally

## Installation

```bash
git clone https://github.com/chmodgaurav/Legal_document_Annalyser.git
cd Legal_document_Annalyser
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Pull the required Ollama models

```bash
ollama pull nomic-embed-text:latest
ollama pull llama3:8b
```

## Usage

### 1. Start Ollama

```bash
ollama serve
```

Runs on `http://localhost:11434` by default.

### 2. (Optional) Bulk-ingest documents

Place PDFs in `./data`, then:

```bash
python chroma.py
```

This builds/updates the Chroma vector store at `./database`.

### 3. Run the app

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` to upload documents from the UI, generate summaries, search, and ask questions directly — the app can also ingest new PDFs without running `chroma.py` separately.

### CLI query (optional)

```bash
python main.py
```

Prompts for a query, runs a similarity search against `./database`, and answers via RAG when the similarity score clears the configured threshold (0.70 by default).

## Configuration

Key settings currently live inline in `streamlit_app.py`:

- `PERSIST_DIRECTORY` — Chroma database path (default `./database`)
- `SIMILARITY_THRESHOLD` — minimum similarity score to trust retrieved context (default `0.70`)
- Embedding model — `nomic-embed-text:latest`
- Chat model — `llama3:8b`

Adjust these constants directly, or set an Ollama model via the `OllamaEmbeddings`/`ChatOllama` calls.

## Notes

- Ensure `ollama serve` is running before starting the Streamlit app — embedding and chat calls will fail otherwise.
- First-time ingestion of large PDFs can take a while since embeddings are computed locally.

## License

See repository for license details.

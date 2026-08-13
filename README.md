# Legal Document Analyzer

A Streamlit RAG (retrieval-augmented generation) application for legal documents. Upload PDFs, DOCX, or TXT files, index them into Pinecone, and chat with an LLM that answers strictly from retrieved context, citing the source file for every claim.

## Features

- **Multi-format ingestion** — accepts PDF, DOCX, and TXT uploads directly in the Streamlit UI
- **Deterministic chunking** — `RecursiveCharacterTextSplitter` (chunk size 1200, overlap 200) with hash-based chunk IDs, so re-uploading the same file updates existing vectors instead of duplicating them
- **Vector search** — Pinecone index (`legal-documents`) queried via `similarity_search_with_score`, with an adjustable retrieval size (`k`, 1–10) in the sidebar
- **Grounded Q&A** — the chat model is instructed to answer only from retrieved context and to say so explicitly when the answer isn't supported, citing filenames and similarity scores
- **Conversation memory** — the last 6 messages are folded into the prompt so follow-up questions retain context
- **Source inspection** — every assistant reply includes an expandable panel showing the retrieved chunks, their source file, and similarity score
- **Batch ingestion script** — `Pinecone.py` bulk-loads and indexes everything in `./data` outside the UI

## Tech Stack

- **UI:** Streamlit
- **Orchestration:** LangChain (`langchain-core`, `langchain-community`, `langchain-text-splitters`, `langchain-pinecone`)
- **Vector store:** Pinecone
- **Embeddings:** OpenRouter (`nvidia/llama-nemotron-embed-vl-1b-v2:free`, 1024 dimensions) via a custom `Embeddings` wrapper around the OpenAI-compatible client
- **Chat model:** OpenRouter (`google/gemini-2.5-flash` by default)
- **Document parsing:** `pypdf`, `python-docx`, `pymupdf`

## Architecture

```
PDF / DOCX / TXT (Streamlit upload)
        │
        ▼
extract_text()  →  pypdf / python-docx / raw decode
        │
        ▼
RecursiveCharacterTextSplitter (chunk_size=1200, overlap=200)
        │
        ▼
OpenRouterEmbeddings (nvidia/llama-nemotron-embed-vl-1b-v2:free)
        │
        ▼
Pinecone index "legal-documents"  (deterministic IDs: file_hash + chunk_hash)
        │
        ▼
Query → similarity_search_with_score(k) → prompt with context + chat history
        │
        ▼
OpenRouter chat completion (google/gemini-2.5-flash) → grounded answer + sources
```

## Installation

```bash
git clone https://github.com/chmodgaurav/Legal_document_Annalyser.git
cd Legal_document_Annalyser
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
PINECONE_API_KEY=your_pinecone_api_key

# Optional overrides (defaults shown)
PINECONE_INDEX_NAME=legal-documents
EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free
EMBEDDING_DIMENSIONS=1024
CHAT_MODEL=google/gemini-2.5-flash
```

Both `OPENROUTER_API_KEY` and `PINECONE_API_KEY` are required — the app calls `st.stop()` on launch if either is missing. The Pinecone index must already exist (dimension must match `EMBEDDING_DIMENSIONS`) before running the app.

## Usage

### Run the app

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501`. Upload documents from the sidebar, click **Process Documents** to embed and index them, then ask questions in the chat input. Each answer shows an expandable **View retrieved sources** panel with file names, similarity scores, and matched content.

### Bulk ingest from `./data` (optional)

```bash
python Pinecone.py
```

Loads every PDF under `./data`, splits it, embeds it, and indexes it into Pinecone — useful for seeding the index outside the UI. Two sample PDFs are included under `data/`.

### CLI query script (optional)

```bash
python main.py
```

A minimal terminal loop for querying the existing Pinecone index directly, without the Streamlit UI.

## Project Structure

| Path | Purpose |
|---|---|
| `streamlit_app.py` | Main application — upload, ingest, chat, and source inspection |
| `Pinecone.py` | Batch ingestion script — loads `./data`, splits, embeds, and indexes into Pinecone |
| `main.py` | Minimal CLI query loop against the existing Pinecone index |
| `utils.py` | Local filesystem helpers (data/database directory management, document-info extraction) — not currently wired into `streamlit_app.py`'s Pinecone flow |
| `requirements.txt` | Python dependencies |
| `data/` | Sample legal PDFs for ingestion |

## Notes

- `utils.py` contains helpers written for a local Chroma-based setup (directory stats, database cleanup) that aren't called by the current Pinecone-backed `streamlit_app.py` — kept in the repo but currently unused by the main flow.
- Retrieved-context size is capped in the CLI script (`main.py`) at 6000 characters to control prompt size; the Streamlit app instead limits result count via the `k` slider.

## Future Improvements

- Wire `utils.py`'s database/stat helpers into the Streamlit UI, or remove if superseded by Pinecone
- Add automated tests (`pytest`/`pytest-cov` are already in `requirements.txt` but no test suite exists yet)
- Support additional document loaders (e.g. scanned PDFs via OCR)

## Screenshots

_Add screenshots of the chat interface and source-inspection panel here._

## Contributing

Issues and pull requests are welcome. Please open an issue describing the change before submitting a large PR.

## License

See repository for license details.
# ⚖️ Legal Document Analyzer

A powerful AI-powered legal document analysis tool built with Streamlit, LangChain, and Ollama. Upload PDFs, get instant summaries, search through documents, and ask intelligent questions about your legal documents.

## Features

✨ **Document Summarization** - Automatically generate concise summaries of legal documents using advanced LLMs

🔍 **Intelligent Search** - Search through your document database using semantic similarity

💡 **AI Assistant** - Ask questions about your documents and get context-aware answers

📁 **PDF Upload** - Easily upload new PDF documents to expand your database

🗂️ **Vector Database** - Persistent storage using Chroma for fast retrieval

🤖 **Ollama Integration** - Uses local LLMs for privacy and offline functionality

### Install Ollama Models

```bash
ollama pull nomic-embed-text:latest
ollama pull mistral:latest
ollama pull llama2:latest
ollama pull neural-chat:latest
```

## Installation

1. **Clone the repository**
   ```bash
   cd /home/neo/Documents/Legal_document_Annalyser
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Start Ollama Server
```bash
ollama serve
```
This should run on `http://localhost:11434`

### 2. Run the Streamlit App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Upload & Summarize Documents
1. Click **"📁 Document Management"** in the sidebar
2. Upload a PDF file
3. Click **"📤 Process & Add to Database"**
4. Go to the **"📄 Summarize"** tab
5. Click **"Generate Summary"** to create an AI summary

### Search Documents
1. Go to the **"🔍 Query"** tab
2. Enter your search query (e.g., "What are the payment terms?")
3. Adjust the number of results using the slider
4. Click **"🔍 Search"**
5. Review the relevant document sections

### Ask Questions (AI Assistant)
1. Go to the **"💡 Ask"** tab
2. Type your question about the documents
3. Click **"💭 Get Answer"**
4. Get AI-powered answers with source citations

## Project Structure

```
Legal_document_Annalyser/
├── app.py                 # Main Streamlit application
├── main.py                # Alternative query interface (CLI)
├── chroma.py              # Vector database initialization
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── data/                  # Uploaded PDF files
├── database/              # Chroma vector database
└── README.md              # This file
```

## Configuration

Edit `config.py` to customize:

- **Models**: Change LLM or embedding models
- **Chunk Size**: Adjust document splitting parameters
- **Top K Results**: Number of search results to return
- **Custom Prompts**: Modify AI prompts for different use cases

## Available LLM Models

You can switch models in the Streamlit sidebar:

- **Mistral** (Recommended) - Fast and accurate
- **Llama 2** - Good for legal documents
- **Neural Chat** - Conversation-focused
- **OpenHermes** - Creative responses

## File Management

### Data Directory (`./data`)
- Upload new PDFs here
- PDFs are automatically saved when processed through the app

### Database Directory (`./database`)
- Contains the vector embeddings
- Persistent storage of processed documents
- Safe to delete (will be regenerated when new docs are added)

## Troubleshooting

### "Error loading embeddings" or "Error loading LLM"
- Make sure Ollama is running: `ollama serve`
- Check if models are installed: `ollama list`
- Install missing models: `ollama pull model-name`

### App is slow
- Vector database is large: Consider removing old documents from `./database`
- LLM is processing: Wait for the response, LLMs can take time

### PDF not uploading
- Check file size (large PDFs may timeout)
- Ensure file is a valid PDF
- Try with a smaller PDF first

### "No documents in database"
- Upload at least one PDF using the sidebar uploader
- Or manually place PDFs in `./data` and run `chroma.py`

## Manual Database Initialization

If you have PDFs in the `./data` folder, initialize the database:

```bash
python chroma.py
```

Then run:
```bash
streamlit run app.py
```


## Limitations & Disclaimers

⚠️ **Important**: This tool is for informational and analytical purposes only. Always consult a qualified legal professional for important legal decisions.

- LLM outputs should be reviewed by humans
- Not a substitute for professional legal advice
- Document privacy: Run locally for sensitive documents
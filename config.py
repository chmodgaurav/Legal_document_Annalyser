"""Configuration file for Legal Document Analyzer"""

# Ollama Models Configuration
EMBEDDINGS_MODEL = "nomic-embed-text:latest"
DEFAULT_LLM_MODEL = "mistral:latest"

# LLM Models available
AVAILABLE_LLM_MODELS = [
    "mistral:latest",
    "llama2:latest", 
    "neural-chat:latest",
    "openhermes:latest"
]

# Document Processing Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# RAG Settings
TOP_K_RESULTS = 5

# Paths
DATA_DIRECTORY = "./data"
DATABASE_DIRECTORY = "./database"

# App Settings
APP_TITLE = "⚖️ Legal Document Analyzer"
APP_ICON = "⚖️"

# LLM Prompts
SUMMARIZATION_PROMPT = """You are a legal document analyzer. Provide a concise and clear summary of the following legal document. 
Focus on key terms, obligations, dates, and important clauses.

Document:
{document_text}

Summary:"""

QUERY_PROMPT = """You are an expert legal assistant. Based on the provided document excerpts, answer the following question clearly and accurately.
If the information is not in the documents, say so.

Document Context:
{context}

Question: {question}

Answer:"""

# Error Messages
ERROR_EMBEDDINGS = "Error loading embeddings model. Make sure Ollama is running."
ERROR_LLM = "Error loading LLM model. Make sure Ollama is running."
ERROR_DATABASE = "Error initializing vector database."
ERROR_PDF_PROCESSING = "Error processing PDF file."

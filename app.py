import streamlit as st
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
import tempfile
import shutil

# Page configuration
st.set_page_config(
    page_title="Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 0rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.1em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "llm" not in st.session_state:
    st.session_state.llm = None

@st.cache_resource
def load_embeddings():
    """Load embeddings model"""
    try:
        return OllamaEmbeddings(model="nomic-embed-text:latest")
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        return None

@st.cache_resource
def load_llm():
    """Load LLM model"""
    try:
        return OllamaLLM(model="llama3:8b")
    except Exception as e:
        st.error(f"Error loading LLM: {e}")
        return None

def initialize_vector_db():
    """Initialize or load existing vector database"""
    embeddings = load_embeddings()
    if embeddings is None:
        return None
    
    db_path = "./database"
    try:
        db = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
        return db
    except Exception as e:
        st.error(f"Error initializing vector database: {e}")
        return None

def process_pdf(pdf_file):
    """Process uploaded PDF file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name
    
    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def add_documents_to_db(documents, file_name):
    """Add documents to vector database"""
    embeddings = load_embeddings()
    if embeddings is None:
        st.error("Embeddings not available")
        return False
    
    try:
        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        split_docs = splitter.split_documents(documents)
        
        # Add metadata
        for doc in split_docs:
            doc.metadata["source_file"] = file_name
        
        db_path = "./database"
        
        # Create or add to existing database
        db = Chroma.from_documents(
            split_docs,
            embeddings,
            persist_directory=db_path,
        )
        
        st.session_state.vector_db = db
        return True
    except Exception as e:
        st.error(f"Error adding documents to database: {e}")
        return False

def summarize_document(text, llm_model=None):
    """Summarize document using LLM"""
    if llm_model is None:
        llm_model = load_llm()
    
    if llm_model is None:
        return "Error: LLM not available"
    
    try:
        prompt_template = ChatPromptTemplate.from_template(
            """You are a legal document analyzer. Provide a concise and clear summary of the following legal document.
            
Document:
{document_text}

Summary:"""
        )
        
        chain = prompt_template | llm_model
        response = chain.invoke({"document_text": text[:4000]})  # Limit input length
        return response
    except Exception as e:
        return f"Error generating summary: {e}"

def query_documents(query_text, db, k=5):
    """Query documents from vector database"""
    try:
        results = db.similarity_search(query_text, k=k)
        return results
    except Exception as e:
        st.error(f"Error querying documents: {e}")
        return []

# Header
st.markdown("# ⚖️ Legal Document Analyzer")
st.markdown("---")

# Sidebar for PDF upload and configuration
with st.sidebar:
    st.header("📁 Document Management")
    
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type="pdf",
        help="Upload a legal document (PDF format)"
    )
    
    if uploaded_file:
        st.info(f"Selected file: {uploaded_file.name}")
        
        if st.button("📤 Process & Add to Database", key="process_pdf"):
            with st.spinner("Processing PDF..."):
                documents = process_pdf(uploaded_file)
                if documents:
                    st.success(f"Loaded {len(documents)} pages")
                    
                    if add_documents_to_db(documents, uploaded_file.name):
                        st.success(f"✅ {uploaded_file.name} added to database!")
                        # Save to data folder
                        data_folder = Path("./data")
                        data_folder.mkdir(exist_ok=True)
                        with open(data_folder / uploaded_file.name, "wb") as f:
                            f.write(uploaded_file.getbuffer())
    
    st.divider()
    
    # Model configuration
    st.header("⚙️ Configuration")
    
    model_choice = st.selectbox(
        "LLM Model",
        ["mistral:latest", "llama3:8b", "neural-chat:latest"],
        help="Select the LLM model to use"
    )
    
    chunk_size = st.slider(
        "Chunk Size",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100,
        help="Size of text chunks for processing"
    )

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["📄 Summarize", "🔍 Query", "💡 Ask"])

# Tab 1: Summarize
with tab1:
    st.header("Document Summary")
    st.markdown("Upload a PDF or select from database to generate a summary")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if uploaded_file:
            st.subheader(f"Summarizing: {uploaded_file.name}")
            
            if st.button("Generate Summary", key="gen_summary"):
                with st.spinner("Generating summary..."):
                    documents = process_pdf(uploaded_file)
                    if documents:
                        # Combine document text
                        full_text = "\n".join([doc.page_content for doc in documents])
                        summary = summarize_document(full_text, load_llm())
                        
                        st.subheader("Summary:")
                        st.write(summary)
                        
                        # Download summary
                        st.download_button(
                            label="📥 Download Summary",
                            data=summary,
                            file_name=f"summary_{uploaded_file.name.replace('.pdf', '.txt')}",
                            mime="text/plain"
                        )
        else:
            st.info("👆 Upload a PDF document to generate a summary")

# Tab 2: Query Database
with tab2:
    st.header("Query Documents")
    st.markdown("Search through your document database")
    
    # Initialize database if not done
    if st.session_state.vector_db is None:
        st.session_state.vector_db = initialize_vector_db()
    
    if st.session_state.vector_db is not None:
        query_input = st.text_input(
            "Enter your query:",
            placeholder="e.g., What are the payment terms?"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            k = st.slider("Number of results", min_value=1, max_value=10, value=5)
        
        if query_input:
            if st.button("🔍 Search", key="search_btn"):
                with st.spinner("Searching..."):
                    results = query_documents(
                        query_input,
                        st.session_state.vector_db,
                        k=k
                    )
                    
                    if results:
                        st.success(f"Found {len(results)} relevant sections")
                        
                        for i, result in enumerate(results, 1):
                            with st.expander(f"Result {i} - {result.metadata.get('source_file', 'Unknown')}"):
                                st.write(result.page_content)
                                st.caption(f"Page: {result.metadata.get('page', 'N/A')}")
                    else:
                        st.warning("No results found")
    else:
        st.warning("⚠️ No documents in database. Upload a PDF first!")

# Tab 3: AI Assistant
with tab3:
    st.header("Legal Assistant")
    st.markdown("Ask questions about your documents with AI-powered responses")
    
    # Initialize database if not done
    if st.session_state.vector_db is None:
        st.session_state.vector_db = initialize_vector_db()
    
    if st.session_state.vector_db is not None:
        question = st.text_area(
            "Ask a question:",
            placeholder="e.g., What are the main obligations of each party?",
            height=100
        )
        
        if st.button("💭 Get Answer", key="ask_btn"):
            if question:
                with st.spinner("Searching and analyzing..."):
                    # Search for relevant documents
                    relevant_docs = query_documents(question, st.session_state.vector_db, k=3)
                    
                    if relevant_docs:
                        # Combine context
                        context = "\n\n".join([doc.page_content for doc in relevant_docs])
                        
                        # Create prompt with context
                        prompt_template = ChatPromptTemplate.from_template(
                            """You are an expert legal assistant. Based on the provided document excerpts, answer the following question clearly and accurately.

Document Context:
{context}

Question: {question}

Answer:"""
                        )
                        
                        llm = load_llm()
                        chain = prompt_template | llm
                        
                        response = chain.invoke({
                            "context": context,
                            "question": question
                        })
                        
                        st.subheader("Answer:")
                        st.write(response)
                        
                        with st.expander("📚 Source Documents"):
                            for i, doc in enumerate(relevant_docs, 1):
                                st.caption(f"Source {i}: {doc.metadata.get('source_file', 'Unknown')}")
                                st.text(doc.page_content[:500] + "...")
                    else:
                        st.warning("No relevant documents found to answer your question.")
            else:
                st.warning("Please enter a question")
    else:
        st.warning("⚠️ No documents in database. Upload a PDF first!")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; margin-top: 30px;">
    <p>Legal Document Analyzer v1.0 | Powered by LangChain & Ollama</p>
    <p>⚠️ Disclaimer: This tool is for informational purposes. Consult a legal professional for important decisions.</p>
</div>
""", unsafe_allow_html=True)

import tempfile
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings,
)
import os
load_dotenv()

PERSIST_DIRECTORY = "./database"
SIMILARITY_THRESHOLD = 0.70

embeddings = MistralAIEmbeddings(
    model="mistral-embed"
    api_key=os.getenv("MISTRAL_API_KEY")
)

llm = ChatMistralAI(
    model="mistral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY")
    temperature=0,
)
# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Legal Document Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Legal Document Assistant")
st.write("Upload a legal document from the sidebar, generate a summary, or ask questions about it.")


# --------------------------------------------------
# Session State
# --------------------------------------------------
if "db" not in st.session_state:
    st.session_state.db = None

if "documents" not in st.session_state:
    st.session_state.documents = None


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def load_document(file_path):
    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        loader = PyPDFLoader(file_path)

    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)

    elif extension == ".txt":
        loader = TextLoader(file_path)

    else:
        raise ValueError("Unsupported file type.")

    return loader.load()


def build_vectorstore(uploaded_file):
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=Path(uploaded_file.name).suffix,
    ) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name

    documents = load_document(temp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(documents)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )

    return db, documents


def summarize_document(documents):

    text = "\n\n".join(doc.page_content for doc in documents)

    prompt = f"""
'''You are an expert Legal Document Analysis AI specializing in reviewing, interpreting, and summarizing legal documents.

Your responsibilities include:
- Analyzing contracts, agreements, policies, legal notices, regulations, licenses, NDAs, employment agreements, court filings, and other legal documents.
- Answering questions ONLY using the provided document context.
- Identifying relevant clauses and explaining them in plain English.
- Producing accurate, objective, and well-structured responses.

Strict Rules

1. Use ONLY the information contained in the provided document context.

2. Never invent:
   - clauses
   - obligations
   - dates
   - legal definitions
   - parties
   - penalties
   - rights
   - legal conclusions

3. If the answer cannot be found in the provided context, respond exactly:

   "The provided document does not contain enough information to answer this question."

4. Never claim something exists unless it appears in the document.

5. Quote the relevant portion of the document whenever possible before explaining it.

6. Distinguish clearly between:
   - Direct document content
   - Interpretation
   - General legal knowledge

7. If legal knowledge outside the document would be required, state:

   "This would require legal interpretation beyond the provided document."

8. Never provide legal advice.

Instead say:

   "This is an informational analysis of the document and should not be considered legal advice."

Response Style

Be concise, precise, and professional.

Use markdown headings when appropriate.

For every answer follow this structure whenever possible:

### Answer
Direct answer.

### Supporting Evidence
Quote or summarize the relevant document section.

### Explanation
Explain the clause in plain language.

### Confidence
High / Medium / Low

Confidence Rules

High:
- Explicitly stated in the document.

Medium:
- Inferred from multiple clauses.

Low:
- Ambiguous wording or incomplete context.

When asked to summarize a document, include:

# Document Summary

## Purpose

## Parties Involved

## Effective Date

## Term / Duration

## Key Obligations

## Rights

## Payment Terms

## Confidentiality

## Intellectual Property

## Liability

## Indemnification

## Termination

## Governing Law

## Important Deadlines

## Risks and Unusual Clauses

## Missing Information

When comparing two documents:

- List similarities.
- List differences.
- Highlight conflicting clauses.
- Identify clauses present in one but absent in the other.

When identifying risks:

Categorize each risk as:
- High
- Medium
- Low

Provide:
- Risk
- Relevant Clause
- Reason
- Potential Impact

Formatting Rules

- Use bullet lists where appropriate.
- Keep explanations simple.
- Do not speculate.
- Do not hallucinate.
- If uncertain, explicitly say so.

Remember:
Accuracy is more important than completeness.
Never answer beyond what is supported by the provided document.''
Document:
{text}
"""

    return llm.invoke(prompt).content


def ask_question(db, query):

    results = db.similarity_search_with_score(query, k=5)

    if not results:
        return llm.invoke(query).content

    best_doc, best_distance = results[0]
    best_similarity = 1 - best_distance

    context = "\n\n".join(
        doc.page_content for doc, _ in results
    )

    # -------------------------
    # High Similarity
    # -------------------------
    if best_similarity >= SIMILARITY_THRESHOLD:

        prompt = f"""
Context:
{context}

Question:
{query}
"""

        return llm.invoke(prompt).content

    # -------------------------
    # Low Similarity
    # -------------------------
    rag_prompt = f"""
You must answer ONLY using the context below.

If the answer cannot be found, reply exactly:

The provided document does not contain enough information to answer this question.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(rag_prompt).content

    if (
        "The provided document does not contain enough information"
        in response
    ):
        return llm.invoke(query).content

    return response


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:

    st.header("📂 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF, DOCX, or TXT file",
        type=["pdf", "docx", "txt"],
    )

    if uploaded_file:

        with st.spinner("Processing document..."):

            db, docs = build_vectorstore(uploaded_file)

            st.session_state.db = db
            st.session_state.documents = docs

        st.success("✅ Document indexed successfully.")


# --------------------------------------------------
# Main Page
# --------------------------------------------------

# Summary
st.header("📑 Document Summary")

if st.session_state.documents:

    if st.button("Generate Summary"):

        with st.spinner("Generating summary..."):

            summary = summarize_document(
                st.session_state.documents
            )

        st.markdown(summary)

else:
    st.info("Upload a document from the sidebar.")


st.divider()

# Question Answering
st.header("💬 Ask Questions")

question = st.text_input(
    "Enter your question about the document"
)

if st.button("Ask"):

    if st.session_state.db is None:

        st.warning("Please upload a document first.")

    elif not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching document..."):

            answer = ask_question(
                st.session_state.db,
                question,
            )

        st.subheader("Answer")
        st.write(answer)
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

doc=DirectoryLoader("./data", glob="**/*.PDF" or "**/*.pdf", loader_cls=PyPDFLoader) #type:ignore
loader=doc.load()

splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
split_docs=splitter.split_documents(loader)

embeddings=OllamaEmbeddings(model="nomic-embed-text:latest")

vector_store = Chroma.from_documents(
    split_docs,
    embeddings,
    persist_directory="./database",
)
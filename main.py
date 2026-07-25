from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

db = Chroma(
    persist_directory="./database",
    embedding_function=embeddings
)

query=input("Enter your query: ")

results = db.similarity_search(
    query,
    k=5
)

for doc in results:
    print(doc.page_content)
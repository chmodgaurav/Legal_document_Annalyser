from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
llm=ChatOllama(
    model="llama3:8b",
    system_prompt='            
)


db = Chroma(
    persist_directory="./database",
    embedding_function=embeddings
)

query=input("Enter your query: ")

results = db.similarity_search_with_score(query, k=1)

doc, distance = results[0]

# If using cosine distance
similarity = 1 - distance

SIMILARITY_THRESHOLD = 0.70

if similarity >= SIMILARITY_THRESHOLD:
    # Use RAG
    for doc in results:
        print(doc.page_content)

else:
    # No relevant document. Ask the LLM directly.
    context = doc.page_content
    prompt = f"""
    Context:
    {context}
    
    Question:
    {query}
    """
    response = llm.invoke(prompt)
    print(response.content)
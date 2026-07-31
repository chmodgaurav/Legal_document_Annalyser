from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

llm = ChatOllama(
    model="llama3:8b",
    system_prompt="""Your existing system prompt here"""
)

db = Chroma(
    persist_directory="./database",
    embedding_function=embeddings
)

query = input("Enter your query: ")

SIMILARITY_THRESHOLD = 0.70

# Retrieve multiple chunks
results = db.similarity_search_with_score(query, k=5)

best_doc, best_distance = results[0]
best_similarity = 1 - best_distance

# ----------------------------
# CASE 1 : High similarity
# ----------------------------
if best_similarity >= SIMILARITY_THRESHOLD:

    context = "\n\n".join(
        doc.page_content for doc, _ in results
    )

    prompt = f"""
Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)
    print(response.content)

# ----------------------------
# CASE 2 : Low similarity
# ----------------------------
else:

    # Still search the retrieved documents
    context = "\n\n".join(
        doc.page_content for doc, _ in results
    )

    rag_prompt = f"""
You must answer ONLY using the context below.

If the answer cannot be found, reply exactly:

The provided document does not contain enough information to answer this question.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(rag_prompt)

    # If answer not found in retrieved documents,
    # switch to normal LLM
    if "The provided document does not contain enough information" in response.content:

        print("No relevant answer found in database.")
        print("Switching to general LLM...\n")

        general_response = llm.invoke(query)

        print(general_response.content)

    else:
        print(response.content)
import os
from dotenv import load_dotenv

from openai import OpenAI
from pinecone import Pinecone
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


class OpenRouterEmbeddings(Embeddings):
    def __init__(
        self,
        api_key: str,
        model: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free",
        dimensions: int = 1024,
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.model = model
        self.dimensions = dimensions

    def embed_documents(self, texts):
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
            encoding_format="float",
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text):
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
            encoding_format="float",
        )
        return response.data[0].embedding


# ----------------------------
# OpenRouter Client
# ----------------------------
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# ----------------------------
# Embeddings
# ----------------------------
embeddings = OpenRouterEmbeddings(
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# ----------------------------
# Pinecone
# ----------------------------
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
)

index = pc.Index("legal-documents")

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
)

while True:
    query = input("\nEnter your query (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    try:
        results = vector_store.similarity_search_with_score(
            query=query,
            k=2,   # use 2 instead of 4 to reduce prompt size
        )

        if not results:
            print("No documents found.")
            continue

        print("\nRetrieved Documents:")
        for i, (doc, score) in enumerate(results, start=1):
            print(f"{i}. Score: {score:.4f}")

        context = "\n\n".join(
            doc.page_content for doc, _ in results
        )

        # Limit context size
        context = context[:6000]

        print("\n========== Context Preview ==========")
        print(context[:1000])
        print("=====================================\n")

        prompt = f"""
You are an expert legal AI assistant.

Use ONLY the provided context to answer the user's question.

If the answer cannot be found in the context, reply exactly:
"I don't know based on the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

        print("Sending request to OpenRouter...")

        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert legal AI assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            max_tokens=512,
        )

        print("\n================ ANSWER ================\n")
        print(response.choices[0].message.content)
        print("\n========================================")

    except Exception as e:
        print("\nERROR:")
        print(type(e).__name__)
        print(e)
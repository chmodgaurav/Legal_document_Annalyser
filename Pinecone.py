import os
from dotenv import load_dotenv

from openai import OpenAI
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

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

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
            encoding_format="float",
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
            encoding_format="float",
        )
        return response.data[0].embedding


# ----------------------------
# Load PDF documents
# ----------------------------
loader = DirectoryLoader(
    "./data",
    glob="**/*.[Pp][Dd][Ff]",
    loader_cls=PyPDFLoader,
)

documents = loader.load()

print(f"Loaded {len(documents)} documents")

# ----------------------------
# Split documents
# ----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)

split_docs = splitter.split_documents(documents)

print(f"Created {len(split_docs)} chunks")

# ----------------------------
# Initialize embeddings
# ----------------------------
embeddings = OpenRouterEmbeddings(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
    dimensions=1024,
)

# ----------------------------
# Initialize Pinecone
# ----------------------------
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
)

index = pc.Index("legal-documents")

# ----------------------------
# Upload to Pinecone
# ----------------------------
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
)

vector_store.add_documents(split_docs)

print(f"Successfully indexed {len(split_docs)} document chunks.")
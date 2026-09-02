from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile

load_dotenv()

embed = OpenAIEmbeddings(model="text-embedding-3-small")


# Sample documents
SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),
]

def demo_chroma_basics():
  with tempfile.TemporaryDirectory() as tmpdir:
    # Create vector store from documents
    store = Chroma.from_documents(SAMPLE_DOCS, embed, persist_directory=tmpdir)
    print("Vetor store created and persisted")

    # Perform similarity search
    query = "What is LangChain?"
    results = store.similarity_search(query, k=2)

    print(f"Top 2 results from query '{query}:")
    for i, doc in enumerate(results):
      print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")


def demo_search_with_score():
  with tempfile.TemporaryDirectory() as tmpdir:
    # Create vector store from documents
    store = Chroma.from_documents(SAMPLE_DOCS, embed, persist_directory=tmpdir)
    print("Vetor store created and persisted")

    # Perform similarity search
    query = "What is LangChain?"

    # Lower score represents more similarity.
    results = store.similarity_search_with_score(query, k=2)

    print(f"Top 2 results from query '{query}:")
    for i, [doc, score] in enumerate(results):
      print(f"Result {i+1}: {doc.page_content} Score: {score} (Source: {doc.metadata['source']})")

def demo_metadata_filtering():
  with tempfile.TemporaryDirectory() as tmpdir:
    store = Chroma.from_documents(SAMPLE_DOCS, embed, persist_directory=tmpdir)

    query = "What databases are available?"
    filter = {"topic": "database"}
    results = store.similarity_search(query, 2, filter)

    print(f"Results with metadata filtering for query '{query}' and filter '{filter}':")
    for i, doc in enumerate(results):
      print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")

def demo_persist_chroma_db():
    store = Chroma.from_documents(SAMPLE_DOCS, embed, persist_directory="./chroma_db")

    query = "What databases are available?"
    filter = {"topic": "database"}
    results = store.similarity_search(query, 2, filter)

    print(f"Results with metadata filtering for query '{query}' and filter '{filter}':")
    for i, doc in enumerate(results):
      print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")

    del store

    store = Chroma(embedding_function=embed, persist_directory="./chroma_db")
    results = store.similarity_search(query, 2, filter)

    print(f"Results with metadata filtering for query '{query}' and filter '{filter}':")
    for i, doc in enumerate(results):
      print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")

def demo_retriever():
  with tempfile.TemporaryDirectory() as tmpdir:
    store = Chroma.from_documents(SAMPLE_DOCS, embed, persist_directory=tmpdir)
    similarity_retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    mmr_retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5})

    chain = RunnableLambda(lambda question: question) | similarity_retriever
    docs = chain.invoke("How do I build AI applications?")
    print(f"Similarity retriever results:")
    for i, doc in enumerate(docs):
      print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")

    print('-' * 40)

    docs = mmr_retriever.invoke("How do I build AI applications?")
    print(f"MMR retriever results:")
    for i, doc in enumerate(docs):
      print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")

def exercise_vector_store_setup():
  """
    Exercise: Create a complete vector store setup that:
      1. Takes a list of text strings
      2. Splits them into chunks
      3. Stores in Chroma
      4. Returns a configured retriever
  """
  def get_retriever(texts: list[str]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_documents(texts)
    store = Chroma.from_documents(chunks, embed)
    return store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

  result = get_retriever(SAMPLE_DOCS).invoke("What is LangChain?")
  for i, doc in enumerate(result):
    print(f"Result {i+1}: {doc.page_content}")


if __name__ == "__main__":
  # demo_chroma_basics()
  # demo_search_with_score()
  # demo_metadata_filtering()
  # demo_persist_chroma_db()
  # demo_retriever()
  exercise_vector_store_setup()

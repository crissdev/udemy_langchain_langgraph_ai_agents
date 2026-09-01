import tempfile

from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
import numpy as np

load_dotenv()

embed = OpenAIEmbeddings(model="text-embedding-3-small")

def demo_basic_embedding():
  text = "What is machine learning?"
  single_embedding = embed.embed_query(text)
  print(F'Vector dimensions: {len(single_embedding)}')
  print(f"First 5 values: {single_embedding[:5]}")
  print(f"Vector norm: {np.linalg.norm(single_embedding):4f}")

def demo_batch_embeddings():
  text = [
    "What is Machine Learning?",
    "Explain the concept of overfitting in ML",
    "How does a neural network work?",
  ]
  embeddings = embed.embed_documents(text)
  for i, emb in enumerate(embeddings):
    print(f"Text {i+1} – Vector dimensions: {len(emb)}")
    print(f"Text {i+1} – First 5 values: {emb[:5]}")
    print(f"Text {i+1} – Vector norm: {np.linalg.norm(emb)}")

def similarity_search():
  docs = [
    "Python is a programming language",
    "JavaScript is used for web development",
    "Machine learning enables AI applications",
    "Deep learning uses neural networks",
    "Cats are popular pets",
  ]

  query = "Tell me what programming languages exist?"

  # Embed documents and query
  doc_embeddings = embed.embed_documents(docs)
  qry_embedding = embed.embed_query(query)

  similarities = [cosine_similarity(qry_embedding, dv) for dv in doc_embeddings]

  # rank documents by similarity
  ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)

  print (f"Query: {query}\n")
  print("Ranked by similarity:")

  for doc, score in ranked_docs:
    print(f"    {score:.4f}: {doc}")


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
  return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def demo_embedding_caching():
  with tempfile.TemporaryDirectory() as tmpdir:
    store = LocalFileStore(tmpdir)
    cached_embed = CacheBackedEmbeddings.from_bytes_store(
      underlying_embeddings=embed,
      document_embedding_cache=store,
      namespace="exercise"
    )

    text = "What is reinforcement learning?"

    # First call - API
    e1 = cached_embed.embed_documents([text])
    print(f"    Embedded: {len(e1)} documents")

    # Second call - Cache
    e2 = cached_embed.embed_documents([text])
    print(f"    Embedded: {len(e2)} documents")

    # Verify same results
    print(f"\nSame vectors: {np.allclose(e1[0], e2[0])}")



if __name__ == '__main__':
  # demo_basic_embedding()
  # demo_batch_embeddings()
  # similarity_search()
  demo_embedding_caching()

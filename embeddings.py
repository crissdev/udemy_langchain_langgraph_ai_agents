from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings


# OllamaEmbeddings("llama2-7b-bedding-q4_0")

load_dotenv()

def demo_embeddings():
  embed = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=256)

  # single text
  text = u'This is a sample text to embed'
  embedding = embed.embed_query(text)
  print(f"mbedding for single text: {embedding}")

def demo_hugging_face_free_model():
  HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


if __name__ == "__main__":
  # demo_embeddings()
  demo_hugging_face_free_model()

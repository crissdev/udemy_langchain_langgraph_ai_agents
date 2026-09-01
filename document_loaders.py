from posixpath import basename

from dotenv import load_dotenv
import tempfile
from pathlib import Path
import bs4
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, WebBaseLoader
import os

load_dotenv()


def demo_load_text_file():
  with tempfile.NamedTemporaryFile(delete=True, suffix=".txt") as tmp_file:
    tmp_file.writelines([
      b"Hello, this is a sample text file.\n",
      b"This file is used to test the TextLoader from langchain_community package"
    ])
    tmp_file.flush()

    loader = TextLoader(tmp_file.name)
    docs = loader.load()

    print("-" * 40)
    print("Loader:")
    for doc in docs:
      print(doc.page_content)
      print(doc.metadata)
      print(doc.type)
      print(doc.id)

    print("-" * 40)
    tmp_file.seek(0)
    for line in tmp_file.readlines():
      print(line)

def demo_web_loader():
  loader = WebBaseLoader("https://frontend-mastery.community")
  docs = loader.load()

  for doc in docs:
    for line in doc.page_content.split("\n"):
      if len(line.strip()) > 4:
        print(line)

def demo_lazy_loader():
  dir = tempfile.gettempdir()
  file_names = []

  try:
    for i in range(5):
      path = Path(dir) / f"doc_{i}.txt"
      file_names.append(path)
      path.write_text(f"This is document {i}. It contains sample content.")

    loader = DirectoryLoader(dir, "doc_*.txt", loader_cls=TextLoader)

    print(f"Initialized lazy loader for directory: {dir}")
    for doc in loader.lazy_load():
      print(f"{basename(doc.metadata['source'])}: {doc.page_content}")
  finally:
    for name in file_names:
      os.unlink(name)

def demo_pdf_loader():
  file_path = '/Users/cristian/Downloads/factura depunere DU.pdf'
  loader = PyPDFLoader(file_path)
  docs = loader.load()

  print(f"Documents loaded: {len(docs)}")

  for i, doc in enumerate(docs):
    print(f"Document {i+1} Content Preview: {doc.page_content[0:100]}")
    print(f"Metadata: {doc.metadata}")

if __name__ == "__main__":
  # demo_load_text_file()
  # demo_web_loader()
  # demo_lazy_loader()
  demo_pdf_loader()
  pass

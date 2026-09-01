from dotenv import load_dotenv

load_dotenv()

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_community.document_loaders import PyPDFLoader

# Sample documents for testing
SAMPLE_TEXT = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled data to train models. The algorithm learns to map inputs to outputs based on example input-output pairs.

Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data. The algorithm discovers structure without predefined labels.

Common algorithms include:
- K-Means Clustering
- Principal Component Analysis
- Autoencoders

## Applications

Machine learning is used in many fields:
1. Image recognition
2. Natural language processing
3. Recommendation systems
4. Fraud detection
5. Autonomous vehicles
""".strip()

SAMPLE_CODE = '''
def quicksort(arr):
    """
    Quicksort implementation in Python.
    Time complexity: O(n log n) average, O(n²) worst case.
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


def binary_search(arr, target):
    """
    Binary search implementation.
    Requires sorted array.
    Time complexity: O(log n)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
'''



def demo_recursive_splitter():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(SAMPLE_TEXT)

    print(f"Original length: {len(SAMPLE_TEXT)} chars")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Chunk sized: {[len(c) for c in chunks]}")
    print(f"\nFirst chunk preview: {chunks[0][:200]}...")


def demo_chunk_size_comparison():
    sizes = [200, 500, 1000]
    print(f"=== Chunk size comparison over {len(SAMPLE_TEXT)} chars ===")
    for size in sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=size // 5
        )
        chunks = splitter.split_text(SAMPLE_TEXT)
        print(f"Size: {size}: {len(chunks)} chunks")


def demo_markdown_splitter():
    headers_to_consider = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_consider)
    chunks = splitter.split_text(SAMPLE_TEXT)

    print(f"Markdown splitter produced {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i + 1} ---\n{chunk}...\n")
        print(f"Metadata: {chunk.metadata}")

def demo_code_splitter():
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.PYTHON,
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(SAMPLE_CODE)
    print(f"Code splitter produced: {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i + 1} ---\n{chunk}...\n")

def demo_pdf_splitter():
    loader = PyPDFLoader("./docs/langchain_demo.pdf")
    docs = loader.load()

    print(f"Loaded {len(docs)} docs from PDF")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    print(f"Split into {len(split_docs)} chunks")
    print(f"\nFirst chunk metadata: {split_docs[0].metadata}")
    print(f"First chunk content: : {split_docs[0].page_content:200}...")




if __name__ == "__main__":
    # demo_recursive_splitter()
    # demo_chunk_size_comparison()
    # demo_markdown_splitter()
    # demo_code_splitter()
    demo_pdf_splitter()

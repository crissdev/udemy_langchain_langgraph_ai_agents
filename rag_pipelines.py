from textwrap import dedent

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langsmith import traceable
from pydantic import BaseModel, Field

load_dotenv()

# Sample knowledge base
KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""

embed = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=256)

def get_kb_store():
  splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
  doc = Document(page_content=KNOWLEDGE_BASE, metadata={"source": "langchain_knowledge_base.md"})
  chunks = splitter.split_documents([doc])
  print(f"Number of chunks: {len(chunks)}")
  for i, c in enumerate(chunks):
    print(f"{i}: {c.page_content[:100]}...")
  store = Chroma.from_documents(chunks, embed)
  return store

def demo_basic_rag():
  store = get_kb_store()
  retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

  model = init_chat_model(model="gpt-5-nano")
  prompt = ChatPromptTemplate.from_template(dedent("""
                  Answer the question based only on the following context:

                  {context}

                  Question: {question}

                  Answer:

                  Make sure to answer in a concise manner, and if you don't know just say 'I don't know'

                  """))

  format_docs = lambda docs: "\n\n".join(doc.page_content for doc in docs)
  rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
  )

  questions = [
    "What is LangChain",
    "Who created LangChain?",
    "What is LangGraph used for?"
  ]
  for question in questions:
    result = rag_chain.invoke(question)
    print(f"Q: {question}")
    print(f"A: {result}")
    print('-' * 40)

def demo_rag_with_structured_output():
  store = get_kb_store()
  retriever = store.as_retriever(search_kwargs={"k": 3})
  #
  class Response(BaseModel):
    answer: str = Field(description="The answer to the question")
    confidence: str = Field(description="high, medium or low")
    sources_used: list[str] = Field(description="List of souces referenced")
    follow_up: str = Field(description="Suggested follow-up question")
  #
  model = init_chat_model(model="gpt-5-nano").with_structured_output(Response)

  prompt = ChatPromptTemplate.from_template(dedent("""
                Based don the context below, answer the question.
                Context:
                  {context}

                Question:
                  {question}

                Provide a structured response.
                    """))

  def format_docs(docs: list[Document]):
    return "\n\n".join(f"[{d.metadata.get('source', 'unknown')}]: {d.page_content}" for d in docs)

  rag_chain = (
    RunnableParallel({"question": RunnablePassthrough(), "context": retriever | format_docs })
    | prompt
    | model
  )

  print("-" * 60)

  q = "What is C#?"
  print(f"Question: {q}")
  result: Response = rag_chain.invoke(q)
  print(f"Answer: {result.answer}")
  print(f"Confidence: {result.confidence}")
  print(f"Sources: {result.sources_used}")
  print(f"Follow-up: {result.follow_up}")

@traceable(name="exercise_document_qa_v1", run_type="chain")
def exercise_document_qa_v1():
  """
    Exercise: Build a complete document Q&A system that:
      1. Takes a text document as input
      2. Splits and embeds it
      3. Allows multiple questions
      4. Returns answers with confidence scores.
  """
  def get_answers(text: str, questions: list[str]):
    class AnswerResponse(BaseModel):
      answer: str = Field(description="Answer to the question")
      confidence: str = Field(description="Confidence score")

    prompt = ChatPromptTemplate.from_template(dedent("""
      Respond to the following question using information only from the context.

      Context:
        {context}

      Question:
        {question}

      Only use the information provided in the context. If you don't know the answer to a question reply with 'I have no information about this'.
      """))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents([Document(page_content=text, metadata={"source": "user provided text"})])
    store = Chroma.from_documents(docs, embedding=embed)
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    model = init_chat_model('gpt-5-nano').with_structured_output(AnswerResponse)
    results: list[dict[str, AnswerResponse]] = []

    def format_docs(docs: list[Document]):
      return "\n\n".join(d.page_content for d in docs)

    for i, q in enumerate(questions):
      context = retriever.invoke(q)
      messages = prompt.format_messages(context=format_docs(context), question=q)
      a: AnswerResponse = model.invoke(messages)
      results.append({"question": q, "answer": a})

    return results

  results = get_answers(KNOWLEDGE_BASE, ["What is LangChain", "Who created LandChain?", "When was LangChain created?"])
  for i, entry in enumerate(results):
    print(f"Q: {entry['question']}")
    print(f"Answer: {entry['answer'].answer}")
    print(f"Confidence: {entry['answer'].confidence}")
    print("-" * 40)

@traceable(name="exercise_document_qa_v2", run_type="chain")
def exercise_document_qa_v2():
  """
    Exercise: Build a complete document Q&A system that:
      1. Takes a text document as input
      2. Splits and embeds it
      3. Allows multiple questions
      4. Returns answers with confidence scores.
  """
  def get_answers(text: str, questions: list[str]):
    class AnswerResponse(BaseModel):
      answer: str = Field(description="Answer to the question")
      confidence: str = Field(description="Confidence score")

    prompt = ChatPromptTemplate.from_template(dedent("""
      Respond to the following question using information only from the context.
      Rate your confidence (high/low/medium)

      Context:
        {context}

      Question:
        {question}

      Only use the information provided in the context. If you don't know the answer to a question reply with 'I have no information about this'.
      """))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents([Document(page_content=text, metadata={"source": "user provided text"})])
    store = Chroma.from_documents(docs, embedding=embed)
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    model = init_chat_model('gpt-5-nano').with_structured_output(AnswerResponse)
    results: list[dict[str, AnswerResponse]] = []

    def format_docs(docs: list[Document]):
      return "\n\n".join(d.page_content for d in docs)

    chain = (
      {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | prompt
      | model
    )

    for i, q in enumerate(questions):
      a: AnswerResponse = chain.invoke(q)
      results.append({"question": q, "answer": a})

    return results

  results = get_answers(KNOWLEDGE_BASE, ["What is LangChain", "Who created LandChain?", "When was LangChain created?"])
  for i, entry in enumerate(results):
    print(f"Q: {entry['question']}")
    print(f"Answer: {entry['answer'].answer}")
    print(f"Confidence: {entry['answer'].confidence}")
    print("-" * 40)



if __name__ == "__main__":
  # chain = RunnablePassthrough(lambda input: input) | RunnableLambda(lambda t: print(f"{t}"))
  # chain.invoke("testing")
  # demo_basic_rag()
  # demo_rag_with_structured_output()
  exercise_document_qa_v1()
  exercise_document_qa_v2()

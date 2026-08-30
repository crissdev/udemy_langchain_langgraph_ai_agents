from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, prompt
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def demo_basic_chain():
  """Demonstrates a basic chain using LCEL and Runnable"""

  # Component 1: Define the prompt template using LCEL
  prompt = ChatPromptTemplate.from_template("You're a helpful assistant. Answer in one sentence: {question}")
  model = ChatOpenAI(model="gpt-5-nano", temperature=0.7)
  parser = StrOutputParser()

  # Compose with pipe operator
  chain = prompt | model | parser

  # Execute the chain with an input
  result = chain.invoke({"question": "What is LangChain?"})
  print(f"Response: {result}")

  return chain


def demo_batch_execution():
  """Demonstrate batch execution for multiple inputs"""
  prompt = ChatPromptTemplate.from_template("Translate to French: {text}")
  model = ChatOpenAI(model="gpt-5-nano", temperature=0.7, reasoning_effort='minimal')
  parser = StrOutputParser()

  chain = prompt | model | parser

  # Batch - run with multiple inputs
  inputs = [
    {"text": "Hello, how are you?"},
    {"text": "What is your name"}
  ]

  results = chain.batch(inputs)

  for text in zip(inputs, results):
    print(f"Input: {text[0]['text']} => Output: {text[1]}")


def demo_streaming():
  """Demonstrate streaming fo real-time output"""
  prompt = ChatPromptTemplate.from_template("Write a haiku about: {topic}")
  model = ChatOpenAI(model="gpt-5-nano", temperature=0.7, reasoning_effort="minimal")
  parser = StrOutputParser()

  chain = prompt | model | parser

  # Streaming - run with streaming enabled
  print("Streaming output:")
  for chunk in chain.stream({"topic": "nature"}):
    print(chunk, end="", flush=True)
  print()


def demo_schema_inspection():
  """Demonstrate input/output schema inspection"""
  prompt = ChatPromptTemplate.from_template("Summarize the following text: {text}")
  model = ChatOpenAI(model_name="gpt-5-nano", temperature=0.7, reasoning_effort="minimal")
  parser = StrOutputParser()

  chain = prompt | model | parser

  # Inspect input and output schema
  input_schema = chain.input_schema.model_json_schema()
  output_schema = chain.output_schema.model_json_schema()

  print(f"Input schema: {input_schema}")
  print(f"Output schema: {output_schema}")


def exercise_first_chain():
  """
  EXERCISE: Create a chain that:
    1. Takes a product name and target audience
    2. Generates a marketing tagline
    3. Returns just the tagline as a string

    Test with: product="AI Course", audience="developers"
  """
  prompt = ChatPromptTemplate.from_template(
    "Create a marketing tagline for a product named '{product}' and audience '{audience}'")
  model = ChatOpenAI(model_name="gpt-5-nano", temperature=0.7, reasoning_effort='minimal')
  parser = StrOutputParser()

  chain = prompt | model | parser

  input = {"product": "AI Course", "audience": "developers"}
  use_streaming = True

  if use_streaming:
    for chunk in chain.stream(input):
      print(chunk, end="")
    print()
  else:
    result = chain.invoke(input)
    print(result)


def universal_way_to_init_models():
  # The universal way to initialize models
  model = init_chat_model("fable", temperature=0.7, reasoning_effort="minimal",
                          max_tokens=100)
  #


if __name__ == "__main__":
  # demo_basic_chain()
  # print("\n--------------\n")
  # demo_batch_execution()
  # print("\n--------------\n")
  # demo_streaming()
  # print("\n--------------\n")
  # demo_schema_inspection()
  # print("\n--------------\n")
  # exercise_first_chain()
  universal_way_to_init_models()

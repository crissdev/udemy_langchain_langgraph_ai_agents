"""
  Section 1: Project Smart Q&A Bot
  A production ready question-answering both with structured output.
"""
import os
from textwrap import dedent
from typing import List

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import BaseModel, Field

from prompt_messages import response

load_dotenv()


#
# langsmithClient = Client(
#   api_key=os.getenv("LANGSMITH_API_KEY"))


class QAResponse(BaseModel):
  answer: str = Field(description="The answer to the user's question")
  confidence: str = Field(description="Confidence label: high, medium or low")
  reasoning: str = Field(description="The reasoning behind the answer provided")
  follow_up_questions: List[str] = Field(description="A list of follow-up questions related to the topic")
  sources_needed: bool = Field(description="Indicates whether sources are needed for the answer", default=False)


class SmartQABot:
  def __init__(self, model_name: str, temperature: float = 0.3):
    self.model = ChatOpenAI(model_name=model_name, temperature=temperature).with_structured_output(QAResponse)
    self.prompt = ChatPromptTemplate([
      ("system", dedent("""You are a knowledgeable Q&A assistant.
                      Your guidelines:
                        - Answer questions accurately and concisely
                        - Be honest about uncertainty — set confidence to 'low' if unsure
                        - Provide clear reasoning for your answers
                        - Suggest relevant follow-up questions
                        - Indicate if external sources would help
                      Always respond with accurate, helpful information.""")),
      ("human", "{question}")
    ])
    self.chain = self.prompt | self.model

  @traceable(name="ask_question", run_type="chain")
  def ask(self, question: str) -> QAResponse:
    try:
      response = self.chain.invoke({"question": question})
      return response
    except Exception as e:
      return QAResponse(
        answer="I'm sorry, I couldn't process your question at this time",
        confidence="low",
        reasoning=str(e),
        follow_up_questions=["Could you please try again later?"],
        sources_needed=False
      )

  @traceable(name="ask_batch", run_type="chain")
  def ask_batch(self, questions: List[str]):
    """Ask multiple questions in parallel"""
    inputs = [{"question": q} for q in questions]
    return self.chain.batch(inputs)


def demo_qa_bot():
  bot = SmartQABot("gpt-5-nano")
  questions = [
    "What is the capital of France?",
    "Explain the theory of relativity",
    "How does photosynthesis work?",
  ]

  print("=" * 60)
  print("Smart Q&A Bot Demo")
  print("=" * 60)

  for question in questions:
    response = bot.ask(question)
    print(f"Question: {question}")
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence}")
    print(f"Reasoning: {response.reasoning}")
    print(f"Follow-up questions: {response.follow_up_questions}")
    print(f"Sources needed: {response.sources_needed}")
    print("-" * 40)


def demo_error_handling():
  bot = SmartQABot("gpt-5-nano")

  print("=" * 60)
  print("Smart Q&A Bot — Error Handling Demo")
  print("=" * 60)

  long_question = "What is " + "very " * 100 + "important?"

  response = bot.ask(long_question)
  print(f"Handled gracefully: {response.confidence}")
  print("-" * 60)
  print("Full API Response:")
  print(response)


if __name__ == "__main__":
  # demo_qa_bot()
  demo_error_handling()

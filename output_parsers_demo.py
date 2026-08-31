from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI(model_name="gpt-5-nano")


def demo_text_parser():
  parser = StrOutputParser()
  prompt = ChatPromptTemplate.from_template("Write a 4 line short poem about {topic}")
  chain = prompt | model | parser
  response = chain.invoke({"topic": "nature"})
  print(response)


def demo_json_parser():
  parser = JsonOutputParser()
  prompt = ChatPromptTemplate.from_messages([
    ("system", "Respond with a JSON object only having the 'content' key holding the result"),
    ("user", "Write a 4 line short poem about {topic}")
  ])
  chain = prompt | model | parser
  response = chain.invoke({"topic": "summer"})
  print(response)


def demo_pydantic_parser():
  class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")

  parser = PydanticOutputParser(pydantic_object=Person)
  print(f"Format instructions: {parser.get_format_instructions()}")

  prompt = ChatPromptTemplate.from_template(
    "Return a JSON object with 'name', 'age' and 'occupation' for {description}").partial(
    format_instructions=parser.get_format_instructions())

  chain = prompt | model | parser
  print(chain.invoke({"description": "A 20 year old artist named Maria"}))


def demo_structured_output():
  class MovieReview(BaseModel):
    title: str = Field(description="The title of the movie")
    review: str = Field(description="A brief review of the movie")
    rating: int = Field(description="The rating of the movie out of 10")

  # bind the schema to the model
  structured_model = model.with_structured_output(MovieReview)
  print(structured_model.invoke("Review Inception"))


if __name__ == "__main__":
  demo_structured_output()

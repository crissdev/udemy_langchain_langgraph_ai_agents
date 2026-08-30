from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

#
prompt = ChatPromptTemplate.from_template("Tell me a {kind} joke about {topic}")
# messages = prompt.format_messages(**{"kind": "funny", "topic": "policeman"})
prompt.messages.insert(0, SystemMessage(
  """You are a comedian and tell jokes. Your output must be exactly one joke and nothing else. Don't make suggestions or add extra text or content apart from the joke itself. Always generate a new joke. Here's a list of what was already generated:
1. Why did the chicken join a band? Because it already had the drumsticks."""))
# messages = prompt.format_messages(kind="funny", topic="chickens")
# response = model.invoke(prompt)

model = ChatOpenAI(model="gpt-5-nano")
parser = StrOutputParser()
chain = prompt | model | parser

response = chain.invoke({"kind": "funny", "topic": "policeman"})
print(response)

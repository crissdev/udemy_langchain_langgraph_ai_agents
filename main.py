from dotenv import load_dotenv

load_dotenv()

from langchain_core import __version__ as langchain_core_version
from langgraph import version as langgraph_version
from langchain_openai import ChatOpenAI, __version__ as langchain_openai_version
from langchain_anthropic import ChatAnthropic, __version__ as langchain_anthropic_version

print("langchain_core: " + langchain_core_version)
print("langgraph: " + langgraph_version.__version__)
print("langchain_openai: " + langchain_openai_version)
print("langchain_anthropic: " + langchain_anthropic_version)


def main():
  llm = ChatOpenAI(model_name="gpt-5-nano", temperature=0)
  response = llm.invoke("Say 'setup complete' in one word")
  print(response)
  print("Hello from langc-course!")


if __name__ == "__main__":
  main()

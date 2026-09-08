import random

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

load_dotenv()


model = init_chat_model('gpt-5-nano')
random.seed()


@traceable()
def demo_basic_tracing():
    """Basic LangSmith tracing"""
    topics = [
        "Agentic Engineering",
        "Prompt Engineering",
        "Context Management",
        "Tool/Function calling",
        "Agents Orchestration"
    ]
    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in one sentence")
    chain = prompt | model | StrOutputParser()
    random_topic = topics[random.randint(0, len(topics) - 1)]
    response = chain.invoke({"topic": random_topic})
    print(response)


@traceable(tags=["development", "local"])
def demo_named_runs():
    """Name your runs for easier identification"""
    prompt = ChatPromptTemplate.from_template(
        "Summarize {text} in no more than 2 sentences")
    chain = prompt | model | StrOutputParser()
    response = chain.invoke(
        {"text": "LangSmith provides observability for LLM applications"})
    print(response)


@traceable(tags=["development"], metadata={"complexity": "low", "api-version": "v2"})
def demo_trace_with_metadata(user_id: str):
    """Add metadata to traces for filtering"""
    response = model.invoke(f"Hello from user {user_id}")
    print(response.content)


if __name__ == '__main__':
    line = '-' * 60
    demo_basic_tracing()
    print(line)
    demo_named_runs()
    print(line)
    demo_trace_with_metadata('cristian')

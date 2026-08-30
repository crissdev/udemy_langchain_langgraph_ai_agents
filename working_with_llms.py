from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI

load_dotenv()


def demo_init_chat_model(model_name: str, prompt: str):
  model = init_chat_model(model_name)
  response = model.invoke(prompt)
  print(f"{model_name}: {response.content}")


def demo_messages():
  model = ChatOpenAI(model_name='gpt-5-nano')
  messages: list[BaseMessage] = [
    SystemMessage(content="You are a pirate. Always answer like a pirate"),
    HumanMessage(content="What is the weather like today?")
  ]
  response = model.invoke(messages)
  print(response.content)

  messages.append(response)
  messages.append(HumanMessage('What about tomorrow?'))
  response = model.invoke(messages)
  print(response.content)


def demo_model_comparison():
  prompt = 'Explain recursion in one sentence'
  demo_init_chat_model('gpt-5-nano', prompt)
  demo_init_chat_model('claude-haiku-4-5-20251001', prompt)


def exercise_multi_model():
  """
    EXERCISE:
      1. Takes a question and a list of model names
      2. Gets responses from all models
      3. Returns a dictionary of {model_name: response}

    Test with: question="What is AI?", models: ["gpt-5-nano", "claude-haiku-4-5-20251001"]
  """
  model_names: list[str] = ["gpt-5-nano", "claude-haiku-4-5-20251001"]
  prompt = "What is AI?"
  result: dict[str, list[BaseMessage]] = {}

  for model_name in model_names:
    result[model_name] = _get_model_response(model_name, prompt)

  print(result)


def _get_model_response(model_name: str, prompt: str) -> list[BaseMessage]:
  model = init_chat_model(model_name)
  messages: list[BaseMessage] = [
    SystemMessage("You are a senior AI consultant giving concise and terse replies for AI-related questions."),
    HumanMessage(prompt)
  ]
  messages.append(model.invoke(messages))
  return messages


if __name__ == '__main__':
  # demo_model_comparison()
  # demo_messages()
  exercise_multi_model()

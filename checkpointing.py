"""
Save and resume agent state
"""

import os
from sqlite3 import Connection

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated
import tempfile
import operator

load_dotenv()

model = init_chat_model('gpt-5-nano')


def demo_sqlite_saver():
    class ChatState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    def chat(state: ChatState) -> ChatState:
        response = model.invoke(state['messages'])
        return {
            'messages': [response]
        }

    graph = StateGraph(ChatState)
    graph.add_node(chat)

    graph.add_edge(START, 'chat')
    graph.add_edge('chat', END)

    if not os.path.exists('./storage'):
        os.mkdir("./storage")
    saver = SqliteSaver(Connection(
        "./storage/chat.db", check_same_thread=False))
    # saver = MemorySaver()
    app = graph.compile(saver)

    config: RunnableConfig = {"configurable": {
        "thread_id": "smith_a8u92ef5-2026-09-06"}}

    result: ChatState = app.invoke(
        ChatState(messages=["My name is Paulo"]), config=config)

    print(f"Turn 1 - AI: {result['messages'][-1].content}")

    # Turn 2 – Conversation continues
    result = app.invoke(ChatState(messages="What is my name?"), config=config)
    print(f"Turn 2 - AI: {result['messages'][-1].content[:100]}...")

    # Check full history
    state = app.get_state(config)
    print(f"\nTotal messages in state: {len(state.values['messages'])}")

    hist = app.get_state_history(config)
    for i, item in enumerate(hist):
        print(f"{i}: {len(item.values['messages'])}")


if __name__ == '__main__':
    demo_sqlite_saver()

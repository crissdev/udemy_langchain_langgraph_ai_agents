""""
Conversation Memory in LangChain
Modern approaches to maintaining conversation context
"""
import logging

from dedent import dedent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableConfig, RunnableWithMessageHistory
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from langsmith import traceable

load_dotenv()

logging.basicConfig(level=logging.INFO)


@traceable(name="demo_basic_memory", run_type="chain")
def demo_basic_memory():
    """Basic conversation memory with RunnableWithMessageHistory"""

    model = init_chat_model(model="gpt-5-nano")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be concise."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    chain = prompt | model | StrOutputParser()

    # Session storage
    store: dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    # Wrap with history
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history")

    # Configuration for this session
    config: RunnableConfig = {"configurable": {"session_id": "user_123"}}

    # Conversation
    messages = [
        "Hi! My name is Cristian.",
        "I'm learning about LangChain.",
        "What's my name and what am I learning?"
    ]

    print("\nConversation:")
    for msg in messages:
        print(f"\nUser: {msg}")
        response = chain_with_history.invoke({"input": msg}, config=config)
        print(f"AI: {response}")

    # Show stored history
    print(
        f"\n--- Stored History ({len(store['user_123'].messages)} messages) ---")
    for msg in store['user_123'].messages:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"   {role}: {msg.content[:50]}...")


@traceable(name="demo_message_trimming", run_type="chain")
def demo_message_trimming():
    messages = [
        ('system', 'You are a helpful assistant'),
        ('human', 'What is Python?'),
        ('ai', 'Python is an interpreted, object-oriented, high-level programming language with dynamic semantics. Its high-level built in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development, as well as for use as a scripting or glue language to connect existing components together.'),
        ('human', 'How do I install it?'),
        ('ai', 'You can install Python from python.org'),
        ('human', 'What about pip?'),
        ('ai', 'Pip is the package installer for Python. You can use it to install packages from the Python Package Index and other indexes.'),
        ('human', 'Can you summarize everything we discussed?'),
    ]
    print(f"\nOriginal messages: {len(messages)}")

    model = init_chat_model('gpt-5-nano')
    trim_max_tokens = 60
    trimmed = trim_messages(messages, max_tokens=trim_max_tokens, strategy="last", token_counter=model,
                            include_system=True, allow_partial=False)

    print(
        f"After trimming (max: {trim_max_tokens} tokens): {len(trimmed)} messages")
    print(f"\nTrimmed messages:")
    for msg in trimmed:
        role = type(msg).__name__.replace("Message", "")
        print(f"  {role}: {msg.content[:60]}...")


@traceable(name="demo_sliding_window_memory", run_type="chain")
def demo_sliding_window_memory():
    class WindowedChatHistory(InMemoryChatMessageHistory):
        k: int = 0

        def __init__(self, k: int = 3):
            super().__init__()
            self.k = k

        def add_message(self, messages):
            super().add_message(messages)
            # Keep only last k pairs (2k messages: human + ai)
            if len(self.messages) > self.k * 2:
                self.messages = self.messages[-(self.k * 2):]

    store: dict[str, WindowedChatHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = WindowedChatHistory(2)
        return store[session_id]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    model = init_chat_model('gpt-5-nano')
    chain = prompt | model | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="history")

    # Simulate a long conversation
    conversation = [
        "My name is Paulo",
        "I live in Seattle",
        "I work as an AI engineer",
        "I have 2 cats",
        "What do you remember about me?"
    ]
    config = {"configurable": {"session_id": "u123"}}

    print(f"Conversation with k=2 window:")
    for i, msg in enumerate(conversation, 1):
        print(f"\nUser: {msg}")
        response = chain_with_history.invoke({"input": msg}, config=config)
        print(f"AI {response}")

        # Show window state after each exchange so students see it sliding
        history = store[config['configurable']['session_id']].messages
        print(f"  [Window: {len(history)} messages] ", end="")
        facts_in_memory = [
            m.content[:40] for m in history if isinstance(m, HumanMessage)
        ]
        print(f"Remembers: {facts_in_memory}")

    # Final state - show what survived and what was lost
    print("\n" + "=" * 60)
    print(f"Result: window only helpt last 2 exchanges!")
    print(f"Lost: name (Paulo), city (Seattle), and job (AI engineer)")
    print("Kept: cats + the 'remember' question")
    print("This is the trade-off: fixed memory = predictable cost, but older context is lost")


@traceable(name="demo_memory_summary", run_type="chain")
def demo_summary_memory():
    """auto-summarize old messages but keep recent onces"""
    print("=" * 60)
    print("Summarize older messages to save tokens")
    print("=" * 60)

    # Setup
    summary_llm = init_chat_model("gpt-5-nano", temperature=0.0)
    chat_llm = init_chat_model("gpt-5-nano", temperature=0.7)

    # The conversation prompt: summary of old context + recent messages
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                dedent("""
                  You are a helpful assistant. Be concise.
                  Summary of earlier conversation:
                  {summary}
                    """)
            ),
            MessagesPlaceholder(variable_name="recent_messages"),
            "{input}"
        ]
    )

    chat_chain = chat_prompt | chat_llm | StrOutputParser()

    # The summarization prompt: compress messages into a running memory
    summarize_prompt = ChatPromptTemplate.from_template(dedent("""
        Condense the current summary and new messages into a single updated summary (2-3 sentences).
        Preserve all key facts about the user.

        Current summary:
        {current_summary}

        New messages:
        {new_messages}

        Updated summary:
          """))
    summarize_prompt = summarize_prompt | summary_llm | StrOutputParser()
    summarize_chain = summarize_prompt | summary_llm | StrOutputParser()

    # State
    running_summary = ''
    recent_messages: list[BaseMessage] = []
    MAX_RECENT = 4

    # Conversation
    exchanges = [
        "My name is Paulo and I'm from Seattle",
        "I work as an AI engineer building RAG systems"
        "I have 2 cats named Luna and Milo",
        "I'm building a LangChain course for Udemy",
        "What do you know about me? List everything."
    ]

    print(f"\nConfig: keep last {MAX_RECENT} messages, summarize the rest")

    for user_input in exchanges:
        print(f"User: {user_input}")

        # 1. Call the LLM with summary + recent messages + new input
        response = chat_chain.invoke({
            "summary": running_summary if running_summary else "No prior conversation",
            "recent_messages": recent_messages,
            "input": user_input
        })
        print(f"AI: {response}")

        # 2. Add this exchange to recent messages
        recent_messages.append(HumanMessage(content=user_input))
        recent_messages.append(AIMessage(content=response))

        # 3. If recent messages exceed limmit summarize the oldest ones
        if len(recent_messages) > MAX_RECENT:
            # Take the oldest messages that will be summarized away
            messages_to_summarize = recent_messages[:-MAX_RECENT]
            formatted = "\n".join(
                f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                for m in messages_to_summarize
            )

            # Updating the running_summary
            running_summary = summarize_chain.invoke({
                "current_summary": running_summary if running_summary else "None yet",
                "new_messages": formatted
            })

            # Keep only the most recent messages
            recent_messages = recent_messages[:-MAX_RECENT]
            print(
                f"  >>> Summarized! Compressed {len(messages_to_summarize)} old messages")
            print(f"  >>> Summary: {running_summary}")
            print(f"  >>> Recent buffer: {len(recent_messages)} messages")

    print()

    # Final state
    print('=' * 60)
    print("Final memory state")
    print('=' * 60)
    print(f"Running summary (compressed old context):\n  {running_summary}")
    print(f"Recent messages kept verbatim: ({len(recent_messages)}):")
    for msg in recent_messages:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {role}: {msg.content[:80]}...")
    print("\nKey insight: All facts preserved (name, city, job, cats, course)")
    print("But token cost stays bounded — old messages are compressed, not deleted")


@traceable(name="exercise_persistent_memory", run_type="chain")
def exercise_persistent_memory():
    """
      Exercise: Build a chat bot with:
        1. Persistent memory (SQLite)
        2. Automatic summarization after 10 messages
        3. User preference tracking

        Hint: combine RunnableWithMessageHistory with SQLChatMessageHistory
    """
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return SQLChatMessageHistory(session_id, connection='sqlite:///./chat_history.db')

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Remember user preferences."),
        MessagesPlaceholder(variable_name="history"),
        "{input}"
    ])
    model = init_chat_model('gpt-5-nano')
    chain = prompt | model | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain, get_session_history, input_messages_key="input", history_messages_key="history")

    print(f"\nPersistent memory chatbot:")
    print(f"Messages saved to SQLite database\n")

    # Test conversation
    test_messages = [
        "Remember that I prefer dark mode themes",
        "What theme do I prefer?"
    ]

    config = {"configurable": {"session_id": "persistent_user"}}

    for msg in test_messages:
        print(f"User: {msg}")
        response = chain_with_history.invoke({"input": msg}, config=config)
        print(f"AI: {response}\n")

    # Test recall
    print("\nTesting recall")
    print(f"User: {msg}")
    chain_with_history = RunnableWithMessageHistory(
        chain, get_session_history, input_messages_key="input", history_messages_key="history")
    response = chain_with_history.invoke({"input": msg}, config=config)
    print(f"AI: {response}\n")


if __name__ == "__main__":
    # demo_basic_memory()
    # demo_message_trimming()
    # demo_sliding_window_memory()
    # demo_summary_memory()
    exercise_persistent_memory()

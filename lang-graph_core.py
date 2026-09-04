import random

from dotenv import load_dotenv
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langsmith import traceable
from langchain_core.runnables.graph import Graph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated

load_dotenv()


# Basic State
class SimpleState(TypedDict):
    input: str
    output: str
    step: int


def demo_simple_graph():
    def process(state: SimpleState) -> dict:
        return {
            "output": state['input'].upper(),
            "step": state['step'] + 1,
        }

    graph = StateGraph(SimpleState)
    graph.add_node("process", process)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)

    app = graph.compile()
    # Vizualize the graph
    visualize_graph(app.get_graph(), "./graph.png")

    result = app.invoke({"input": "hello", "step": 10})

    print(result)


class AccumulatingState(TypedDict):
    messages: Annotated[list[str], operator.add]
    count: Annotated[int, operator.add]


def demo_accumulating_state():
    def step_one(state: AccumulatingState) -> dict:
        return {
            "messages": ["Step 1 executed"],
            "count": 1
        }

    def step_two(state: AccumulatingState) -> dict:
        return {
            "messages": ["Step 2 executed"],
            "count": 1
        }

    graph = StateGraph(AccumulatingState)
    graph.add_node("step_one", step_one)
    graph.add_node("step_two", step_two)
    graph.add_edge(START, "step_one")
    graph.add_edge("step_one", "step_two")
    graph.add_edge("step_two", END)
    app = graph.compile()
    visualize_graph(app.get_graph(), "graph-acc.png")
    result = app.invoke({})
    print(result)


class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def demo_message_state():
    llm = init_chat_model('gpt-5-nano')

    def chat_node(state: MessageState) -> dict:
        response = llm.invoke(state['messages'])
        return {
            "messages": [response]
        }

    graph = StateGraph(MessageState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)
    app = graph.compile()
    visualize_graph(app.get_graph(), './graph_llm_call.png')
    result: MessageState = app.invoke({
        "messages": [
            ("system", "You are a sarcastic person. You respond short and sarcastic"),
            "Hey there! How's your day so far?"
        ]
    })
    for msg in result['messages'][1:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        print(f"{role}: {msg.content}")

    # print(result['messages'][-1].content)


class MultiStepState(TypedDict):
    input: str
    analyzed: str
    enhanced: str
    final: str
    # messages: Annotated[list[BaseMessage], add_messages]


def demo_multi_node_graph():
    llm = init_chat_model('gpt-5-nano')

    def analyze(state: MultiStepState):
        response = llm.invoke(
            f"Analyze the following input and summarize it in one sentence: {state['input']}")
        return {"analyzed": response.content}

    def enhance(state: MultiStepState):
        response = llm.invoke(
            f"Take the following analysis and enhance it with more details: {state['analyzed']}")
        return {"enhanced": response.content}

    def finalize(state: MultiStepState):
        response = llm.invoke(
            f"Take the following enhanced analysis and finalize it into a concise summary: {state['enhanced']} ")
        return {"final": response.content}

    graph = StateGraph(MultiStepState)
    graph.add_node(analyze)
    graph.add_node(enhance)
    graph.add_node(finalize)

    graph.add_edge(START, analyze.__name__)
    graph.add_edge(analyze.__name__, enhance.__name__)
    graph.add_edge(enhance.__name__, finalize.__name__)
    graph.add_edge(finalize.__name__, END)

    app = graph.compile()
    visualize_graph(app.get_graph(), './multi_step_graph.png')
    print(app.get_graph().draw_mermaid())

    result: MultiStepState = app.invoke(
        {"input": "Artificial Intelligence."})

    print(f"Input: {result['input']}")
    print(f"Analyzed: {result['analyzed'][:100]}...")
    print(f"Enhanced: {result['enhanced'][:100]}...")
    print(f"Final: {result['final'][:200]}...")


def exercise_first_lang_graph():
    """
      Exercise: Create a LangGraph that:
        1. Takes a topic as input
        2. Node 1: Generates 3 questions about the topic
        3. Node 2: Answers one of the questions
        4. Returns both questions and answer
    """
    class TopicState(TypedDict):
        topic: str
        questions: list[str]
        answeredQuestionIndex: int
        answer: str

    model = init_chat_model('gpt-5-nano')

    def generate_questions(state: TopicState):
        class QuestionsOutput(BaseModel):
            questions: list[str] = Field(
                description="A list of philosofical questions")

        response: QuestionsOutput = model.with_structured_output(QuestionsOutput).invoke([
            ("system", "You are a reflective individual creating short and interesting philosofical questions for a given topic."),
            HumanMessage(
                f"Generate 3 philosofical questions for topic: {state['topic']}")
        ])
        return {
            "questions": response.questions
        }

    def answer_random_question(state: TopicState):
        index = random.randint(0, len(state['questions']) - 1)

        class AnswerOutput(BaseModel):
            answer: str = Field("Answer to the given question")

        response: AnswerOutput = model.with_structured_output(AnswerOutput).invoke([
            ("system", "You are a reflective individual answering to philosofical questions in a concise style."),
            HumanMessage(
                f"Question: {state['questions'][state['answeredQuestionIndex']]}")
        ])
        return {
            "answeredQuestionIndex": index,
            "answer": response.answer
        }

    graph = StateGraph(TopicState)
    graph.add_node(generate_questions)
    graph.add_node(answer_random_question)
    graph.add_edge(START, generate_questions.__name__)
    graph.add_edge(generate_questions.__name__,
                   answer_random_question.__name__)
    graph.add_edge(answer_random_question.__name__, END)

    app = graph.compile()
    state: TopicState = app.invoke({
        "topic": "Mindset",
        "questions": [],
        "answer": "",
        "answeredQuestionIndex": -1
    })

    print(f"Questions: ")
    for i, q in enumerate(state['questions']):
        print(f"  Q{i+1}: {q}")
    print(
        f"\nSelected question to answer: {state['questions'][state['answeredQuestionIndex']][:100]}...")
    print(f"Answer to selected question: {state['answer']}")


def visualize_graph(graph: Graph, filename: str):
    with open(filename, mode="wb") as f:
        f.write(graph.draw_mermaid_png(background_color="white"))


if __name__ == '__main__':
    # demo_simple_graph()
    # demo_accumulating_state()
    # demo_message_state()
    # demo_multi_node_graph()
    exercise_first_lang_graph()

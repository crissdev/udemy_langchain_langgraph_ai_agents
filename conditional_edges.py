from enum import Enum
from typing import Literal, TypedDict

from dedent import dedent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.runnables.graph import Graph
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
import asyncio

load_dotenv()


model = init_chat_model('gpt-5-nano')


def demo_basic_routing():
    class RouterState(TypedDict):
        query: str
        response: str
        query_type: str

    def classify_query(state: RouterState) -> dict:
        response = model.invoke(dedent(f"""
            Classify this query as 'question', 'command' or 'statement'. Reply
            with just the word.

            {state['query']}
            """))
        return {
            "query_type": response.content.lower().strip()
        }

    def handle_question(state: RouterState) -> dict:
        response = model.invoke(f"Answer this question: {state['query']}")
        return {
            "response": f"Answer: {response.content}"
        }

    def handle_command(state: RouterState) -> dict:
        return {
            "response": f"""[Executing] I'll help you with: '{state["query"]}'"""
        }

    def handle_statement(state: RouterState) -> dict:
        return {
            "response": f"""[Acknowledged] Thanks for sharing '{state["query"]}'"""
        }

    def route_by_type(state: RouterState) -> Literal['question', 'command', 'statement']:
        qt = state['query_type']
        if 'question' in qt:
            return 'question'
        elif 'command' in qt:
            return 'command'
        else:
            return 'statement'

    graph = StateGraph(RouterState)
    graph.add_node(classify_query)
    graph.add_node(handle_question)
    graph.add_node(handle_command)
    graph.add_node(handle_statement)

    graph.add_edge(START, classify_query.__name__)
    graph.add_conditional_edges(
        classify_query.__name__,
        route_by_type, {
            'question': handle_question.__name__,
            'command': handle_command.__name__,
            'statement': handle_statement.__name__,
        })

    graph.add_edge(handle_question.__name__, END)
    graph.add_edge(handle_command.__name__, END)
    graph.add_edge(handle_statement.__name__, END)

    app = graph.compile()
    visualize_graph(app.get_graph(), "./conditional_graph.png")

    questions = [
        'I love programming',
        'What is the capital of France?',
        'Send an email to Cristian'
    ]

    for q in questions:
        response: RouterState = app.invoke(RouterState({
            "query": q,
            "query_type": "",
            "response": ""
        }))

        print(f"Query: {q}")
        print(f"Type: {response['query_type']}")
        print(f"{response['response']}")
        print("-" * 60)


def demo_conditional_loop():
    class QualityState(TypedDict):
        content: str
        quality_score: int
        feedback: str
        final_content: str
        iteration: int

    def evaluate_equality(state: QualityState) -> dict:
        response = model.invoke(
            f"Rate this content quality from 1-10. Reply with just the number.\n\n"
            f"Content: {state['content']}"
        )
        try:
            score = int(response.content.strip())
        except:
            score = 5
        return {"quality_score": score}

    def improve_content(state: QualityState) -> dict:
        response = model.invoke(
            f"Improve this content to be more engaging and clear:\n\n"
            f"{state['content']}"
        )

        return {
            "content": response.content,
            "iteration": state['iteration'] + 1
        }

    def finalize_content(state: QualityState) -> dict:
        return {
            "final_content": state['content'],
            "feedback": f"Approved after {state['iteration']} iterations with score {state['quality_score']}"
        }

    def should_continue(state: QualityState) -> Literal["improve", "finalize"]:
        if state['quality_score'] >= 7:
            return 'finalize'
        elif state['iteration'] >= 3:
            return "finalize"
        else:
            return "improve"

    graph = StateGraph(QualityState)
    graph.add_node(evaluate_equality)
    graph.add_node(improve_content)
    graph.add_node(finalize_content)

    graph.add_edge(START, "evaluate_equality")
    graph.add_conditional_edges(
        "evaluate_equality",
        should_continue,
        {
            "improve": 'improve_content',
            "finalize": 'finalize_content'
        })
    graph.add_edge('improve_content', 'evaluate_equality')
    graph.add_edge('finalize_content', END)

    app = graph.compile()
    visualize_graph(app.get_graph(), './loop-graph.png')

    result = app.stream(QualityState({
        'content': 'This is a sample content',
        'quality_score': 0,
        'feedback': '',
        'final_content': '',
        'iteration': 0,
    }))

    for chunk in result:
        print(chunk)


def demo_multi_path_routing():
    class UrgencyEnum(str, Enum):
        URGENT = 'urgent'
        NORMAL = 'normal'

    class ComplexityEnum(str, Enum):
        COMPLEX = "complex"
        SIMPLE = "simple"

    class TaskState(TypedDict):
        task: str
        urgency: UrgencyEnum
        complexity: ComplexityEnum
        handler: str
        result: str

    def urgency(state: TaskState) -> UrgencyEnum:
        class UrgencyResponse(BaseModel):
            urgency: UrgencyEnum = Field(
                description="The urgency of the task")

        response: UrgencyResponse = model.with_structured_output(
            UrgencyResponse).invoke(f"Is this task urgent?\nTask: {state['task']}")
        return response.urgency

    def complexity(state: TaskState) -> ComplexityEnum:
        class ComplexityResponse(BaseModel):
            complexity: ComplexityEnum = Field(
                description="The complexity of the task")
        response: ComplexityResponse = model.with_structured_output(
            ComplexityResponse).invoke(f"Is this task complex?\nTask: {state['task']}")
        return response.complexity

    def analyse_task(state: TaskState) -> dict:
        return {
            "urgency": urgency(state),
            "complexity": complexity(state),
        }

    def urgent_complex_handler(state: TaskState) -> dict:
        return {
            "handler": "Senior Team",
            "result": "Escalated to senior team for immediate action"
        }

    def urgent_simple_handler(state: TaskState) -> dict:
        return {
            "handler": "Quick Response",
            "result": "Handled immediately by available agent"
        }

    def normal_complex_handler(state: TaskState) -> dict:
        return {
            "handler": "Specialist",
            "result": "Assigned to specialist for throrough handling"
        }

    def normal_simple_handler(state: TaskState) -> dict:
        return {
            "handler": "Standard",
            "result": "Added to standard queue"
        }

    def route_task(state: TaskState) -> Literal["urgent_complex", "urgent_simple", "normal_complex", "normal_simple"]:
        is_urgent = state['urgency'] == UrgencyEnum.URGENT
        is_complex = state['complexity'] == ComplexityEnum.COMPLEX

        if is_urgent and is_complex:
            return "urgent_complex"
        elif is_urgent:
            return "urgent_simple"
        elif is_complex:
            return "normal_complex"
        else:
            return "normal_simple"

    graph = StateGraph(TaskState)
    graph.add_node("analyze", analyse_task)
    graph.add_node("normal_simple", normal_simple_handler)
    graph.add_node("urgent_simple", urgent_simple_handler)
    graph.add_node("normal_complex", normal_complex_handler)
    graph.add_node("urgent_complex", urgent_complex_handler)

    graph.add_edge(START, "analyze")
    graph.add_conditional_edges("analyze", route_task, {
        "urgent_complex": "urgent_complex",
        "urgent_simple": "urgent_simple",
        "normal_complex": "normal_complex",
        "normal_simple": "normal_simple",
    })

    for node in ["urgent_complex", "urgent_simple", "normal_complex", "normal_simple"]:
        graph.add_edge(node, END)

    app = graph.compile()
    visualize_graph(app.get_graph(), "./multi_path_routing.png")

    tasks = [
        "Server is down! Need immediate fix!",
        "Update the documentation for the API!",
        "Redesign the entire database schema",
        "Fix the typo on the homepage"
    ]

    for task in tasks:
        result: TaskState = app.invoke(TaskState({
            "task": task
        }))
        print(f"Task: {task}")
        print(f"Urgency: {result['urgency']}")
        print(f"Complexity: {result['complexity']}")
        print(f"Handler: {result['handler']}")
        print(f"Result: {result['result']}")
        print(f"-" * 60)

    # print(json.dumps(analyse_task(
    #     TaskState({'task': "Escalated to senior team for immediate action"})), indent=2))


def visualize_graph(graph: Graph, filename: str):
    with open(filename, mode="wb") as f:
        f.write(graph.draw_mermaid_png(background_color="white"))


if __name__ == '__main__':
    # demo_basic_routing()
    # demo_conditional_loop()
    # asyncio.run(main())
    demo_multi_path_routing()

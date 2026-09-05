from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
import operator

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph

from graph_utils import visualize_graph


load_dotenv()


"""
Cycles and Loops in LangGraph
Self-correcting agents and iterative refinement
"""

model = init_chat_model('gpt-5-nano')


def demo_self_correcting_code():
    """Self-correcting code generator"""
    class CodeGenState(TypedDict):
        task: str
        code: str
        errors: Annotated[list[str], operator.add]
        iteration: Annotated[int, operator.add]
        max_iterations: int
        success: bool

    def generate_code(state: CodeGenState) -> dict:
        if state['iteration'] == 0:
            # First prompt
            prompt = f"Write Python code for: {state['task']}\nReturn only the code, without any code fences or markers."
        else:
            # Correction attempt
            prompt = (
                f"Fix this Pything code:\n{state['code']}\n\n"
                f"Errors:\n{state['errors'][-1]}\n\n"
                f"Return only the corrected code."
            )
        response = model.invoke(prompt)
        code = response.content.strip()

        # Intentionally break the code on first iteration
        if state['iteration'] == 0:
            code = code.replace("isinstance", "is instance")

        return {
            "code": code,
            # "iteration": state["iteration"] + 1
        }

    def validate_code(state: CodeGenState) -> dict:
        code = state['code']

        try:
            # Try to compile the code
            compile(code, "<string>", "exec")
            return {"success": True}
        except SyntaxError as e:
            return {
                "errors": [f"SyntaxError: {e}"],
                "success": False
            }

    def should_continue(state: CodeGenState) -> Literal['generate', 'end']:
        if state['success']:
            return 'end'
        elif state['iteration'] >= state['max_iterations']:
            return 'end'
        else:
            return 'generate'

    def finalize(state: CodeGenState) -> dict:
        return state

    graph = StateGraph(CodeGenState)
    graph.add_node('generate', generate_code)
    graph.add_node('validate', validate_code)
    graph.add_node('finalize', finalize)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges("validate", should_continue, {
        "generate": "generate",
        "end": "finalize"
    })
    graph.add_edge("finalize", END)

    app = graph.compile()
    visualize_graph(app.get_graph(), "./self_correcting_code.png")
    result: CodeGenState = app.invoke(CodeGenState({
        "task": "a function that calculates factorial recursively",
        "code": "",
        "errors": [],
        "iteration": 0,
        "max_iterations": 3,
        "success": False,
    }))

    print(f"Task: {result['task']}")
    print(f"Iterations: {result['iteration']}")
    print(f"Success: {result['success']}")
    print(f"Final code:\n\n{result['code']}")


def broken_needs_structured_output__demo_iterative_research():
    """"Iterative research that goes deeper based on findings"""

    class ResearcherState(TypedDict):
        topic: str
        findings: Annotated[list[str], operator.add]
        questions: list[str]
        iteration: Annotated[int, operator.add]
        max_depth: int
        summary: str

    def research(state: ResearcherState) -> ResearcherState:
        if state["iteration"] == 0:
            query = f"Give me 3 key facts about: {state['topic']}"
        else:
            query = f"Based on these findings:\n{state['findings'][-1]}\n\nGo deeper"
        response = model.invoke(query)
        return ResearcherState(findings=[response.content])

    def generate_questions(state: ResearcherState) -> ResearcherState:
        response = model.invoke(
            f"Based on this finding:\n{state['findings'][-1]}\n\n"
            f"What's one deeper question to explore? Reply with just the question."
        )
        return {
            "questions": [response.content]
        }

    def synthesize(state: ResearcherState) -> ResearcherState:
        all_findings = "\n\n".join(state['findings'])
        response = model.invoke(
            f"Synthesize these findings into a coherent summary: \n\n{all_findings}"
        )
        return {"summary": response.content}

    def should_continue(state: ResearcherState) -> Literal['research', 'synthesize']:
        if state['iteration'] >= state['max_depth']:
            return "synthesize"
        else:
            return "research"

    graph = StateGraph(ResearcherState)
    graph.add_node('research', research)
    graph.add_node('generate_questions', generate_questions)
    graph.add_node('synthesize', synthesize)

    graph.add_edge(START, 'research')
    graph.add_edge('research', 'generate_questions')
    graph.add_conditional_edges('generate_questions', should_continue, {
        "synthesize": 'synthesize',
        "research": 'research'
    })
    graph.add_edge('synthesize', END)

    app = graph.compile()
    visualize_graph(app.get_graph(), "./research_graph.png")

    result: ResearcherState = app.invoke(ResearcherState(
        topic="quantum computing applications",
        findings=[],
        questions=[],
        iteration=0,
        max_depth=2,
        summary=""
    ))
    print(f"Topic: {result['topic']}")
    print(f"Iterations: {result['iteration']}")
    print(f"\nFindings collected: {len(result['findings'])}")
    print(f"\nFinal summary: {result['summary']}")


if __name__ == "__main__":
    # demo_self_correcting_code()
    # demo_iterative_research()
    pass

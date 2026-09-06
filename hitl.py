""""
Human in the loop (HITL) Patterns in LangGraph
Interrupt, review, modify and resume
"""

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model

from graph_utils import visualize_graph

load_dotenv()


model = init_chat_model('gpt-5-nano')


def demo_interrupt_for_approval():
    class ApprovalState(TypedDict):
        request: str
        draft: str
        approved: bool
        feedback: str
        final: str

    def create_draft(state: ApprovalState) -> ApprovalState:
        response = model.invoke(
            f"Create a professional response for: {state['request']}. Repond only with the response."
        )
        return {
            "draft": response.content
        }

    def wait_for_approval(state: ApprovalState) -> ApprovalState:
        # This node is where we'll interrupt
        return state

    def finalize(state: ApprovalState) -> ApprovalState:
        if state['approved']:
            return {
                "final": state['draft']
            }
        else:
            # Incorporate feedback
            response = model.invoke(
                f"Revise this draft based on feedback:\n\n"
                f"Draft: {state['draft']}"
                f"Feedback: {state['feedback']}"
            )
            return {
                "final": response.content
            }

    graph = StateGraph(ApprovalState)
    graph.add_node('draft', create_draft)
    graph.add_node('approval', wait_for_approval)
    graph.add_node('finalize', finalize)

    graph.add_edge(START, 'draft')
    graph.add_edge('draft', 'approval')
    graph.add_edge('approval', 'finalize')
    graph.add_edge('finalize', END)

    memory = MemorySaver()
    app = graph.compile(memory, interrupt_before=['approval'])

    visualize_graph(app.get_graph(), "./hitl.png")

    # Example usage

    # Configuration for this thread
    config = {"configurable": {"thread_id": "demo"}}

    # 1. Run until interrupt
    result = app.invoke(ApprovalState(
        request="Write a thank-you email for a job interview",
        draft="",
        approved=False,
        feedback="",
        final=""
    ), config=config)

    print(f"\nDraft created:\n{result['draft'][:200]}...")
    print(f"\n[ Execution paused for human review ]")

    # 2. Human reviews and provides feedback
    current_state = app.get_state(config)
    print(f"\nCurrent node: {current_state.next}")

    # 3. Simulate human feedback and continue
    print("\nStep 2: Human provides feedback and continues...")

    # Update state with human input
    app.update_state(config, ApprovalState(
        approved=False, feedback="Make it more concise and app specific mention of the company culture"))

    # Continue execution
    final_result: ApprovalState = app.invoke(None, config)

    print(f"\nFinal result:\n{final_result['final']}")


if __name__ == '__main__':
    demo_interrupt_for_approval()

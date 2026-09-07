"""
Parallel Agent Execution in LangGraph
Running multiple agents simultaneously
"""

from typing import Literal, TypedDict

from dedent import dedent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langsmith import traceable
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from graph_utils import visualize_graph

load_dotenv()


model = init_chat_model('gpt-5-nano')


@traceable(name='demo_parallel_agents', run_type="chain")
def demo_parallel_agents():
    class ParallelState(TypedDict):
        query: str
        research_result: str
        creative_result: str
        technical_result: str
        final_synthesis: str

    def parallel_research():
        """Three research agents working in parallel"""

        def research_agent(state: ParallelState) -> dict:
            """Academy/factual research"""
            response = model.invoke([
                SystemMessage(
                    "You are an academic researcher. Provide factual, well-sourced information."),
                HumanMessage(f"Research this topic: {state['query']}")
            ])
            return {
                "research_result": response.content
            }

        def creative_agent(state: ParallelState) -> dict:
            """Creative perspectives"""
            response = model.invoke([
                SystemMessage(
                    "You are a creative thinker. Provide novel perspectives and ideas."),
                HumanMessage(f"Give creative insights on: {state['query']}")
            ])
            return {
                "creative_result": response.content
            }

        def technical_agent(state: ParallelState) -> dict:
            """Technical analysis"""
            response = model.invoke([
                SystemMessage(
                    "You are a technical analyst. Provide practical, implementation-focused insights."),
                HumanMessage(f"Analyze technically: {state['query']}")
            ])
            return {
                "technical_result": response.content
            }

        def synthesize(state: ParallelState) -> dict:
            """Combine all perspectives"""
            synthesis_prompt = dedent(f"""
                  Synthesize these three perspectives into a comprehensive response:

                  RESEARCH: {state['research_result']}
                  CREATIVE: {state['creative_result']}
                  TECHNICAL: {state['technical_result']}
                  Create a unified, well-structured response.""")

            response = model.invoke([
                synthesis_prompt,
                HumanMessage(synthesis_prompt)
            ])
            return {
                "final_synthesis": response.content
            }

        graph = StateGraph(ParallelState)
        graph.add_node("research", research_agent)
        graph.add_node("creative", creative_agent)
        graph.add_node("technical", technical_agent)
        graph.add_node("synthesize", synthesize)

        # Fan-out: START foes to all three agents
        graph.add_edge(START, "research")
        graph.add_edge(START, "creative")
        graph.add_edge(START, "technical")

        # Fan-in
        graph.add_edge("research", "synthesize")
        graph.add_edge("creative", "synthesize")
        graph.add_edge("technical", "synthesize")

        graph.add_edge("synthesize", END)

        return graph.compile()

    agent = parallel_research()
    visualize_graph(agent.get_graph(), "./parallel_agents.png")

    """Demo parallel agent execution."""

    print("Parallel Agent Execution Demo:\n")

    result = agent.invoke(
        {
            "query": "The future of remote work",
            "research_result": "",
            "creative_result": "",
            "technical_result": "",
            "final_synthesis": "",
        }
    )

    print("Individual Perspectives:")
    print(f"\n[Research]\n{result['research_result'][:300]}...")
    print(f"\n[Creative]\n{result['creative_result'][:300]}...")
    print(f"\n[Technical]\n{result['technical_result'][:300]}...")

    print(f"\n{'='*50}")
    print(f"[SYNTHESIZED]\n{result['final_synthesis']}")


if __name__ == '__main__':
    demo_parallel_agents()

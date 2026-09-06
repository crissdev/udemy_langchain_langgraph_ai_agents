from typing import Literal, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langsmith import traceable
from typing_extensions import Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from graph_utils import visualize_graph

load_dotenv()


model = init_chat_model('gpt-5-nano')


@traceable(name="demo_tool_agent", run_type="chain")
def demo_tool_agent():
    @tool()
    def calculate(expression: str) -> str:
        """Calculate a mathematical expression. Example: calculat('2 + 2')"""
        try:
            result = eval(expression)
            return f"The result of {expression} is {result}"
        except Exception as e:
            return f"Error calculating: {e}"

    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        # Simulated weather data
        weather_data = {
            "new york": "72°F, Sunny",
            "london": "58°F, Cloudy",
            "tokyo": "68°F, Clear",
            "paris": "65°F, Partly Cloudy",
        }
        city_lower = city.lower()
        if city_lower in weather_data:
            return f"Weather in {city}: {weather_data[city_lower]}"
        return f"Weather data not available for {city}"

    @tool()
    def search_web(query: str) -> str:
        """Simulate a web search for a query"""
        search_results = {
            "python_programming": "Python is a high-level programming language know for its...",
            "latest news": "Today's top news: AI continues to advance, impacting various industries...",
            "best restaurants in new york": "Top restaurants in New York include Le Bernading, ..."
        }
        query_lower = query.lower()
        if query_lower in search_results:
            return f"Search results for '{query}': {search_results[query_lower]}"
        return f"No search results found for '{query}'"

    class AgentState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    def create_tool_agent():
        """Create a basic tool-calling agent"""

        tools = [calculate, get_weather, search_web]
        model_with_tools = model.bind_tools(tools)

        def agent_node(state: AgentState) -> AgentState:
            response = model_with_tools.invoke(state['messages'])
            return {
                "messages": [response]
            }

        def should_continue(state: AgentState) -> Literal["tools", "end"]:
            """Check if we should continue to 'tools' or 'end'"""
            last_message = state["messages"][-1]

            if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
                return "end"
            return "tools"

        tool_node = ToolNode(tools)

        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)

        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", should_continue, {
            "tools": "tools",
            "end": END
        })
        graph.add_edge("tools", "agent")  # Loop back after tool execution
        return graph.compile()

    agent = create_tool_agent()
    visualize_graph(agent.get_graph(), "./tool_calling_agent.png")

    queries = [
        "What's 25 * 17?",
        "What's the weather in Tokyo?",
        "What's 100 / 4 and what's the weather in London?"
    ]

    for query in queries:
        print(f"Query: {query}")
        result: AgentState = agent.invoke(AgentState(
            messages=[("user", query)]
        ))

        # Get final response
        final_message = result["messages"][-1]
        print(f"Response: {final_message.content}")
        print(f"Total messages: {len(result['messages'])}")
        print('-' * 40)


if __name__ == '__main__':
    demo_tool_agent()

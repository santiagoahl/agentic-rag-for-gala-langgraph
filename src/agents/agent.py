
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode

from typing import List, TypedDict, Literal
import os, getpass, sys
sys.path.append(os.path.abspath("src"))

from rag import RAGTool
from tools import search_tool, weather_tool
from IPython.display import display, Image

MAX_ITERATIONS = 3

# TODO: Migrate _get_var to utils script
def _get_var(var) -> None:
    if os.getenv(var):
        print(f"{var} successfully processed")
    else:
        os.environ[var] = getpass.getpass(prompt=f"Type the value of {var}: ")


_get_var("OPENAI_API_KEY")

# 1/4 State


class GalaState(TypedDict):
    messages: List[BaseMessage]
    intermediate_steps: list
    iterations: int
    done: bool


model = ChatOpenAI(model="gpt-3.5-turbo")
tools = [RAGTool, weather_tool, search_tool]
tools_node = ToolNode(tools)
model_with_tools = model.bind_tools(tools)


def agent(state: GalaState) -> GalaState:
    textual_description = """
    LLM model. Reasons and observes the GalaState to make a decision or prompt a response.

    Parameters
    ----------
    state : GalaState
        Information about the chat. It has the following keys: graph_state, chat_history, next_node
    
    Returns:
        type: GalaState
    
    Example:
        >>> agent({"messages": "Who is Batman"})
        Batman is a cool superhero
    """
    chat_history = state.get("messages", [])
    current_iterations = state.get("iterations", 0)

    formatted_messages = [
        {"role": "user" if msg.type == "human" else "assistant", "content": msg.content}
        for msg in chat_history
    ]
    response = model_with_tools.invoke(input=formatted_messages)

    result = {"messages": state.get("messages", []) + [response]}

    # Handle tool calls if they exist
    if hasattr(response, "tool_calls") and response.tool_calls:
        # This will trigger the tool execution in LangGraph
        return {"messages": chat_history + [response], "iterations": current_iterations + 1}

    # For regular responses
    return {
        "messages": chat_history + [response],
        **{k: v for k, v in state.items() if k != "messages"},  # Preserve other state
        "iterations": current_iterations + 1
    }


def should_use_tool(state: GalaState) -> Literal["tools", "end"]:
    """ "
    Determine whether to use tools based on the agent's last response
    """
    last_message = state.get("messages", [])[-1]
    current_agent_iterations = state.get("iterations", 0)
    
    # Finish flow is max number of iterations is reached
    if current_agent_iterations >= MAX_ITERATIONS:
        return "end"

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        next_node = "tools"
    else:
        next_node = "end"

    return next_node


# Build the graph

builder = StateGraph(state_schema=GalaState)

# Add nodes
builder.add_node("agent", agent)
builder.add_node("tools", tools_node)

# Add edges
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_use_tool, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")  # Agent processes tool results
builder.add_edge(
    "agent", END
)  # Define a direct edge to END after the agent runs (either initially or after tools)

agent_graph = builder.compile()


graph_image_bytes = agent_graph.get_graph().draw_mermaid_png()
with open("images/agent_architecture.png", "wb") as f:
    f.write(graph_image_bytes)


def test_app() -> None:
    prompt_1 = (
        "Hi. Tell me please Who is Ada and Which will be the story to tell about she?"
    )
    prompt_2 = "Hi. Tell me please Which was the result of the match Barcelona vs Inter today for the 24/25 Champions League Semifinal"
    prompt_3 = "Hi. Tell me please how's the weather today at Barcelona"
    response_1 = agent_graph.invoke({"messages": [HumanMessage(content=prompt_1)]}, {"recursion_limit": 10})
    print(response_1)


if __name__ == "__main__":
    test_app()

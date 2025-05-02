# TODO: re-structure files intro 3 main folders: 1) agents, 2) tools, 3) src

# Agent components
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage  # , AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode

# Observability
from langfuse.callback import CallbackHandler

# Utils
from typing import List, TypedDict, Literal
import os, sys

# from IPython.display import display, Image
sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("src/tools"))


# Local modules
from rag import RAGTool
from tools import rag_tool, search_tool, weather_tool
from utils import get_var

MAX_ITERATIONS = 3
get_var("OPENAI_API_KEY")

langfuse_callback_handler = CallbackHandler()


# Chat Prompt Template
system_role_message = ""
with open(
    "src/prompts/system_role_message.txt", "r"
) as f:  # TODO: Add Location file to include abs paths
    for line in f:
        system_role_message += line

prompt_messages = [
    ("system", system_role_message),
    # ("user", "Hello"),
    # ("ai", "Hello, how can I help you?"),
    ("user", "Think step by step. {input}"),
]
prompt_template = ChatPromptTemplate.from_messages(messages=prompt_messages)


# Graph State
class GalaState(TypedDict):
    messages: List[BaseMessage]
    intermediate_steps: list
    iterations: int
    done: bool


model = ChatOpenAI(model="gpt-3.5-turbo")
tools = [rag_tool, weather_tool, search_tool]
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
        return {
            "messages": chat_history + [response],
            "iterations": current_iterations + 1,
        }

    # For regular responses
    return {
        "messages": chat_history + [response],
        **{k: v for k, v in state.items() if k != "messages"},  # Preserve other state
        "iterations": current_iterations + 1,
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
with open("./data/images/agent_architecture.png", "wb") as f:
    f.write(graph_image_bytes)


def test_app() -> None:
    prompt_1 = "Tell me about Ada Lovelace and a story to share about her."
    prompt_2 = "Tell me please Which was the result of the match Barcelona vs Inter today for the 24/25 Champions League Semifinal"
    prompt_3 = "Tell me please how's the weather today at Barcelona"

    formatted_prompt = prompt_template.invoke({"input": prompt_1})
    response = agent_graph.invoke(
        input={
            "messages": [
                HumanMessage(content=formatted_prompt.to_messages()[1].content)
            ]
        },
        config={"callbacks": [langfuse_callback_handler]},
    )

    # Extract the agent's response
    agent_response = response["messages"][-1].content

    # Format and save the conversation
    chat_content = ""

    with open("./src/prompts/chat_history_summary.txt", "r") as chat_history_template:
        for line in chat_history_template:
            chat_content += line

    chat_content = chat_content.format(
        **{"prompt": prompt_1, "agent_response": agent_response}
    )

    with open("./data/examples/chat_1.md", "w") as chat_file:
        chat_file.write(chat_content)

    return agent_response


if __name__ == "__main__":
    test_app()

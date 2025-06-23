
from typing import Sequence     # Sequence is used to define a sequence of messages, such as a list or tuple, 
                                # automatically adding messages to the history
from typing import Annotated    # Annotated is used to add metadata to types, useful for tools in LangGraph
from typing import TypedDict    # TypedDict is used to define a dictionary with specific keys and types, useful for structured data 
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage # BaseMessage is the base class for all message types in LangChain, such as HumanMessage and AIMessage
from langchain_core.messages import ToolMessage  # ToolMessage is used to represent messages that invoke tools in LangChain
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display
import io


load_dotenv()


class AgentState(TypedDict):
    """ State of the agent, including the chat history and any other relevant information. """
    
    messages: Annotated[Sequence[BaseMessage], add_messages]  # List of messages in the conversation


@tool
def add_numbers(a: int, b: int) -> int:
    """ Adds two numbers together. """
    return a + b

tools = [add_numbers]

# llm = ChatOpenAI(
#    model="gpt-4o",
#    temperature=0.7,
#    max_tokens=2048,
#    top_p=0.9,  # Top-p sampling for controlling diversity  
#    n=1,  # Number of responses to generate
#    tools=tools,  # Register the tools with the model
#    tool_choice="auto",  # Automatically choose the best tool based on the input
#)

# OR alternatively, you can use the following line to register tools with the model:
# llm_with_tools = llm.bind_tools(tools) binds the tools to the LLM instance
# This lets you dynamically apply different tools to different instances derived from the same base model.

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=2048,
    top_p=0.9,  # Top-p sampling for controlling diversity  
    n=1,  # Number of responses to generate
).bind_tools(tools)  # Register the tools with the model

def model_call(state: AgentState) -> AgentState:
    """ Process the current state of the agent and return the updated state. """
    
    system_prompt = SystemMessage(
        content="You are a helpful assistant that can perform calculations and answer questions."
    )
    
    # Generate a response using the LLM
    # and then append the existing conversation history to the response
    response = llm.invoke([system_prompt] + state['messages'])
    
    # print(f"\n AI: {response.content}")
    # print("Current state:", state["messages"])
    
    return {"messages": [response]}  # Return the response as a new state with messages


def check_tool_call(state: AgentState) -> AgentState:
    """ Check if the last message is a tool call and return the state accordingly. """
    
    messages = state['messages']
    # if messages and isinstance(messages[-1], ToolMessage) and messages[-1].tool_call:  # This didn't work!!!
    if messages[-1].tool_calls: 
        # If the last message is a tool call, return the state with the tool call
        return "continue_tool_call"
    else:
        # If not, route to the end!
        return "end"
    
    
graph = StateGraph(AgentState)
graph.add_node("agent", model_call)  # Add the model call node

graph.set_entry_point("agent")  # Set the entry point of the graph

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)  # Add the tool call node

graph.add_conditional_edges(
    "agent",
    check_tool_call,  # Check if the last message is a tool call
    {
        "continue_tool_call": "tools",  # If the last message is a tool call, go to the tools node
        "end": END,  # Otherwise, go to the end
    },
)

graph.add_edge("tools", "agent")  # After the tool call, go back to the agent node to continue processing

app = graph.compile()  # Compile the graph into an app



# Assuming draw_mermaid_png() returns PNG bytes
# Uncomment to display the graph as a Mermaid diagram
# png_bytes = app.get_graph().draw_mermaid_png()
# display(Image(data=png_bytes))

# ------------

# Assuming draw_mermaid_png() returns PNG bytes
# png_bytes = app.get_graph().draw_mermaid_png()

# Save to a file
# with open("./images/agent_react_graph.png", "wb") as f:
#     f.write(png_bytes)

# Open the image (this will use the default image viewer on your OS)
# img = Image.open(io.BytesIO(png_bytes))
# img.show()  # This opens the image in a viewer window


def print_stream(stream):
    for ms in stream:
        message = ms['messages'][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()  # Pretty print the message if it's not a tuple 
        
inputs = {"messages": [("user", "Hello, can you add 5 and 10?")]}  
print_stream(app.stream(inputs, stream_mode="values"))  # Invoke the app with the inputs and print the stream
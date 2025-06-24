
from typing import Sequence     # Sequence is used to define a sequence of messages, such as a list or tuple, 
                                # automatically adding messages to the history
from typing import Annotated    # Annotated is used to add metadata to types, useful for tools in LangGraph
from typing import TypedDict    # TypedDict is used to define a dictionary with specific keys and types, useful for structured data 
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage # BaseMessage is the base class for all message types in LangChain, such as HumanMessage and AIMessage
from langchain_core.messages import ToolMessage  # ToolMessage is used to represent messages that invoke tools in LangChain
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from pprint import pprint


# Global variable to hold the document content
# This will be updated by the agent as it processes the document
document_content = ""


load_dotenv()

class AgentState(TypedDict):
    """ State of the agent, including the chat history and any other relevant information. """
    
    messages: Annotated[Sequence[BaseMessage], add_messages]  # List of messages in the conversation
    
    
@tool
def update(content: str) -> str:
    """ Updates the document content with the provided text. """
    
    global document_content
    document_content = content
    return f"Document updated with {len(content)} characters.\nCurrent content length: {len(document_content)} characters."

@tool
def save_document(filename: str) -> str:
    """ Saves the current document content to a file. 
    
    Args:
        filename (str): The name of the file to save the document content to.
        
    Returns:
        str: Confirmation message indicating the document has been saved.
    """
    
    global document_content
    
    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"
        
    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."
    
    except Exception as e:
        return f"Error saving document: {str(e)}"
        

tools = [update, save_document]  # List of tools that the agent can use

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=2048,
    top_p=0.9,  # Top-p sampling for controlling diversity  
    n=1,  # Number of responses to generate
    tool_choice="auto",  # Automatically choose the best tool based on the input
).bind_tools(tools)  # Register the tools with the model


def agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)
    
    if not state['messages']:
        user_input = "Hello, I need help with my document."
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document?\n")
        print("\n👤 User input:", user_input)
        user_message = HumanMessage(content=user_input)
        
    all_messages = [system_prompt] + list(state['messages']) + [user_message]
    
    response = model.invoke(all_messages)
    
    print(f"\n🤖 AI Bot: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
    
    return {"messages": list(state['messages']) + [user_message, response]}  # Append the new messages to the state

        
def check_tool_call(state: AgentState) -> AgentState:
    """Check whether there is a tool call involved or we should end the conversation."""

    messages = state['messages']
    
    if not messages:
        return "continue"
    
    # This looks for the most recent tool message....
    for message in reversed(messages):
        # and checks if this is a ToolMessage resulting from save
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end" # goes to the end edge
        
    return "continue"


def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")
 

tool_node = ToolNode(tools) 
            
graph = StateGraph(AgentState)

graph.add_node("agent", agent)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")


graph.add_conditional_edges(
    "tools",
    check_tool_call,  # Check if the last message is a tool call
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()  # Compile the graph into an app


def run_agent():
    print("Welcome to the Document Drafter Agent!"
          "\nYou can update and save your document content.")
    
    state = {"messages": []}  # Initialize the state with an empty message history
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\nThank you for using the Document Drafter Agent! Goodbye!")
    

# ###################    RUN AGENT   ################### #
    
if __name__ == "__main__":
    run_agent()  # Run the agent to start the conversation
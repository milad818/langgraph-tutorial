
from typing import Any, Dict, List, Optional, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
# from langchain_ollama import ChatOllama     # deprecated, use langchain_community instead
from langchain_community.chat_models import ChatOllama
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI


load_dotenv()

class AgentState(TypedDict):
    """ State of the agent, including the chat history and any other relevant information. """
    
    messages: List[HumanMessage]

    
# llm = ChatOllama(
#    model="llama3.1-70b",
#    temperature=0.7,
#    max_tokens=2048,
#    top_p=0.9,
#    n=1,
# )

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=2048,
    top_p=0.9,
    n=1,
)

def process(state: AgentState) -> AgentState:
    """ Process the current state of the agent and return the updated state. """
    
    # # Create a human message from the current state
    # human_message = HumanMessage(content=" ".join([msg.content for msg in state['messages']]))
    
    # # Append the human message to the messages list
    # state['messages'].append(human_message)
    
    # # Generate a response using the LLM
    # response = llm.invoke(state['messages'])
    
    # # Append the response to the messages list
    # state['messages'].append(response)
    
    response = llm.invoke(state['messages'])
    print(f"Response: {response.content}")
    
    return state

graph = StateGraph(AgentState)

graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = input("Enter your message (or type 'exit' to quit): ")

while user_input.lower() != "exit":
    # Initialize the agent state with the user input
    # Get the next user input
    user_input = input("Enter your message (or type 'exit' or 'quit' to quit): ")
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    
print("Exiting the agent bot. Goodbye! :wave:")



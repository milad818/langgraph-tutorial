
import os
from typing import List, Union, Dict, Any, Optional, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


class AgentState(TypedDict):
    """ State of the agent, including the chat history and any other relevant information. """
    
    messages: List[Union[HumanMessage, AIMessage]]  # List of messages in the conversation
    
    # memory: Dict[str, Any]  # Additional memory to store relevant information
    # user_input: Optional[str]  # User input for the next interaction
    
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=2048,
    top_p=0.9,
    n=1,
)
    
def process(state: AgentState) -> AgentState:
    """ Process the current state of the agent and return the updated state. """

    # Generate a response using the LLM
    response = llm.invoke(state['messages'])
    
    state['messages'].append(AIMessage(content=response.content))  # Append the response to the messages list
    print(f"\n AI: {response.content}")
    print("Current state:", state["messages"])
    
    return state
    

graph = StateGraph(AgentState)
graph.add_node("process", process) 
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile() 

conversation_history = []

user_input = input("Enter your message (or type 'exit' to quit): ")
while user_input.lower() != "exit":
    # Initialize the agent state with the user input
    conversation_history.append(HumanMessage(content=user_input))
    # agent_state = AgentState(messages=conversation_history)
    
    # Invoke the agent with the current state
    result = agent.invoke({"messages": conversation_history})
    # print(f"Agent response: {result['messages'][-1].content}")
    
    conversation_history = result['messages']  # Update the conversation history with the latest messages
    
    # Get the next user input
    user_input = input("Enter your message (or type 'exit' to quit): ")

with open("conversation_history.txt", "w") as f:
    f.write("Your Conversation History:\n\n=============================\n\n")
    
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"User: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n")
    f.write("\nEnd of the conversation!")
    
print("Your conversation history has been saved to 'conversation_history.txt'.")
    
    
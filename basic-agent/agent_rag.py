
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
from langchain_openai import OpenAIEmbeddings

import os
from langchain.document_loaders import PyPDFLoader  # Import the PDF loader to handle PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Import the text splitter to split documents into manageable chunks
from langchain.vectorstores import Chroma  # Import Chroma for vector storage and similarity search



load_dotenv()


llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=2048,
    top_p=0.9,  # Top-p sampling for controlling diversity  
    n=1,  # Number of responses to generate
)

# Define the embeddings model
# This model is used to convert text into vector embeddings for similarity search
# You can choose a different model based on your requirements
# For example, you can use "text-embedding-3-small" for smaller embeddings    
# or "text-embedding-3-large" for larger embeddings
# The choice of model affects the quality and size of the embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"    
)


pdf_path = "data/TAKE_A_STEP_BACK.pdf"  # Path to the PDF file containing the document to be processed


if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"The file {pdf_path} does not exist. Please check the path and try again.")    


pdf_loader = PyPDFLoader(pdf_path)  # Load the PDF file using PyPDFLoader   


try:
    pages = pdf_loader.load()  # Load the documents from the PDF file   
    print(f"Successfully loaded {len(pages)} pages from {pdf_path}.")  # Print the number of pages loaded
except Exception as e:
    raise RuntimeError(f"Failed to load documents from {pdf_path}. Error: {e}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Size of each text chunk
    chunk_overlap=200,  # Overlap between chunks to maintain context
    length_function=len  # Function to determine the length of the text
)


chunks = text_splitter.split_documents(pages)  # Split the loaded documents into smaller chunks

persist_directory = "./data/chromadb"  # Directory where the vector store will be persisted
collection_name = "take_a_step_back"  # Name of the collection in the vector store


if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)  # Create the directory if it does not exist 


try:
    vector_store = Chroma.from_documents(
        documents=chunks,  # The chunks of text to be stored
        embedding=embeddings,  # The embeddings model used to convert text into vectors
        persist_directory=persist_directory,  # Directory to persist the vector store
        collection_name=collection_name  # Name of the collection in the vector store
    )
    print(f"Vector store created and persisted at {persist_directory} with collection name '{collection_name}'.")
except Exception as e:
    raise RuntimeError(f"Failed to create vector store. Error: {e}")


retriever = vector_store.as_retriever(
    search_type="similarity",  # Type of search to perform, here it is similarity search
    search_kwargs={"k": 3}  # Number of documents to retrieve based on similarity
)


@tool
def retriever_tool(
    query: Annotated[str, "The query to search for in the document."]
) -> str:
    """
    Retrieve relevant documents from the vector store based on the query.
    
    Args:
        query (str): The query to search for in the document.
        
    Returns:
        str: A string containing the retrieved documents.
    """
    
    # Use the retriever to get relevant documents based on the query
    docs = retriever.invoke(query)
    
    if not docs:
        return "No relevant documents found."
    
    results = []
    for i, doc in enumerate(docs):
        # Format the retrieved documents for better readability
        results.append(f"Document {i + 1}:\n{doc.page_content}\n")
        
    return "\n".join(results)  # Join the results into a single string for output


tools = [retriever_tool]  # List of tools
llm_with_tools = llm.bind_tools(tools)  # Bind the tools to the LLM instance    


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]  # List of messages in the conversation
    

def check_tool_call(state: AgentState) -> str:
    """
    Check if a tool the called.
    
    Args:
        state (AgentState): The current state of the agent.
        
    Returns:
        bool: True if the last message is a tool call, False otherwise.
    """

    if not state['messages']:
        return "No messages in the conversation."
    
    last_message = state['messages'][-1]  # Get the last message in the conversation
    
    return hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0  # Check if the last message is a tool call 


system_prompt = """
You are a knowledgeable AI assistant tasked with answering questions about a scientific paper titled **Take a Step Back**, using the loaded PDF document as your primary source.
Use the retriever tool to find relevant information about Large Language Models, making as many retrievals as necessary.
Feel free to perform lookups before asking follow-up questions if it helps clarify or improve your response.
Always provide precise citations from the document to support your answers.
"""


tools_dict = {tool.name: tool for tool in tools}  # Create a dictionary of tools for easy access

def call_llm(state: AgentState) -> AgentState:
    """
    Call the LLM with the current state and return the updated state.
    
    Args:
        state (AgentState): The current state of the agent.
        
    Returns:
        AgentState: The updated state after processing the LLM response.
    """
    
    messages = list(state['messages'])  # Create a copy of the messages in the state
    messages = [SystemMessage(content=system_prompt)] + messages  # Prepend the system prompt to the messages   
    message = llm.invoke(messages)  # Generate a response using the LLM 
    
    return {'messages': [message]}  # Return the updated state with the new message


def action(state: AgentState):
    """ Execute the tool calls requested by the LLM's response. """
    
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", action)

graph.add_conditional_edges(
    "llm",
    check_tool_call,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


if __name__ == "__main__":
    running_agent()
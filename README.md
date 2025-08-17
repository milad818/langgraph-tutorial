# LangGraph Tutorial

A comprehensive tutorial series for learning LangGraph, the framework for building stateful, multi-actor applications with Large Language Models (LLMs). This repository contains hands-on examples ranging from simple "Hello World" agents to complex looping and conditional graphs which is mainly inspired by [LangGraph Complete Course for Beginners](https://www.youtube.com/watch?v=jGg_1h0qzaM&list=PLJ--RI9AhXQLsIsAiM4UWJjI6bkMoIJwX&index=24&t=10879s) introduced by freeCodeCamp on their YouTube channel.

## 📖 Table of Contents

- [About LangGraph](#about-langgraph)
- [Repository Structure](#repository-structure)
- [Tutorial Notebooks](#tutorial-notebooks)
- [Basic Agent Examples](#basic-agent-examples)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Key Concepts Covered](#key-concepts-covered)
- [Contributing](#contributing)
- [License](#license)

## 🤖 About LangGraph

LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain by providing a way to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner. This is particularly useful for agent-like behaviors where you need to iterate between planning, acting, and reasoning.

## 📁 Repository Structure

```
langgraph-tutorial/
├── tutorial-notebooks/
│   ├── 00_hello_world_agent.ipynb
│   ├── 01_multiple_input_graph.ipynb
│   ├── 01ex_singlenode_multioperational_graph.ipynb
│   ├── 02_multiplenode_sequential_graph.ipynb
│   ├── 02ex_multiplenode_sequential_graph.ipynb
│   ├── 03_conditional_graph.ipynb
│   ├── 03ex_conditional_graph.ipynb
│   └── 04_looping_graph.ipynb
├── basic-agent/
│   ├── agent_bot.py
│   ├── agent_doc_drafter.py
│   ├── agent_rag.py
│   ├── agent_react.py
│   ├── agent_with_memory.py
│   ├── data/
│   ├── images/
│   └── conversation_history.txt
├── LICENSE
├── README.md
```

## 📚 Tutorial Notebooks

The notebooks are designed to be followed sequentially, building complexity gradually:

### Core Concepts Series
1. **00_hello_world_agent.ipynb** - Your first LangGraph agent
   - Basic StateGraph creation
   - Simple node functions
   - Graph compilation and execution

2. **01_multiple_input_graph.ipynb** - Handling multiple data inputs
   - Multi-input node processing
   - State management with complex data types
   - Data flow between nodes

3. **02_multiplenode_sequential_graph.ipynb** - Sequential processing
   - Creating multi-node workflows
   - Sequential data processing
   - State passing between nodes

4. **03_conditional_graph.ipynb** - Dynamic routing and decision making
   - Conditional node execution
   - Dynamic path selection
   - Logic-based control flow

5. **04_looping_graph.ipynb** - Iterative and cyclical workflows
   - Creating loops in graphs
   - Iterative refinement patterns
   - Exit conditions and cycle control

### Exercise Notebooks
- **01ex_singlenode_multioperational_graph.ipynb**
- **02ex_multiplenode_sequential_graph.ipynb** 
- **03ex_conditional_graph.ipynb**

## 🤖 Basic Agent Examples

The `basic-agent/` directory contains practical implementations:

- **agent_bot.py** - A conversational AI agent with chat capabilities
- **agent_doc_drafter.py** - An agent for drafting and refining documents
- **agent_rag.py** - Retrieval-Augmented Generation implementation
- **agent_react.py** - ReAct (Reasoning + Acting) pattern agent
- **agent_with_memory.py** - Stateful agent that maintains conversation history

## 🔧 Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Basic understanding of Python and LLMs

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd langgraph-tutorial
   ```

2. **Install dependencies:**
   ```bash
   pip install langgraph langchain langchain-openai langchain-community python-dotenv
   ```

3. **Set up environment variables:**
   Create a `.env` file in the root directory:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Start with the tutorial notebooks:**
   ```bash
   cd tutorial-notebooks
   jupyter notebook 00_hello_world_agent.ipynb
   ```

5. **Explore the basic agents:**
   ```bash
   cd ../basic-agent
   python agent_bot.py
   ```

## 🎯 Key Concepts Covered

- **State Management**: Learn how to define and manage state across graph nodes
- **Node Functions**: Create and combine different types of processing nodes
- **Graph Compilation**: Understand how to compile and execute LangGraph workflows
- **Conditional Logic**: Implement dynamic routing and decision-making
- **Loops and Iteration**: Build iterative workflows with exit conditions
- **Memory and Persistence**: Maintain state across conversations and sessions
- **Tool Integration**: Connect external tools and APIs to your agents
- **ReAct Pattern**: Implement reasoning and acting cycles
- **RAG Systems**: Build retrieval-augmented generation workflows

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

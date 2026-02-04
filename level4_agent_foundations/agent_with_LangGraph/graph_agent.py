print("RUNNING graph_agent.py")

# graph_agent.py
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, AIMessage

# ✅ NODE (Pillar 1): takes state, returns updates to state
def mock_llm(state: MessagesState):
    # Add an AI message to the conversation
    return {"messages": [AIMessage(content="hello world")]}

# ✅ GRAPH
graph = StateGraph(MessagesState)

# add node
graph.add_node("mock_llm", mock_llm)

# ✅ EDGES (Pillar 3)
graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", END)

# compile
app = graph.compile()

if __name__ == "__main__":
    # ✅ STATE (Pillar 2): MessagesState expects message objects
    result = app.invoke({"messages": [HumanMessage(content="hi")]})

    # result["messages"] contains HumanMessage + AIMessage objects
    print(result["messages"][-1].content)  # -> hello world

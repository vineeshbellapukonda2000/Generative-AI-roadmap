# conditional_agent.py
from typing import TypedDict, Literal, List, Optional
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# -------------------------
# 1) STATE (Pillar 2)
# -------------------------
class AgentState(TypedDict):
    messages: List[BaseMessage]
    memory: str
    tool_result: Optional[str]
    next_step: Literal["tool", "final"]


# -------------------------
# 2) NODES (Pillar 1)
# -------------------------
def ingest_user(state: AgentState) -> AgentState:
    # In real apps, user input is already in state["messages"]
    # This node is here to show the structure.
    return state


def attach_memory(state: AgentState) -> AgentState:
    # Fake memory injection (in real life: load from DB / cache / summary store)
    # Here we keep it constant just to show where it happens.
    state["memory"] = state.get("memory") or "User likes concise answers."
    return state


def reason_node(state: AgentState) -> AgentState:
    """
    This simulates the LLM Reason/Plan step.
    - Reads messages + memory + tool_result
    - Decides whether to call tool or finalize
    """

    user_text = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            user_text = m.content
            break

    # If we already have a tool_result, we can usually finalize.
    if state.get("tool_result"):
        state["next_step"] = "final"
        return state

    # Simple decision rule (simulate LLM decision):
    # If the question looks like it needs fresh info, call the tool.
    needs_tool_keywords = ["latest", "current", "today", "news", "price", "weather"]
    if any(k in user_text.lower() for k in needs_tool_keywords):
        state["next_step"] = "tool"
    else:
        state["next_step"] = "final"

    return state


def tool_executor(state: AgentState) -> AgentState:
    """
    ReAct 'Act' step.
    In real life, this node calls Tavily/Google/DB/API.
    Here, we simulate a tool result.
    """
    # Fake tool output
    state["tool_result"] = "SIMULATED_TOOL_RESULT: Found relevant up-to-date info."
    return state


def observe_tool(state: AgentState) -> AgentState:
    """
    ReAct 'Observe' step.
    Usually: append tool output into messages so the LLM can see it next time.
    """
    tool_text = state.get("tool_result") or ""
    state["messages"].append(ToolMessage(content=tool_text, tool_call_id="tool_call_1"))
    return state


def finalize_answer(state: AgentState) -> AgentState:
    # Ask the LLM using full conversation state
    response = llm.invoke(state["messages"])

    # Append real AI response to state
    state["messages"].append(response)

    return state


# -------------------------
# 3) CONDITIONAL ROUTER (Edges logic)
# -------------------------
def route_after_reason(state: AgentState) -> str:
    # This function reads state and tells LangGraph which node to run next
    return "tool_executor" if state["next_step"] == "tool" else "finalize_answer"


# -------------------------
# 4) BUILD GRAPH (Pillar 3 = edges)
# -------------------------
def build_app():
    graph = StateGraph(AgentState)

    graph.add_node("ingest_user", ingest_user)
    graph.add_node("attach_memory", attach_memory)
    graph.add_node("reason_node", reason_node)
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("observe_tool", observe_tool)
    graph.add_node("finalize_answer", finalize_answer)

    # Normal edges
    graph.add_edge(START, "ingest_user")
    graph.add_edge("ingest_user", "attach_memory")
    graph.add_edge("attach_memory", "reason_node")

    # Conditional edges (the key learning today)
    graph.add_conditional_edges(
        "reason_node",
        route_after_reason,
        {
            "tool_executor": "tool_executor",
            "finalize_answer": "finalize_answer",
        },
    )

    # Tool path continues, then loops back to reason_node
    graph.add_edge("tool_executor", "observe_tool")
    graph.add_edge("observe_tool", "reason_node")

    # Final path ends
    graph.add_edge("finalize_answer", END)

    return graph.compile()


# -------------------------
# 5) RUN
# -------------------------
if __name__ == "__main__":
    app = build_app()

    # Try TWO questions to see branching:
    tests = [
        "explain me what is GMT time.",
        "What is the best coffe vote among the world?"
    ]

    for q in tests:
        print("\n==============================")
        print("USER:", q)

        init_state: AgentState = {
            "messages": [HumanMessage(content=q)],
            "memory": "",
            "tool_result": None,
            "next_step": "final",
        }

        out = app.invoke(init_state)
        print("AI:", out["messages"][-1].content)

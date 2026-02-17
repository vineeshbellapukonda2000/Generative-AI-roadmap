# react_langgraph_tavily.py
from __future__ import annotations

import os
from typing import List, Optional, Literal, TypedDict

from dotenv import load_dotenv
load_dotenv()


from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
)
from langchain_community.tools.tavily_search import TavilySearchResults


# -------------------------
# 0) Load env vars
# -------------------------
# This tries to load .env from the SAME folder as this file.
HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))


# -------------------------
# 1) State (shared memory)
# -------------------------
class AgentState(TypedDict):
    messages: List[BaseMessage]
    tool_calls: int
    max_tool_calls: int
    next_step: Literal["tool", "final"]
    tool_query: Optional[str]


# -------------------------
# 2) LLM + Tool
# -------------------------
# Use a safe default model. You can change it later.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Real web search tool (requires TAVILY_API_KEY)
search_tool = TavilySearchResults(max_results=5)


# -------------------------
# 3) Nodes
# -------------------------
def reason_node(state: AgentState) -> AgentState:
    """
    ReAct: REASON
    Decide: do we need web search or can we answer directly?
    If tool is needed, produce a search query (tool_query).
    """

    # Guardrail: if we've already used tools too many times, force final
    if state["tool_calls"] >= state["max_tool_calls"]:
        state["next_step"] = "final"
        state["tool_query"] = None
        return state

    system = SystemMessage(
        content=(
            "You are a careful assistant.\n"
            "Decide whether the user question requires live / up-to-date information.\n"
            "If it requires live info (news, 'today', prices, current events, releases), choose TOOL.\n"
            "If it can be answered from general knowledge, choose FINAL.\n\n"
            "Return exactly one line in this format:\n"
            "DECISION: TOOL | QUERY: <search query>\n"
            "or\n"
            "DECISION: FINAL | QUERY: NONE\n"
        )
    )

    # Ask the LLM to choose
    resp = llm.invoke([system] + state["messages"])
    text = (resp.content or "").strip()

    # Very strict parsing to keep it reliable
    # Default to FINAL if parsing fails
    decision = "FINAL"
    query = None

    if "DECISION:" in text:
        try:
            # Example: "DECISION: TOOL | QUERY: ... "
            parts = [p.strip() for p in text.split("|")]
            dec_part = parts[0].split("DECISION:")[1].strip().upper()
            decision = dec_part

            if decision == "TOOL":
                q_part = parts[1].split("QUERY:")[1].strip()
                query = q_part if q_part and q_part.upper() != "NONE" else None
        except Exception:
            decision = "FINAL"
            query = None

    state["next_step"] = "tool" if decision == "TOOL" and query else "final"
    state["tool_query"] = query
    return state


def tool_executor(state: AgentState) -> AgentState:
    """
    ReAct: ACT
    Call Tavily search with the tool_query, store results in a ToolMessage.
    """
    query = state.get("tool_query") or ""
    if not query:
        # If no query, just skip to final
        state["next_step"] = "final"
        return state

    results = search_tool.invoke({"query": query})

    # Store tool output as a ToolMessage (what the LLM will "observe")
    tool_text = str(results)

# ✅ Put tool output into normal chat context (no ToolMessage)
    state["messages"].append(SystemMessage(content=f"Tool result:\n{tool_text}"))

    state["tool_calls"] += 1
    return state

def finalize_answer(state: AgentState) -> AgentState:
    """
    Final answer: use messages (including tool results if present) to produce user-facing response.
    """
    system = SystemMessage(
        content=(
            "You are a helpful assistant.\n"
            "If tool results are present, use them to answer accurately.\n"
            "If tool results are not present, answer directly.\n"
            "Keep the answer clear and concise.\n"
        )
    )
    resp = llm.invoke([system] + state["messages"])
    state["messages"].append(AIMessage(content=resp.content))
    return state


# -------------------------
# 4) Routing (Conditional Edges)
# -------------------------
def route_after_reason(state: AgentState) -> str:
    return "tool_executor" if state["next_step"] == "tool" else "finalize_answer"


# -------------------------
# 5) Build Graph
# -------------------------
def build_app():
    g = StateGraph(AgentState)

    g.add_node("reason_node", reason_node)
    g.add_node("tool_executor", tool_executor)
    g.add_node("finalize_answer", finalize_answer)

    g.add_edge(START, "reason_node")

    # Conditional edge: tool vs final
    g.add_conditional_edges(
        "reason_node",
        route_after_reason,
        {
            "tool_executor": "tool_executor",
            "finalize_answer": "finalize_answer",
        },
    )

    # Loop back after tool execution
    g.add_edge("tool_executor", "reason_node")

    # End after final answer
    g.add_edge("finalize_answer", END)

    return g.compile()


# -------------------------
# 6) Run (Automated tests)
# -------------------------
if __name__ == "__main__":
    # Quick key checks (helps avoid silent failures)
    print("OPENAI_API_KEY loaded?", bool(os.getenv("OPENAI_API_KEY")))
    print("TAVILY_API_KEY loaded?", bool(os.getenv("TAVILY_API_KEY")))

    app = build_app()

    tests = [
        "Who won the latest grand slam, what is the next grand slam and when is it, and who are the top 5 in the world right now?",
        "What are the top rises in stocks today?"
        "What is the current stock price of NVIDIA right now?"
        "What happenend in AI news in the last 24 hours ",
    ]

    for q in tests:
        print("\n==============================")
        print("USER:", q)

        init_state: AgentState = {
            "messages": [HumanMessage(content=q)],
            "tool_calls": 0,
            "max_tool_calls": 2,   # guardrail: no infinite loops
            "next_step": "final",
            "tool_query": None,
        }

        out = app.invoke(init_state)
        print("AI:", out["messages"][-1].content)

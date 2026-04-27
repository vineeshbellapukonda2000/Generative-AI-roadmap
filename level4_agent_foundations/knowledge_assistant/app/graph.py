from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from app.rag.retrieve import retrieve_chunk_objs


class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    standalone_question: Optional[str]


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def rewrite_query_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    latest_question = messages[-1]["content"]

    if len(messages) <= 1:
        state["standalone_question"] = latest_question
        return state

    history = "\n".join(
        [f"{m['role']}: {m['content']}" for m in messages[:-1]]
    )

    prompt = f"""
Rewrite the user's latest question into a standalone question.

Conversation history:
{history}

Latest question:
{latest_question}

Return ONLY the rewritten question.
""".strip()

    response = llm.invoke(prompt)
    state["standalone_question"] = response.content.strip()


    return state


def rag_answer_node(state: AgentState) -> AgentState:
    user_question = state["messages"][-1]["content"]
    query = state.get("standalone_question") or user_question

    retrieved = retrieve_chunk_objs(query, k=3)

    if not retrieved:
        state["messages"].append(
            {
                "role": "assistant",
                "content": "I couldn’t find anything relevant in your documents for that question.",
            }
        )
        return state

    context = "\n\n---\n\n".join([r["text"] for r in retrieved])

    sources = []
    source_map = {}

    for r in retrieved:
        src = r["source"]
        if src not in source_map:
            index = len(source_map) + 1
            source_map[src] = index
            sources.append(src)

    prompt = f"""
You are a helpful RAG assistant.

Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't know."

Keep the answer:
- clear
- concise
- well-structured

Do not add information outside the context.

Context:
{context}

User Question:
{user_question}
""".strip()

    response = llm.invoke(prompt)

    formatted_sources = "\n".join(
        [f"[{source_map[s]}] {s}" for s in sources]
    )

    final_answer = response.content + "\n\nSources:\n" + formatted_sources

    state["messages"].append(
        {
            "role": "assistant",
            "content": final_answer,
        }
    )

    return state


def build_app():
    graph = StateGraph(AgentState)

    graph.add_node("rewrite_query_node", rewrite_query_node)
    graph.add_node("rag_answer_node", rag_answer_node)

    graph.set_entry_point("rewrite_query_node")

    graph.add_edge("rewrite_query_node", "rag_answer_node")
    graph.add_edge("rag_answer_node", END)

    memory = MemorySaver()

    return graph.compile(checkpointer=memory)
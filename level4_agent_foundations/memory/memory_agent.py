# memory_agent.py
# Week 2 – Day 2: Short-term memory for chat and an Agent with memory
# Run:  python memory_agent.py

print(">>> running memory_agent.py")

from pathlib import Path
import os

from dotenv import load_dotenv

# Always load .env from the SAME folder as this file (bulletproof)
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

# Quick sanity check
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY not found. Make sure a .env file exists next to memory_agent.py "
        "and contains OPENAI_API_KEY=..."
    )

from langchain_openai import ChatOpenAI

# Option A: simple memory chat (works reliably)
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory


# ---------------------------
# Option A — Simple chat with memory (clean + stable)
# ---------------------------
def run_simple_memory_chat():
    """
    Minimal, stable "memory" demo.
    Keeps chat history in memory and sends the full history each time.
    """
    print("\n=== Option A: Simple Chat with Memory ===")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    history = InMemoryChatMessageHistory()

    # Scripted demo
    print("\n--- Scripted demo ---")
    user_1 = "My name is Vineesh. Remember that."
    history.add_user_message(user_1)
    ai_1 = llm.invoke(history.messages)
    history.add_ai_message(ai_1.content)
    print("Assistant:", ai_1.content)

    user_2 = "What is my name?"
    history.add_user_message(user_2)
    ai_2 = llm.invoke(history.messages)
    history.add_ai_message(ai_2.content)
    print("Assistant:", ai_2.content)

    user_3 = "Yesterday you suggested a low-caffeine coffee; which one was it? (make something up if needed)"
    history.add_user_message(user_3)
    ai_3 = llm.invoke(history.messages)
    history.add_ai_message(ai_3.content)
    print("Assistant:", ai_3.content)

    # Interactive
    print("\n--- Interactive (press Enter on empty line to exit) ---")
    while True:
        user = input("You: ").strip()
        if not user:
            print("[Exiting Option A]")
            break

        history.add_user_message(user)
        ai = llm.invoke(history.messages)
        history.add_ai_message(ai.content)
        print("Assistant:", ai.content)


# ---------------------------
# Option B — Agent with memory + a tiny tool (LangGraph tool-calling)
# ---------------------------
def keyword_extractor(text: str) -> str:
    import re
    from collections import Counter

    words = re.findall(r"[A-Za-z][A-Za-z\\-']{2,}", text.lower())
    common = [w for w, _ in Counter(words).most_common(8)]
    return "Keywords: " + ", ".join(common)


def run_agent_with_memory():
    """
    Modern replacement for initialize_agent + AgentType:
    Use LangGraph + create_react_agent with a memory checkpoint.
    """
    print("\n=== Option B: Agent with Tool + Memory (LangGraph) ===")

    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    @tool
    def KeywordExtractor(text: str) -> str:
        "Extract top keywords from the given text."
        return keyword_extractor(text)

    tools = [KeywordExtractor]

    # Memory store for the agent
    checkpointer = MemorySaver()

    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=checkpointer,
    )

    # IMPORTANT: thread_id keeps memory across turns
    config = {"configurable": {"thread_id": "vineesh_memory_demo"}}

    def ask(q: str):
        result = agent.invoke({"messages": [HumanMessage(content=q)]}, config=config)
        return result["messages"][-1].content

    # Scripted demo
    print("\n--- Scripted demo ---")
    print("> You: My favorite topic is specialty coffee beans from Ethiopia.")
    print("Assistant:", ask("My favorite topic is specialty coffee beans from Ethiopia."))

    print("> You: Extract some keywords from my preference.")
    print("Assistant:", ask("Extract some keywords from my preference."))

    print("> You: Based on what I said earlier, suggest a café style I might like.")
    print("Assistant:", ask("Based on what I said earlier, suggest a café style I might like."))

    # Interactive
    print("\n--- Interactive (press Enter on empty line to exit) ---")
    while True:
        user = input("You: ").strip()
        if not user:
            print("[Exiting Option B]")
            break
        print("Assistant:", ask(user))


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # Start with Option A (most stable)
    run_simple_memory_chat()

    # Uncomment this after Option A works
    # run_agent_with_memory()
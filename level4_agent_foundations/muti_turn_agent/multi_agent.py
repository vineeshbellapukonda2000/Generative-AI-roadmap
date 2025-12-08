# multi_agent.py
# Level 4 → Week 2 → Day 4
# Multi-turn agent with:
# - OpenAI LLM (reasoning)
# - Memory (conversation summary)
# - Tools (web search, calculator, time)
# - ReAct loop (Reason → Act → Observe → Answer)

# Requirements:
#   pip install langchain langchain-openai python-dotenv tavily-python
# .env must contain:
#   OPENAI_API_KEY=sk-...
#   TAVILY_API_KEY=tvly-...

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationSummaryMemory
from tavily import TavilyClient
from datetime import datetime
import math


# ============================================================
# 1️⃣ LLM (the brain)
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0  # deterministic, good for reasoning
)


# ============================================================
# 2️⃣ Memory (multi-turn, summarized)
# ============================================================

# ConversationSummaryMemory:
# - Compresses long history into a running summary
# - Keeps key facts, preferences, and context
memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True
)


# ============================================================
# 3️⃣ Tools
# ============================================================

# --- A) Web search (live info) ---
def tavily_search_tool(query: str) -> str:
    """
    Search the live web and return compact, deduped snippets.
    """
    client = TavilyClient()
    res = client.search(query, max_results=6)
    seen = set()
    snippets = []

    for r in res.get("results", []):
        url = (r.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)

        title = (r.get("title") or url).strip()
        content = (r.get("content") or "").strip()

        # Trim very long text
        if len(content) > 500:
            content = content[:500] + " …"

        snippets.append(f"{title}\n{url}\n{content}")
        if len(snippets) >= 3:
            break

    return "\n\n".join(snippets) if snippets else "No relevant results found."


# --- B) Calculator tool (safe-ish) ---
def safe_calculator(expression: str) -> str:
    """
    Evaluate basic math expressions safely.
    Examples:
      "2 + 3 * 4"
      "math.sqrt(16)"
    """
    try:
        allowed_globals = {"__builtins__": {}, "math": math}
        result = eval(expression, allowed_globals, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


# --- C) Current date & time ---
def current_datetime(_: str = "") -> str:
    """
    Return the current date and time as a human-friendly string.
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


tools = [
    Tool(
        name="WebSearch",
        func=tavily_search_tool,
        description=(
            "Use this to search the live web for up-to-date information, news, "
            "facts, or events. Input: a clear search query."
        ),
    ),
    Tool(
        name="Calculator",
        func=safe_calculator,
        description=(
            "Use this to perform math calculations. Input should be a valid "
            "Python math expression, e.g. '2 + 3*4' or 'math.sqrt(16)'."
        ),
    ),
    Tool(
        name="CurrentDateTime",
        func=current_datetime,
        description="Use this to get the current date and time. Input can be empty.",
    ),
]


# ============================================================
# 4️⃣ Agent (ReAct + tools + memory)
# ============================================================

SYSTEM_GUIDELINES = (
    "You are a helpful multi-turn assistant. "
    "You have tools: WebSearch, Calculator, CurrentDateTime.\n"
    "- First, think if you actually need a tool.\n"
    "- Use WebSearch only when you need fresh or external information.\n"
    "- Use Calculator for numeric/math tasks.\n"
    "- Use CurrentDateTime only when the user asks about date or time.\n"
    "- Keep answers short, clear, and in natural language.\n"
    "- Remember important user preferences and context via chat_history."
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # ReAct-style agent
    memory=memory,
    verbose=True,           # Shows Thought / Action / Observation
    max_iterations=4,       # Limit ReAct loops
    early_stopping_method="generate",
)


# ============================================================
# 5️⃣ Interactive multi-turn loop
# ============================================================

if __name__ == "__main__":
    print("=== Multi-turn Agent (ReAct + Memory + Tools) ===")
    print("You can chat normally. The agent will remember context and use tools when needed.")
    print("Press Enter on an empty line to exit.\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                break

            # Prefix system guidelines so the agent behaves consistently
            full_input = f"{SYSTEM_GUIDELINES}\n\nUser: {user_input}"
            answer = agent.run(full_input)
            print("\nAssistant:", answer, "\n")
    except KeyboardInterrupt:
        print("\n[Exiting]")

# ============================================================
# 5️⃣ Interactive multi-turn loop
# ============================================================

if __name__ == "__main__":
    print("=== Multi-turn Agent (ReAct + Memory + Tools) ===")
    print("You can chat normally. The agent will remember context and use tools when needed.")
    print("Press Enter on an empty line to exit.\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                break

            # Prefix system guidelines so the agent behaves consistently
            full_input = f"{SYSTEM_GUIDELINES}\n\nUser: {user_input}"
            answer = agent.run(full_input)
            print("\nAssistant:", answer, "\n")
    except KeyboardInterrupt:
        print("\n[Exiting]")

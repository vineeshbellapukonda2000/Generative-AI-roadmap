# tool_agent2.py
# Level 4 → Week 2 → Day 3.5 : Agent with Memory + Clarification + Real Search (General)
# Requirements:
#   pip install langchain langchain-openai python-dotenv tavily-python

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from tavily import TavilyClient

# ============================================================
# 1️⃣ Universal Rules and Helpers
# ============================================================

CLARITY_RULES = (
    "Universal Rules:\n"
    "1) If the user question is AMBIGUOUS, ask ONE clarifying question first.\n"
    "   Ambiguity types: missing entity (who/what?), missing location/region, missing timeframe, missing units,\n"
    "   unclear pronouns (they/it/this), or vague scope (best, latest, near me, recently).\n"
    "2) If the question is CLEAR, rewrite a focused web-search query that includes explicit entities, timeframe,\n"
    "   and location/region if relevant.\n"
    "3) Perform at most 2 web searches per question. If results repeat, stop searching.\n"
    "4) Produce a short, neutral final answer (2–4 sentences). Then list 2–4 sources as 'Title — URL'.\n"
    "5) If the user clarifies, continue with one search and answer.\n"
)

def rewrite_for_search(llm: ChatOpenAI, user_q: str) -> dict:
    """
    Returns either:
      {"clarify": "<ask this>"}  OR
      {"query": "<rewritten search query>"}
    Works for ANY topic.
    """
    prompt = (
        f"{CLARITY_RULES}\n\n"
        "Given the user's question below, decide if it's ambiguous.\n"
        "- If ambiguous, return JSON: {\"clarify\": \"<ask this>\"}\n"
        "- If clear, return JSON: {\"query\": \"<rewritten search query with explicit entity/time/location>\"}\n\n"
        "User question:\n"
        f"{user_q}\n"
    )
    msg = llm.invoke(prompt).content.strip()
    if '"clarify"' in msg:
        start = msg.find('{'); end = msg.rfind('}')
        return {"clarify": msg[start:end+1].split(':',1)[1].strip().strip(' "}')}
    start = msg.find('{'); end = msg.rfind('}')
    return {"query": msg[start:end+1].split(':',1)[1].strip().strip(' \"}')}

# ============================================================
# 2️⃣ Search Tool (Tavily)
# ============================================================

def tavily_search_compact(query: str, max_snippets: int = 3) -> list[dict]:
    """
    Calls Tavily and returns a compact list of results.
    Dedupes and shortens snippets.
    """
    client = TavilyClient()
    res = client.search(query, max_results=6)
    out, seen = [], set()
    for r in res.get("results", []):
        url = (r.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        title = (r.get("title") or url).strip()
        content = (r.get("content") or "").strip()
        if len(content) > 600:
            content = content[:600] + " …"
        out.append({"title": title, "url": url, "content": content})
        if len(out) >= max_snippets:
            break
    return out or [{"title": "No results", "url": "", "content": "No results."}]

def format_snippets(snips: list[dict]) -> str:
    lines = []
    for s in snips:
        lines.append(f"- {s['title']}\n  {s['url']}\n  {s['content']}")
    return "\n\n".join(lines)

def search_tool(q: str) -> str:
    snips = tavily_search_compact(q, max_snippets=3)
    return format_snippets(snips)

# ============================================================
# 3️⃣ LLM, Memory, and Agent Setup
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

tools = [
    Tool(
        name="WebSearch",
        func=search_tool,
        description="Search the live web. Input must be a clear query string."
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
)

SYSTEM_NUDGE = (
    f"{CLARITY_RULES}\n"
    "Final Answer format:\n"
    "- 2–4 sentence summary answering the user.\n"
    "- Then 'Sources:' with 2–4 bullets (Title — URL).\n"
    "- If nothing reliable found, say so and suggest a clearer query."
)

# ============================================================
# 4️⃣ Interactive Loop
# ============================================================

if __name__ == "__main__":
    print("=== Agent with Memory + Clarification + Real Search ===")
    print("Ask anything (science, tech, sports, current news, etc.)")
    print("Press Enter on an empty line to exit.\n")

    try:
        while True:
            user_q = input("You: ").strip()
            if not user_q:
                break

            # Step 1: Clarify or rewrite
            decision = rewrite_for_search(llm, user_q)
            if "clarify" in decision:
                print("\nAssistant (clarifying):", decision["clarify"], "\n")
                continue

            rewritten = decision["query"]

            # Step 2: Run the agent
            prefixed = f"{SYSTEM_NUDGE}\n\nSearch query: {rewritten}\n\nUser: {user_q}"
            answer = agent.run(prefixed)
            print("\nAssistant:", answer, "\n")

    except KeyboardInterrupt:
        print("\n[Exiting]")
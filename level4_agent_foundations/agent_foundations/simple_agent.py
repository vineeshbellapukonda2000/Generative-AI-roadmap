# simple_agent.py

import textwrap
from dotenv import load_dotenv
load_dotenv(override=True)  # ensure the .env here is used

from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI  # modern import



# ---- TOOL DEFINITIONS ----

def web_research_tool(query: str) -> str:
    """Simulated web search. In a real case, this could call SerpAPI or TavilySearch."""
    # Pretend these are snippets from real sources
    results = """
    - 2024 National Coffee Association report: Coffee consumption among Gen Z rose 25% since 2020.
    - Research from Mintel: Rise driven by café culture, social media influence, and convenience drinks like cold brew.
    - Forbes article: Increased remote work and premium coffee marketing improved daily consumption rates.
    """
    return textwrap.dedent(results).strip()

def summarizer_tool(text: str) -> str:
    """Summarize insights into a concise explanation."""
    return (
        "Coffee consumption among young adults has surged mainly due to café culture, "
        "social media trends, remote work lifestyles, and the popularity of premium and cold beverages."
    )

# ---- SET UP LLM AND TOOLS ----
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools = [
    Tool(
        name="WebResearch",
        func=web_research_tool,
        description="Find current reports or studies related to coffee consumption trends."
    ),
    Tool(
        name="Summarizer",
        func=summarizer_tool,
        description="Summarize a collection of research notes into key insights."
    ),
]

# ---- CREATE THE AGENT ----
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type="zero-shot-react-description",  # ReAct-style reasoning loop
    verbose=True  # shows Thoughts / Actions / Observations
)

# ---- RUN ----
question = "What are the main reasons coffee consumption has increased among young adults in recent years?"
result = agent.invoke({"input": question})   # modern API
print("\nFINAL ANSWER:\n", result["output"])


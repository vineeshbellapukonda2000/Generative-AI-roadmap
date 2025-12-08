# tool_agent.py
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from tavily import TavilyClient


# ---- 1.  Tool: real web search ----
def tavily_search_tool(query: str) -> str:
    client = TavilyClient()
    res = client.search(query)
    return "\n".join([r["content"] for r in res["results"][:3]])


# ---- 2.  LLM + Memory ----
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# ---- 3.  Register tools ----
tools = [
    Tool(
        name="TavilySearch",
        func=tavily_search_tool,
        description="Search the web for current information on any topic."
    )
]


# ---- 4.  Create the agent ----
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    max_iterations=3,               
    early_stopping_method="generate"
)


# ---- 5.  Demo ----
if __name__ == "__main__":
    print("=== Agent with Memory + Real Search ===")
    question = "what is the latest news worldwide?"
    answer = agent.run(question)
    print("\nFINAL ANSWER:\n", answer)

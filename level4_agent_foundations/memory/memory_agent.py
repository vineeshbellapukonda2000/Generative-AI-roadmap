# memory_agent.py
# Week 2 – Day 2: Short-term memory for chat and an Agent (ReAct) with memory
# Run:  python memory_agent.py

print(">>> running memory_agent.py")

from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY from .env

from langchain_openai import ChatOpenAI

# Option A: minimal conversational memory
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Option B: Agent (ReAct) with a tiny tool + memory
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType


# ---------------------------
# Option A — ConversationChain (simplest way to *see* memory)
# ---------------------------
def run_conversation_chain_demo():
    """
    ConversationChain expects memory_key='history' and input_key='input'.
    This demo proves the model remembers earlier turns.
    """
    print("\n=== Option A: ConversationChain with Memory ===")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    memory = ConversationBufferMemory(
        memory_key="history",   # <-- required by ConversationChain
        input_key="input",      # <-- the field name you'll pass to .predict
        return_messages=True,   # keep raw messages for fidelity
    )

    convo = ConversationChain(llm=llm, memory=memory, verbose=True)

    # Scripted mini-demo
    print("\n--- Scripted demo ---")
    print("> You: My name is Vineesh. Remember that.")
    print(convo.predict(input="My name is Vineesh. Remember that."))
    print("> You: What is my name?")
    print(convo.predict(input="What is my name?"))
    print("> You: Yesterday you suggested a low-caffeine coffee; which one was it?")
    print(convo.predict(input="Yesterday you suggested a low-caffeine coffee; which one was it? (make something up if needed)"))

    # Interactive loop (Ctrl+C to exit)
    print("\n--- Interactive (type to chat; Ctrl+C to exit) ---")
    try:
        while True:
            user = input("You: ")
            reply = convo.predict(input=user)
            print("Assistant:", reply)
    except KeyboardInterrupt:
        print("\n[Exiting ConversationChain demo]")


# ---------------------------
# Option B — Agent (ReAct) with memory + a tiny tool
# ---------------------------
def keyword_extractor(text: str) -> str:
    """
    Tiny demo tool so the Agent has something to call.
    In Day 3 you'll plug real search tools (Tavily/SerpAPI).
    """
    import re
    from collections import Counter

    words = re.findall(r"[A-Za-z][A-Za-z\-']{2,}", text.lower())
    common = [w for w, _ in Counter(words).most_common(8)]
    return "Keywords: " + ", ".join(common)


def run_agent_with_memory_demo():
    """
    ReAct-style Agent that ALSO remembers chat history.
    Agents expect memory_key='chat_history'.
    Shows: Thought → Action → Observation + memory across turns.
    """
    print("\n=== Option B: Agent (ReAct) with Memory ===")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    tools = [
        Tool(
            name="KeywordExtractor",
            func=keyword_extractor,
            description="Extracts top keywords from the given text.",
        )
    ]

    memory = ConversationBufferMemory(
        memory_key="chat_history",  # <-- Agents use 'chat_history'
        return_messages=True,
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # ReAct pattern
        verbose=True,                # prints Thought/Action/Observation
        memory=memory,               # <-- enable memory
        handle_parsing_errors=True,
    )

    # Scripted mini-demo
    print("\n--- Scripted demo ---")
    print("> You: My favorite topic is specialty coffee beans from Ethiopia.")
    print(agent.run("My favorite topic is specialty coffee beans from Ethiopia."))
    print("> You: Extract some keywords from my preference.")
    print(agent.run("Extract some keywords from my preference."))
    print("> You: Based on what I said earlier, suggest a café style I might like.")
    print(agent.run("Based on what I said earlier, suggest a café style I might like."))

    # Interactive loop (Ctrl+C to exit)
    print("\n--- Interactive (type to chat; Ctrl+C to exit) ---")
    try:
        while True:
            user = input("You: ")
            reply = agent.run(user)
            print("Assistant:", reply)
    except KeyboardInterrupt:
        print("\n[Exiting Agent demo]")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    try:
        # Pick ONE to start. Both show memory; Option B also shows ReAct tool use.
        # Comment out the one you don't want for now.

        # Option A: simplest memory
        run_conversation_chain_demo()

        # Option B: agent + memory (uncomment to try)
        # run_agent_with_memory_demo()

    except Exception as e:
        import traceback
        print("\n[ERROR] An exception occurred:\n")
        traceback.print_exc()
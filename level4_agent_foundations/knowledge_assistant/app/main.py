import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
import uuid


def main():
    # Load environment variables (.env)
    load_dotenv()

    # Import your graph builder
    from app.graph import build_app

    # Build the LangGraph app
    app = build_app()

    # One thread id for the whole session
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("\nKnowledge Assistant is running. Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()

        if q.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        if not q:
            continue

        init_state = {
            "messages": [
                {
                    "role": "user",
                    "content": q
                }
            ]
        }

        # Invoke graph with memory config
        out = app.invoke(init_state, config=config)

        messages = out.get("messages", [])

        if messages:
            last = messages[-1]

            if hasattr(last, "content"):
                print("AI:", last.content)
            else:
                print("AI:", last.get("content", last))
        else:
            print("AI: (No output returned)")


if __name__ == "__main__":
    main()
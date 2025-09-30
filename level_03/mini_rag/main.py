# main.py
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from openai import OpenAI
from retriever import load_artifacts, search_mmr
from dotenv import load_dotenv

load_dotenv()  # <-- load .env first

def build_prompt(query, retrieved_chunks):
    context = "\n\n".join([f"- {c['text']}" for c in retrieved_chunks])
    return f"""
You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {query}

Answer:
""".strip()

def rag_pipeline(query, k=5):
    # 1) load retriever artifacts
    chunks, vecs, index, model = load_artifacts()

    # 2) retrieve (MMR)
    retrieved = search_mmr(query, model, index, chunks, vecs, k=k)

    # 3) build prompt
    prompt = build_prompt(query, retrieved)

    # 4) call LLM
    api_key = os.getenv("OPENAI_API_KEY")           # <-- read from env
    client = OpenAI(api_key=api_key)                # <-- create client now
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    print("💬 Mini RAG App (type 'exit' to quit)\n")
    while True:
        q = input("Ask a question: ")
        if q.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break

        answer = rag_pipeline(q, k=5)
        print("\n✨ Answer:", answer, "\n")
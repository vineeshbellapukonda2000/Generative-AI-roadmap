# main.py
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# local modules
from retriever import rag_pipeline
from structured import value_of, guess_title_from_query  # ensure structured.py defines both

# ---- setup ----
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # quiet HF warning
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- helpers ----
def _to_context(retrieved):
    """
    Turn whatever rag_pipeline returned into a single context string.
    Accepts: list[dict(text=...)], list[str], or str.
    """
    if isinstance(retrieved, str):
        return retrieved
    if isinstance(retrieved, (list, tuple)):
        parts = []
        for c in retrieved:
            if isinstance(c, dict) and "text" in c:
                parts.append(c["text"])
            else:
                parts.append(str(c))
        return "\n\n".join(parts)
    return str(retrieved)

def answer_with_llm(query: str, context: str) -> str:
    """
    Send query + retrieved context to the LLM for the final answer.
    The model MUST answer only from the context.
    """
    prompt = (
        "You are a helpful assistant. Answer using only the context.\n"
        "If the answer isn't in the context, say \"I don't know.\"\n\n"
        "Important:\n"
        "- Rating is a 0–10 scale.\n"
        "- Metascore is a 0–100 scale.\n"
        "Do NOT confuse them.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()

# -------- intent parsing (numeric + descriptive) --------
NUMERIC_PATTERNS = {
    "rating":    [r"\brating\b"],
    "metascore": [r"\bmeta\s*score\b", r"\bmetascore\b"],
    "runtime":   [r"\bruntime\b", r"\brun\s*time\b", r"\bminutes?\b", r"\bhow\s+long\b"],
    "votes":     [r"\bvotes?\b"],
    "revenue":   [r"\brevenue\b"],
    "year":      [r"\byear\b", r"\brelease(d)?\b", r"\brelease\s+year\b"],
}

DESC_PATTERNS = {
    "genre":       [r"\bgenre\b", r"\bwhat\s+type\s+of\s+movie\b"],
    "description": [r"\bplot\b", r"\boverview\b", r"\bsummary\b", r"\bdescription\b", r"\bwhat\s+is\s+it\s+about\b"],
    "actors":      [r"\bactors?\b", r"\bcast\b", r"\bstars?\b"],
    "director":    [r"\bdirect(or|ed)\b"],
}

def parse_intents(q: str):
    low = q.lower()
    num = {k: any(re.search(p, low) for p in pats) for k, pats in NUMERIC_PATTERNS.items()}
    desc = {k: any(re.search(p, low) for p in pats) for k, pats in DESC_PATTERNS.items()}
    return num, desc

def format_numeric(key, val, title):
    if val is None: return None
    if key == "rating":    return f"{title} has a Rating of {val}."
    if key == "metascore": return f"{title} has a Metascore of {int(val)}."
    if key == "runtime":   return f"{title} runs for {int(val)} minutes."
    if key == "votes":     return f"{title} has {int(val)} votes."
    if key == "revenue":   return f"{title} made ${val} million."
    if key == "year":      return f"{title} was released in {int(val)}."
    return None

def ask_llm_field(title: str, field: str):
    """Query RAG specifically for one descriptive field."""
    question_map = {
        "genre":       f"What is the genre of {title}?",
        "description": f"Give a 1-2 sentence plot summary of {title}.",
        "actors":      f"Who are the main actors in {title}?",
        "director":    f"Who directed {title}?",
    }
    q = question_map[field]
    retrieved = rag_pipeline(q, k=5)
    context = _to_context(retrieved)
    return answer_with_llm(q, context)

# ---------------- REPL ----------------
if __name__ == "__main__":
    print("💬 Mini RAG App (type 'exit' to quit)\n")
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY not found. Put it in .env next to main.py")

    while True:
        q = input("Ask a question: ").strip()
        if q.lower() in {"exit", "quit", "q"}:
            print("👋 Bye!")
            break
        try:
            # 1) understand intents
            num_intents, desc_intents = parse_intents(q)

            # 2) guess title (robust) and fallback to text after last " of "
            title = guess_title_from_query(q)
            if not title and " of " in q.lower():
                title = q[q.lower().rfind(" of ") + 4:].strip(" ?.")

            parts = []

            # 3) structured answers for ALL numeric intents present
            if title:
                for key, needed in num_intents.items():
                    if needed:
                        val = value_of(title, key)
                        msg = format_numeric(key, val, title)
                        if msg:
                            parts.append(msg)

            # 4) descriptive answers for ALL descriptive intents present (via RAG+LLM)
            for field, needed in desc_intents.items():
                if needed and title:
                    ans = ask_llm_field(title, field)
                    if ans and ans.lower() != "i don't know":
                        parts.append(ans.strip())

            # 5) if we produced anything, print combined; else fall back to generic RAG
            if parts:
                print("\n✨ Answer:", " ".join(parts), "\n")
                continue

            # Fallback: generic RAG for questions without clear numeric/desc intents
            retrieved = rag_pipeline(q, k=5)
            context = _to_context(retrieved)
            answer = answer_with_llm(q, context)
            print(f"\n✨ Answer: {answer}\n")

        except Exception as e:
            print(f"⚠️ Error: {e}\n")

# recommend.py
import os
import argparse
from dotenv import load_dotenv
from openai import OpenAI

from retriever import search_movies, build_context_snippets, leave_feedback

# Load environment variables from .env (OPENAI_API_KEY)
load_dotenv()


def recommend(
    query: str,
    k: int = 5,
    genre: str | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    min_rating: float | None = None,
):
    # ---- Retrieval ----
    hits = search_movies(
        query=query,
        k=k,
        genre=genre,
        min_year=min_year,
        max_year=max_year,
        min_rating=min_rating,
    )
    if not hits:
        print("No results matched your filters. Try relaxing them.")
        return

    # ---- Print Top Matches with Trending ----
    print("\n--- Top Matches (Semantic Retrieval) ---")
    for h in hits:
        print(
            f"{h['rank']}. {h['title']} ({h['genre']}, {h['year']}) "
            f"| rating={h['rating']} | score={h['score']:.3f} "
            f"| trending={h.get('trending_raw', 0):.2f} (norm {h.get('trending', 0):.2f})"
        )
        # ---- Optional: collect user feedback to personalize future results ----
    print("\n(Optionally give feedback so future results personalize to you)")
    for h in hits:
        while True:
            ans = input(f"Feedback for '{h['title']}' ( + / - / Enter to skip ): ").strip()
            if ans == "+":
                leave_feedback(h["title"], +1)
                print("  → saved 👍")
                break
            elif ans == "-":
                leave_feedback(h["title"], -1)
                print("  → saved 👎")
                break
            elif ans == "":
                break
            else:
                print("  Please enter +, -, or just press Enter.")

    # ---- GPT explainer (optional) ----
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nOPENAI_API_KEY not found in .env – showing retrieval results only.")
        return

    client = OpenAI(api_key=api_key)

    context = build_context_snippets(hits)
    sys_prompt = (
        "You are a helpful movie recommender for a streaming service. "
        "Given a user query and candidate titles with plot/metadata, "
        "recommend the best 3–5 items and explain 'Why this show?' for each. "
        "Ground every explanation in the provided context; keep it concise."
    )
    user_prompt = f'User query: "{query}"\n\nCandidates:\n{context}'

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.5,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    print("\n=== Netflix-style Recommendation ===\n")
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=str, required=True, help="Natural language query")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--genre", type=str, default=None)
    ap.add_argument("--min_year", type=int, default=None)
    ap.add_argument("--max_year", type=int, default=None)
    ap.add_argument("--min_rating", type=float, default=None)
    args = ap.parse_args()

    recommend(
        query=args.q,
        k=args.k,
        genre=args.genre,
        min_year=args.min_year,
        max_year=args.max_year,
        min_rating=args.min_rating,
    )

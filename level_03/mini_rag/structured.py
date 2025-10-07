# structured.py (drop-in)
import pandas as pd
from pathlib import Path
import re

DF = pd.read_csv(Path("data/movies.csv"))
DF["_title_norm"] = DF["Title"].str.lower().str.replace(r"[^a-z0-9 ]+", " ", regex=True).str.split()

VALID_COLS = {
    "rating": "Rating",
    "metascore": "Metascore",
    "runtime": "Runtime (Minutes)",
    "runtime (minutes)": "Runtime (Minutes)",
    "votes": "Votes",
    "revenue": "Revenue (Millions)",
    "revenue (millions)": "Revenue (Millions)",
    "year": "Year",
}

STOPWORDS = set("""
a an the film movie movies runtime run time minutes minute length of is was were what which tell give list show with about for in on to
""".split())

def _norm_tokens(text: str):
    return [t for t in re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split() if t and t not in STOPWORDS]

def guess_title_from_query(query: str):
    """Return the best-matching title from the CSV given a noisy query."""
    qtok = set(_norm_tokens(query))
    if not qtok:
        return None

    best_title, best_score = None, 0.0
    for _, row in DF.iterrows():
        ttok = set(row["_title_norm"])
        if not ttok:
            continue
        # Jaccard overlap over tokens (simple & fast)
        inter = len(qtok & ttok)
        union = len(qtok | ttok)
        score = inter / union if union else 0.0
        if score > best_score:
            best_score = score
            best_title = row["Title"]

    # require a little overlap so “what is the rating” (no title) doesn’t match
    return best_title if best_score >= 0.2 else None

def value_of(title: str, column_key: str):
    col = VALID_COLS.get(column_key.lower())
    if not col:
        return None
    if not title:
        return None
    # exact first
    hit = DF.loc[DF["Title"].str.strip().str.lower() == title.strip().lower()]
    if hit.empty:
        # fuzzy by guesser
        title_guess = guess_title_from_query(title)
        if not title_guess:
            return None
        hit = DF.loc[DF["Title"] == title_guess]
    row = hit.iloc[0]
    val = row[col]
    return None if pd.isna(val) else val

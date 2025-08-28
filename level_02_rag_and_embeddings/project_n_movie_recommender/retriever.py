import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import pickle, faiss, numpy as np, pandas as pd
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

BASE_DIR   = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "vector_index" / "faiss_movies.index"
META_PATH  = BASE_DIR / "vector_index" / "movie_metadata.pkl"
FEEDBACK_PATH = BASE_DIR / "memory" / "feedback.json"

_model = _index = _df = None

def _load_all():
    global _model, _index, _df
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    if _index is None:
        _index = faiss.read_index(str(INDEX_PATH))
    if _df is None:
        with open(META_PATH, "rb") as f:
            _df = pd.DataFrame(pickle.load(f))
    return _model, _index, _df

def embed_text(texts: List[str]) -> np.ndarray:
    m, _, _ = _load_all()
    embs = m.encode(texts)
    return np.array(embs, dtype="float32")

def _passes_filters(row: pd.Series, genre: Optional[str], min_year: Optional[int],
                    max_year: Optional[int], min_rating: Optional[float]) -> bool:
    if genre and (genre.lower() not in str(row.get("Genre","")).lower()):
        return False
    if min_year and int(row.get("Year", 0)) < min_year:
        return False
    if max_year and int(row.get("Year", 9999)) > max_year:
        return False
    if min_rating and float(row.get("Rating", 0.0)) < min_rating:
        return False
    return True

def _load_feedback() -> Dict[str, int]:
    import json
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if FEEDBACK_PATH.exists():
        try:
            return json.loads(FEEDBACK_PATH.read_text())
        except Exception:
            return {}
    return {}

def _save_feedback(store: Dict[str, int]) -> None:
    import json
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEEDBACK_PATH.write_text(json.dumps(store, indent=2))

def leave_feedback(title: str, delta: int):
    """delta = +1 for like, -1 for dislike."""
    store = _load_feedback()
    store[title] = store.get(title, 0) + delta
    _save_feedback(store)

def search_movies(query: str, k: int = 5, genre: Optional[str] = None,
                  min_year: Optional[int] = None, max_year: Optional[int] = None,
                  min_rating: Optional[float] = None,
                  alpha_semantic: float = 0.75,
                  beta_trending: float = 0.15,
                  gamma_feedback: float = 0.10) -> List[Dict[str, Any]]:
    """
    Returns ranked hits with a blended score:
      blended = alpha*semantic + beta*trending_norm + gamma*feedback_norm
    """
    _, index, df = _load_all()
    q_emb = embed_text([query])

    # 1) initial FAISS pool
    pool = max(k * 10, 60)
    D, I = index.search(q_emb, pool)
    I = I[0]

    # 2) filter candidates
    candidates = []
    for idx in I:
        row = df.iloc[int(idx)]
        if _passes_filters(row, genre, min_year, max_year, min_rating):
            candidates.append((idx, row))
    if not candidates:
        return []

    # 3) cosine re-rank (semantic)
    contents = []
    for _, r in candidates:
        c = r.get("content") or f"{r.get('Title','')} - {r.get('Genre','')}: {r.get('Description','')}"
        contents.append(c)
    doc_embs = embed_text(contents)
    q = q_emb[0]
    qn = q / (np.linalg.norm(q) + 1e-12)
    dn = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-12)
    cos_scores = dn @ qn  # (n,)

    # 4) normalize auxiliary signals
    # trending
    trending = np.array([float(r.get("trending_score", 0.0)) for _, r in candidates], dtype="float32")
    if trending.size and trending.max() > trending.min():
        trending_norm = (trending - trending.min()) / (trending.max() - trending.min())
    else:
        trending_norm = np.zeros_like(trending)

    # feedback
    fb_store = _load_feedback()
    feedback = np.array([float(fb_store.get(str(r.get("Title","")), 0)) for _, r in candidates], dtype="float32")
    if feedback.size:
        # tanh squashes extremes; shift to [0,1]
        feedback_norm = (np.tanh(feedback) + 1.0) / 2.0
    else:
        feedback_norm = np.zeros_like(trending)

    # 5) blend
    blended = alpha_semantic * cos_scores + beta_trending * trending_norm + gamma_feedback * feedback_norm
    order = np.argsort(-blended)[:k]

    results = []
    for rank, pos in enumerate(order, start=1):
        idx, row = candidates[pos]
        results.append({
            "rank": rank,
            "score": float(blended[pos]),
            "semantic": float(cos_scores[pos]),
            "trending": float(trending_norm[pos]),
            "feedback": float(feedback_norm[pos]),
            "row_index": int(idx),
            "title": row.get("Title",""),
            "genre": row.get("Genre",""),
            "year": int(row.get("Year", 0)) if pd.notna(row.get("Year")) else None,
            "rating": float(row.get("Rating", 0.0)) if pd.notna(row.get("Rating")) else None,
            "description": row.get("Description","")
        })
    return results

def build_context_snippets(hits: List[Dict[str, Any]]) -> str:
    lines = []
    for h in hits:
        g = f" ({h['genre']})" if h["genre"] else ""
        y = f", {h['year']}" if h["year"] else ""
        r = f" | Rating: {h['rating']}" if (h["rating"] is not None) else ""
        lines.append(f"- {h['title']}{g}{y}{r}\n  Plot: {h['description']}")
    return "\n".join(lines)

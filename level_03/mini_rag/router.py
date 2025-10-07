# router.py
from structured import try_answer_structured
from retriever import load_artifacts, search_mmr

def route_query(query: str, k=5):
    # A) try structured first (numbers, filters, counts, per-title numeric facts)
    ans = try_answer_structured(query)
    if ans:
        return {"mode": "structured", "answer": ans}

    # B) else semantic RAG (MMR; you can swap to cosine)
    chunks, vecs, index, model = load_artifacts()
    retrieved = search_mmr(query, model, index, chunks, vecs, k=k)
    context = "\n\n".join([c["text"] for c in retrieved])

    return {"mode": "rag", "context": context, "chunks": retrieved}

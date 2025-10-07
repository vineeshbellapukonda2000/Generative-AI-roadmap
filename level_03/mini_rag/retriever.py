# retriever.py
import argparse, json, os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

CHUNKS_PATH = "chunks.jsonl"
EMB_PATH = "embeddings.npy"
INDEX_PATH = "faiss.index"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def load_artifacts():
    if not (os.path.exists(CHUNKS_PATH) and os.path.exists(EMB_PATH) and os.path.exists(INDEX_PATH)):
        raise FileNotFoundError("Missing artifacts. Run `python ingest.py` first.")
    # chunks
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    # embeddings + index + model
    vecs = np.load(EMB_PATH)
    index = faiss.read_index(INDEX_PATH)
    model = SentenceTransformer(MODEL_NAME)
    return chunks, vecs, index, model

def search_cosine(query, model, index, chunks, k=5):
    qv = model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, k)
    out = []
    for score, idx in zip(D[0], I[0]):
        c = chunks[int(idx)]
        out.append({"score": float(score), "id": c["id"], "text": c["text"], "idx": int(idx)})
    return out

def _mmr_select(qv, cand_vecs, k=6, lambda_mult=0.6):
    # qv shape (d,), cand_vecs shape (n,d); all normalized
    sims_q = cand_vecs @ qv
    pair = cand_vecs @ cand_vecs.T
    selected = []
    remaining = list(range(len(cand_vecs)))
    for _ in range(min(k, len(remaining))):
        if not selected:
            i = int(np.argmax(sims_q[remaining]))
            selected.append(remaining.pop(i))
            continue
        best_i, best_score = None, -1e18
        for r in remaining:
            penalty = float(np.max(pair[r, selected]))
            score = lambda_mult * float(sims_q[r]) - (1 - lambda_mult) * penalty
            if score > best_score:
                best_i, best_score = r, score
        selected.append(best_i)
        remaining.remove(best_i)
    return selected

def search_mmr(query, model, index, chunks, vecs, fetch_k=25, k=6, lambda_mult=0.6):
    qv = model.encode([query], normalize_embeddings=True)[0].astype("float32")
    D, I = index.search(qv[None, :], fetch_k)      # candidate pool by cosine
    cand_vecs = vecs[I[0]]                         # normalized from ingest
    keep_positions = _mmr_select(qv, cand_vecs, k=k, lambda_mult=lambda_mult)
    out = []
    for pos in keep_positions:
        idx = int(I[0][pos])
        c = chunks[idx]
        out.append({"score": float(D[0][pos]), "id": c["id"], "text": c["text"], "idx": idx})
    return out

def cli():
    ap = argparse.ArgumentParser(description="Retriever CLI")
    ap.add_argument("query", help="Your question")
    ap.add_argument("--mode", choices=["cosine", "mmr"], default="mmr")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--fetch_k", type=int, default=25, help="MMR candidate pool (only for --mode mmr)")
    ap.add_argument("--lambda_mult", type=float, default=0.6, help="MMR relevance/diversity balance")
    args = ap.parse_args()

    chunks, vecs, index, model = load_artifacts()
    if args.mode == "cosine":
        results = search_cosine(args.query, model, index, chunks, k=args.k)
    else:
        results = search_mmr(args.query, model, index, chunks, vecs,
                             fetch_k=args.fetch_k, k=args.k, lambda_mult=args.lambda_mult)

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] score={r['score']:.4f}  id={r['id']}\n{r['text']}\n")

if __name__ == "__main__":
    cli()

def rag_pipeline(query, k=5):
    chunks, vecs, index, model = load_artifacts()
    retrieved = search_mmr(query, model, index, chunks, vecs, k=k)
    return retrieved  # or return the texts you want


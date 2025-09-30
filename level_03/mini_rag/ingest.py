# ingest.py
import os, re, glob, json
import numpy as np
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

DATA_DIR = "data"
CHUNKS_PATH = "chunks.jsonl"
EMB_PATH = "embeddings.npy"
INDEX_PATH = "faiss.index"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def load_documents(data_dir=DATA_DIR):
    docs = []
    for path in glob.glob(os.path.join(data_dir, "*")):
        ext = os.path.splitext(path)[1].lower()
        if ext in [".txt", ".md"]:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                docs.append(f.read())
        elif ext == ".pdf":
            reader = PdfReader(path)
            text = "\n".join((p.extract_text() or "") for p in reader.pages)
            if text.strip(): docs.append(text)
        elif ext == ".csv":
            df = pd.read_csv(path).fillna("")
            text_cols = [c for c in df.columns if df[c].dtype == object]
            for _, row in df[text_cols].iterrows():
                joined = " ".join(str(v) for v in row if str(v).strip())
                if joined.strip(): docs.append(joined)
    return docs

def chunk_text(text, chunk_words=220, overlap_words=40):
    words = re.split(r"\s+", text.strip())
    step = max(1, chunk_words - overlap_words)
    chunks = []
    for i in range(0, len(words), step):
        piece = " ".join(words[i:i+chunk_words]).strip()
        if piece: chunks.append(piece)
    return chunks

def main():
    # 1) Load docs
    docs = load_documents()
    if not docs:
        raise SystemExit("No documents found in ./data . Add files (pdf/txt/md/csv) and retry.")

    # 2) Chunk
    chunks = []
    for d_i, doc in enumerate(docs):
        for c_i, c in enumerate(chunk_text(doc)):
            chunks.append({"id": f"d{d_i}_c{c_i}", "text": c})

    # 3) Embed (normalized for cosine)
    model = SentenceTransformer(MODEL_NAME)
    vecs = model.encode([c["text"] for c in chunks],
                        batch_size=64,
                        show_progress_bar=True,
                        normalize_embeddings=True)
    vecs = np.asarray(vecs, dtype="float32")

    # 4) Build FAISS (cosine via inner product on normalized vectors)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    # 5) Save artifacts
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    np.save(EMB_PATH, vecs)
    faiss.write_index(index, INDEX_PATH)

    print(f"✅ Ingest complete: {len(chunks)} chunks")
    print(f"✅ Embeddings saved: {EMB_PATH}")
    print(f"✅ Chunks saved:     {CHUNKS_PATH}")
    print(f"✅ FAISS index saved successfully: {INDEX_PATH}")

if __name__ == "__main__":
    main()

# ingest.py
import os
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from pathlib import Path

# ----------------------------
# Paths (locked to this file)
# ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH  = SCRIPT_DIR / "data" / "movies.csv"
OUT_DIR    = SCRIPT_DIR / "vector_index"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Reading:", DATA_PATH)

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv(DATA_PATH)

required = {
    'Rank','Title','Genre','Description','Director','Actors',
    'Year','Runtime (Minutes)','Rating','Votes','Revenue (Millions)','Metascore'
}
if not required.issubset(df.columns):
    print("⚠️ Please check your CSV column names. Required at minimum: Title, Genre, Description")
    print("Current columns:", df.columns.tolist())
    raise SystemExit(1)

# ----------------------------
# Build searchable text
# ----------------------------
df["content"] = (
    " | movie is " + df["Title"] + " - " +
    " | type: " + df["Genre"] + " | " +
    " | context is: " + df["Description"] +
    " | Directed by " + df["Director"] +
    " | Starring: " + df["Actors"] +
    " | Year: " + df["Year"].astype(str) +
    " | Runtime: " + df["Runtime (Minutes)"].astype(str) + " minutes" +
    " | Rating: " + df["Rating"].astype(str) +
    " | Votes: " + df["Votes"].astype(str) +
    " | Revenue: " + df["Revenue (Millions)"].astype(str) + "M" +
    " | Metascore: " + df["Metascore"].astype(str)
)

# ----------------------------
# Embeddings (normalize -> cosine via inner product)
# ----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df["content"].tolist(), show_progress_bar=True)
embeddings = normalize(embeddings)                       # L2-normalize for cosine
embeddings = embeddings.astype(np.float32)

# ----------------------------
# FAISS index (IP = cosine on normalized)
# ----------------------------
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# ----------------------------
# Save index + metadata next to this script
# ----------------------------
faiss.write_index(index, str(OUT_DIR / "faiss_movies.index"))
with open(OUT_DIR / "movie_metadata.pkl", "wb") as f:
    pickle.dump(df.to_dict(), f)

print("✅ FAISS index and metadata saved:", OUT_DIR)
print("   -", OUT_DIR / "faiss_movies.index")
print("   -", OUT_DIR / "movie_metadata.pkl")

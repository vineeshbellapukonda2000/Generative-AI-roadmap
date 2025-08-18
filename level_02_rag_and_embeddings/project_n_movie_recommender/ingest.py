import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

# Load dataset
import os
script_dir = os.path.dirname(os.path.abspath(__file__))          # folder where ingest.py lives
data_path  = os.path.join(script_dir, "data", "movies.csv")      # .../project_n_movie_recommender/data/movies.csv
print("Reading:", data_path)                                     # debug print; keep it

df = pd.read_csv(data_path)

# Adjust column names if needed (make sure these exist in your CSV)
if not {'Rank','Title', 'Genre', 'Description','Director','Actors','Year','Runtime (Minutes)','Rating','Votes','Revenue (Millions)','Metascore'}.issubset(df.columns):
    print("⚠️ Please check your CSV column names. Required: title, genre, description")
    print("Current columns:", df.columns.tolist())
    exit()

# Combine fields into one searchable string
df["content"] = (
    " | movie is " + df["Title"] + " - " +
    " | type: " + df["Genre"] + " | " +
    " | context is: " +  df["Description"] +
    " | Directed by " + df["Director"] +
    " | Starring: " + df["Actors"] +
    " | Year: " + df["Year"].astype(str) +
    " | Runtime: " + df["Runtime (Minutes)"].astype(str) + " minutes" +
    " | Rating: " + df["Rating"].astype(str) +
    " | Votes: " + df["Votes"].astype(str) +
    " | Revenue: " + df["Revenue (Millions)"].astype(str) + "M" +
    " | Metascore: " + df["Metascore"].astype(str)
)


from sklearn.preprocessing import normalize

# Create and normalize embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df["content"].tolist(), show_progress_bar=True)
embeddings = normalize(embeddings)  # L2 normalize

# Create FAISS index (inner product works like cosine on normalized vectors)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype(np.float32))

# Ensure vector_index folder exists
os.makedirs("vector_index", exist_ok=True)

# Save FAISS index
faiss.write_index(index, "vector_index/faiss_movies.index")

# Save metadata (movie info)
with open("vector_index/movie_metadata.pkl", "wb") as f:
    pickle.dump(df.to_dict(), f)

print("✅ FAISS index and metadata saved successfully!")

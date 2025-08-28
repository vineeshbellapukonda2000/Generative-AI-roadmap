# scripts/add_trending.py
import pandas as pd
import numpy as np

df = pd.read_csv("data/movies.csv")
if "trending_score" not in df.columns:
    rng = np.random.default_rng(42)
    df["trending_score"] = (0.4 * (df.get("Votes", 1000) / (df.get("Votes", 1000).max() or 1))
                            + 0.6 * rng.random(len(df)))
df.to_csv("data/movies.csv", index=False)
print("Added/updated trending_score in data/movies.csv")

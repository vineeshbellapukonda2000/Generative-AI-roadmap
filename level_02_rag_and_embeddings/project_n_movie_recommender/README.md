🎬 GenAI Movie & Show Recommender

An end-to-end Generative AI project that combines semantic search, retrieval-augmented generation (RAG), and user feedback loops to deliver Netflix-style movie recommendations — with explanations.

🚀 Project Overview

This project demonstrates how modern recommenders can be enhanced with embeddings + RAG + LLMs:

Use embeddings + FAISS for semantic retrieval of movies.

Use GPT for explainable recommendations (answers “Why this show?”).

Integrate feedback memory for personalization.

Add trending signals to simulate OTT-style boosts for recent/popular titles.

This is part of my GenAI Learning Roadmap (Level 2 Project) — applying concepts of vector embeddings, RAG pipelines, and memory systems.

✅ Features Implemented
Phase 1: Data Ingestion + Indexing

Loaded and processed movies dataset (movies.csv).

Generated embeddings using SentenceTransformers (all-MiniLM-L6-v2).

Built FAISS vector index for similarity search.

Stored metadata (.pkl) for efficient retrieval.

Phase 2: Retrieval-Augmented Generation (RAG)

Semantic search to get Top-K similar movies using cosine similarity.

Passed candidates + query into GPT (OpenAI) to generate explanations.

Netflix-style outputs: “Here’s why this movie fits your query.”

Phase 3: Advanced Features

Feedback Memory: thumbs up/down stored to refine personalization.

Trending Signal: boosting scores for recent/popular movies.

Hybrid Ranking: final ranking blends semantic + trending + feedback signals.

📌 Tech Stack

Python (3.11)

SentenceTransformers for embeddings (all-MiniLM-L6-v2)

FAISS for vector similarity search

OpenAI GPT for natural language generation (RAG)

dotenv for key management

pandas / numpy / pickle for dataset + metadata

⚡ Project Pipeline

1. Embedding Pipeline

Movies.csv → Embeddings → FAISS Vector Index


2. RAG Pipeline

User Query → Embedding → FAISS Semantic Search → Top-K Matches → GPT → “Why this show?”


3. Personalization & Ranking

Semantic Score + Trending Boost + Feedback Memory → Final Recommendations

▶️ Usage

Clone repo & install requirements

git clone https://github.com/<your-username>/Generative-AI-roadmap.git
cd level_02_rag_and_embeddings/project_n_movie_recommender
pip install -r requirements.txt


Add your OpenAI key
Create a .env file in the project folder:

OPENAI_API_KEY=your_api_key_here


Ingest dataset & build index

python ingest.py --data data/movies.csv


Run recommendation

python recommend.py --q "underrated mind-bending time travel movies"

📊 Example Output

Top Matches (Semantic Retrieval):

1. Predestination (2014) | Rating: 7.5
2. Limitless (2011) | Rating: 7.4
3. Lavender (2016) | Rating: 5.2


Netflix-style Explanations (RAG):

- Predestination (2014): A standout in time travel storytelling...
- Limitless (2011): Expands mental capabilities with a thrilling arc...
- Lavender (2016): Atmospheric psychological thriller with a unique twist...

🔮 Future Work

🎨 Build a Streamlit UI for interactive recommendations.

🎥 Add support for multi-modal retrieval (text + posters/trailers).

📈 Enhance personalization with collaborative filtering + RAG hybrid.

📌 Why This Project Matters

This project shows how RAG + vector search + feedback memory can make recommendations:

Explainable (every movie has a “Why this show?”)

Personalized (adapts to user feedback)

Scalable (FAISS indexing for large datasets)

Streaming platforms like Netflix, Prime, Hulu can adopt similar pipelines to provide transparent, trustworthy, and adaptive recommendations.

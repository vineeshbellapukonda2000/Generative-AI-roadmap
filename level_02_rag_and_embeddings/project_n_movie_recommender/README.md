# Project N — GenAI Movie & Show Recommender 

RAG-powered semantic recommender that suggests movies/shows by mood, theme, or example titles, with explainability and Netflix-style enhancements.

## Status
- ✅ Phase 1: Ingestion/Embeddings
  - `data/movies.csv` ingested
  - Embeddings generated with `sentence-transformers (all-MiniLM-L6-v2)`
  - Vector index stored with FAISS

## Next
- Phase 2: Retrieval + GPT prompt for "Why this show?"
- Phase 3: Netflix-style enhancements (trending filter, mood filters, feedback memory)

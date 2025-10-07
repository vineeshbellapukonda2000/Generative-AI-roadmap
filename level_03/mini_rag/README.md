📜 Project Journey & Learnings
🏁 Phase 1 — Foundation & Dataset Understanding

Started with the movie dataset (movies.csv), exploring columns such as Title, Genre, Description, Year, Rating, Runtime, Revenue, and Metascore.

Preprocessed and cleaned the dataset using Pandas, preparing it for both structured queries and semantic embedding generation.

Built the first version of the ingest.py script to chunk movie descriptions and create embeddings using OpenAI’s embedding model.

Stored embeddings in a FAISS vector index, setting the foundation for retrieval.

🧱 Phase 2 — Building the RAG Pipeline

Developed the retrieval system (retriever.py) that loads FAISS, performs vector similarity search, and returns relevant chunks.

Implemented semantic search that allowed the model to answer descriptive questions such as:

“What is the movie Inception about?” or “Who acted in The Dark Knight?”

Initially, the system could only answer questions directly related to text chunks from the dataset, which we verified successfully through multiple test queries.

⚙️ Phase 3 — Introducing Structured Query Handling

Identified a major limitation: the model couldn’t handle numeric or factual questions (like rating, runtime, revenue, or release year).

Designed a new module structured.py to query structured data directly from the CSV using Pandas.

Implemented regex-based intent recognition to identify patterns like:

“What is the rating of ___?”

“How long is ___?”

“When was ___ released?”

This made the system capable of handling fact-based queries with precision.

🧠 Phase 4 — Merging Structured + RAG Pipelines

Combined both systems into a Hybrid Intelligent Layer (main.py) that routes questions automatically:

If the query involves numeric/factual data → handled by Structured Lookup.

If it’s descriptive/contextual → handled by RAG + LLM.

Implemented fusion logic to handle multi-intent questions, e.g.

“What is the rating and genre of The Dark Knight Rises?”
This required merging structured data (rating) with retrieved chunks (genre), which we achieved using router.py.

🧩 Phase 5 — Fine-Tuning the Interaction Layer

Created robust context-building logic in _to_context() for better LLM responses.

Improved query parsing (guess_title_from_query, parse_intents) to handle flexible question structures and capitalization.

Added OpenAI GPT-4o-mini as the reasoning model for final answer generation.

Enforced strict instruction in the LLM prompt:

“Answer using only the context. If the answer isn’t in context, say I don’t know.”

This made the responses accurate, interpretable, and context-bound.

🧗 Struggles & How We Overcame Them

Challenge	Description	How We Solved It
FAISS and file path errors	Early on, file directories like level_03/mini_rag caused “no such file” errors due to spaces in folder names.	Fixed by using quoted paths and relative file references (Path(__file__).parent).
Mixing Structured + RAG answers	Initially, LLM returned only one type of answer (either numeric or descriptive).	Created a fusion router that merges structured lookups and semantic retrieval results before prompting the LLM.
Mislabeling (Rating vs Metascore)	The LLM often confused Rating (0–10) and Metascore (0–100).	Explicitly added prompt constraints in main.py to distinguish between the two scales.
“I don’t know” responses	LLM returned “I don’t know” for queries slightly mismatched to dataset text.	Enhanced semantic search recall (MMR) and added fallback descriptive prompts.
Permission/Conda path issues	Encountered Mac “permission denied” and environment confusion while navigating paths.	Fixed by using chmod -R 755 and quoting all conda/env paths.
Complex multi-intent questions	LLM ignored second intent (e.g., genre + rating).	Implemented intent parsing and joined both structured and RAG responses.
🧾 Key Takeaways

Learned how to combine structured data retrieval and unstructured RAG search into a unified conversational system.

Understood how embeddings, FAISS, and semantic similarity enable question answering beyond keywords.

Built an architecture capable of fusing symbolic (structured) and semantic (vector) intelligence — a real-world foundation for enterprise knowledge assistants.

Practiced debugging, environment management, and prompt engineering for reliable hybrid AI systems.
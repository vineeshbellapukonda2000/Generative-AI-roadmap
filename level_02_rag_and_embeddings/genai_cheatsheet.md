GenAI Cheat Sheet – Level 2: RAG, Embeddings, Vector Stores, Memory


🔹 Embeddings
Converts text → high-dimensional vector

Captures semantic meaning

Models: sentence_transformers, text-embedding-3-small, all-MiniLM-L6-v2

Code:

python
Copy
Edit
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
vector = model.encode("What are the best exercises for biceps?")

🔹 Cosine Similarity
Measures how close two vectors are (angle-based)

Used to retrieve relevant documents


🔹 Vector Stores
Stores embedded documents

Supports similarity search

Popular Stores:

FAISS (local, fast)

ChromaDB (simple, Pythonic)

Pinecone, Weaviate (cloud-based)


🔹 RAG (Retrieval-Augmented Generation)
Pipeline:

markdown
Copy
Edit
1. User Query → 
2. Embed Query → 
3. Vector Store Search (cosine similarity) → 
4. Retrieve top-k documents → 
5. Combine with prompt → 
6. Send to LLM (GPT) → 
7. Generate Answer
Goal: Use retrieved knowledge to reduce hallucination and improve relevance.


🔹 Memory in GenAI
Type	Purpose	Tool
ConversationBufferMemory	Short-term memory (last N turns)	LangChain
VectorStoreRetrieverMemory	Long-term memory (semantic search)	FAISS + LangChain
CombinedMemory	Uses both	LangChain


🔹 LangChain Components
Component	Description
Embeddings	Converts text into vectors
VectorStore	Stores/retrieves documents
Retriever	Interface to search vector DB
Memory	Tracks chat context
LLMChain	Runs query + context through LLM

🔹 Chunking
Break large docs into small chunks before embedding

Use overlap (e.g., 500 tokens with 50 overlap) to preserve meaning


🔹 Best Practices
Use same embedding model for both docs and queries

Fine-tune chunk size and retrieval k

Avoid prompt flooding (keep context under LLM token limits)

🔹 Example Use Cases
Chatbots grounded in personal notes

Legal assistants with case law documents

Fitness/Nutrition coaches with custom plans

Movie recommenders from reviews and tags


LEVEL 2: RAG Systems & Embeddings

Core Focus: Vector Embeddings • Retrieval-Augmented Generation (RAG) • Vector Stores • Memory • Semantic Search



1. What are Embeddings?
Definition:
Embeddings are numerical vector representations of text (or other media like images/audio) that capture semantic meaning — i.e., similar content will have similar vectors.

Why use them?
They let you compare meaning across texts, rather than exact word match. This powers search, question-answering, and RAG systems.

Popular Models:
OpenAI's text-embedding-3-small/large

SentenceTransformers (e.g., all-MiniLM, mpnet-base)

Cohere, Google's BERT, E5 models

Use Case Example:
You embed both:

Documents (knowledge base)

Query (user input)
Then retrieve top-k closest matches (via cosine similarity), and send them to the LLM to generate a better answer.

Now let’s understand this:

Step 1: Build a Knowledge Base
Let’s say you have a bunch of .txt or .csv files like:

top_1000_movies.csv

movies_reviews.txt

genre_classics.txt

hidden_gems.json

Each contains textual descriptions like:

vbnet
Copy
Edit
Title: Predestination
Genre: Sci-fi, Time Travel
Review: One of the most mind-bending time travel movies with a closed-loop paradox...

Step 2: Convert Documents into Embeddings
You pass each chunk of data (each movie entry or paragraph) to an embedding model like:

python
Copy
Edit
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("One of the most mind-bending time travel movies...")
Now the text becomes a vector:

text
Copy
Edit
[0.213, -0.412, 0.001, ...] ← numeric representation
You do this for all your movie entries and store them in a vector store like FAISS or Chroma.

Step 3: User Asks a Query
User asks:

“Give me time-travel thrillers with a plot twist.”

This query is also embedded using the same model, and now it's a vector too:

text
Copy
Edit
[0.204, -0.409, 0.005, ...]

Step 4: Find Closest Matches
You compare the user vector with all movie vectors using cosine similarity. The vector store gives you:

text
Copy
Edit
Top 3 matching movie chunks:
1. Predestination
2. Coherence
3. Timecrimes
Step 5: Send Context + Query to LLM
Now you take those top 3 matching chunks (movie info) and send them along with the user’s query into GPT:

python
Copy
Edit
prompt = f"""
You are a movie assistant. The user asked: {user_query}

Here are some relevant movie descriptions:
1. Predestination: One of the most mind-bending time travel thrillers...
2. Coherence: A dinner party goes haywire when alternate realities collide...
3. Timecrimes: A man stumbles into a time loop after entering a machine...

Based on this, give a smart recommendation and explain why.
"""

response = openai.ChatCompletion.create( ... )
And boom — GPT now answers using real context, like:

“You might love Coherence — it's an underrated thriller with a nonlinear timeline, similar to Predestination but even more grounded in tension.”

Summary
Step	        What Happens
Embed Docs	    Turn each movie description into a vector
Embed Query	    Turn user's question into a vector
Retrieve	    Find top N closest movie chunks
Generate	    GPT uses them to create a tailored answer

sentence_transformers is a Python library that provides pre-trained embedding models to convert text into numeric vector representations.

What Makes Sentence Transformers Special?
Built on BERT-like transformer models
Trained for semantic similarity

Great for tasks like:
RAG (Retrieval-Augmented Generation)
Semantic Search
Clustering similar sentences
FAQ answering
Duplicate detection

Code Example:
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # Light, fast model

sentence = "Suggest me a good vegetarian protein source"
embedding = model.encode(sentence)

print(embedding[:5])  # Just print first 5 numbers for preview

How It's Used in RAG:
You embed:

All your documents (movie descriptions, articles, etc.)

Every query from the user

Then match them via cosine similarity

That’s how you find the semantically closest content — not just exact word match.




2. RAG (Retrieval-Augmented Generation)
RAG = Search + LLM
RAG architecture retrieves relevant context (docs/chunks) and gives it to the LLM to answer a query. This solves the LLM limitation of fixed context window and hallucination.

RAG Pipeline:

[User Query]
      ↓
[Embed the query]
      ↓
[Search Vector Store for relevant chunks]
      ↓
[Send top-k context + query → LLM]
      ↓
[Generated Response]

Key Components:
Embedding Model
Vector Store (FAISS, ChromaDB, Pinecone, Weaviate)
Chunking strategy (size, overlap)
Prompt Template

Key insight:
The “R” in RAG — Retrieval — officially starts when:
You embed the user query
And retrieve relevant document chunks using similarity search
So yes, once you’ve embedded both your docs and your query — you’re in RAG mode 

From This Point On, You're Doing “RAG”
Here's the breakdown:

 Step	      Action	          Tool
1. Embedding your documents	      Turn all your source content into vectors	   SentenceTransformer or OpenAIEmbeddings
2. Storing embeddings	  Save them in a searchable vector DB	          FAISS, ChromaDB, Pinecone
3. Embedding the user query	    Convert input question to a vector	  Same embedding model
4. Retrieval	    Search for the closest document vectors using cosine similarity      Vector store methods
5. Augmentation	     Plug the retrieved results + query into a GPT prompt	  OpenAI, LLMChain, etc.
6. Generation	LLM generates a response based on the retrieved content	LLM     (GPT-4 or similar)

Key Insight:
The “R” in RAG — Retrieval — officially starts when:
You embed the user query
And retrieve relevant document chunks using similarity search
So yes, once you’ve embedded both your docs and your query — you’re in RAG mode

Phase	  Description	       Is cosine similarity involved?
1. Retrieval	Find the most relevant document chunks by comparing the user query with all stored chunks (via cosine similarity)	 Yes
2. Augmented Generation	Send the retrieved chunks + the original query to an LLM to generate the final answer	 No (this uses text-based prompting)


The retrieval part using cosine similarity is a core part of the RAG (Retrieval-Augmented Generation) process.




3. Vector Stores
These store embeddings efficiently and allow similarity search.

Examples:
FAISS (Facebook AI Similarity Search) → Local, fast

Chroma → Python-native, great for prototyping

Pinecone, Weaviate → Cloud-based, scalable

Store Structure:
text
Copy
Edit
{
  id: "doc123",
  text: "How to treat Type 2 diabetes...",
  embedding: [0.12, 0.34, 0.89, ...]
}

Each document is saved with:
ID
Raw text
Vector embedding



4. Memory in Gen AI Apps

Types:
ConversationBufferMemory: Stores last few interactions.

VectorStoreRetrieverMemory: Stores past conversations as embeddings for smarter recall.

CombinedMemory: Merge strategies for long-form chat.


Memory allows a Generative AI application to:

Remember previous user interactions (across sessions or turns)

Recall relevant past context to make smarter responses

Personalize conversations over time

Why Do We Need Memory?
Without memory:
User: "Who is the CEO of Google?"
Bot: "Sundar Pichai."

User: "Where was he born?"
Bot: Doesn’t know who “he” refers to — loses context.

With memory:
Bot remembers “he = Sundar Pichai” and answers


Types of Memory (in GenAI & RAG apps)
1. ConversationBufferMemory (Short-term memory)
Stores just the last N messages in the chat.

Useful for recent context (e.g., 3–5 messages).

Built into tools like LangChain.

python
Copy
Edit
memory = ConversationBufferMemory()
Good for chatbots like:

“Remind me what I said earlier about my workout split.”

2. VectorStoreRetrieverMemory (Long-term semantic memory)
Stores each user message + response as embeddings in a vector DB.

On each new query, it retrieves past interactions semantically related to the current query.

This is where RAG + memory combine beautifully!

text
Copy
Edit
Query: “Give me the protein advice you gave me last week”
→ Retrieved: “You said I work out 5x/week and weigh 68kg…”
Great for apps like:

Personalized health coach, finance advisor, or study assistant.

3. CombinedMemory (Hybrid)
Merges short-term + long-term memory.

Uses a buffer for recent turns and vector store for older, similar topics.
Ideal for large apps with many users or large conversations.

How It Works in a RAG App:
Step	Description
1.	Each user message + response is stored in a vector store as an embedding
2.	New query is embedded and compared using cosine similarity
3.	Top-matching past conversation snippets are retrieved
4.	These are added as context to the prompt sent to the LLM
5.	LLM responds with knowledge of past interactions

Implementation Example in LangChain
python
Copy
Edit
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.load_local("my_memory_store", embedding_model)

retriever = vectorstore.as_retriever()
memory = VectorStoreRetrieverMemory(retriever=retriever)

# When chat runs
memory.save_context({"input": user_input}, {"output": bot_response})




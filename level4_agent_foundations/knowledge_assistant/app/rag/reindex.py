from app.rag.ingest import load_docs_with_sources
from app.rag.chunk import chunk_docs_with_sources
from app.rag.embed import upsert_chunks


def rebuild_index():
    docs = load_docs_with_sources()
    chunk_objs = chunk_docs_with_sources(docs)

    chunks = [c["text"] for c in chunk_objs]
    metadatas = [{"source": c["source"]} for c in chunk_objs]

    upsert_chunks(chunks, metadatas=metadatas)
    print(f"Reindexed {len(docs)} docs into {len(chunks)} chunks.")


if __name__ == "__main__":
    rebuild_index()
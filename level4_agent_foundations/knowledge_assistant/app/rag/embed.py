from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


DB_DIR = Path(__file__).resolve().parents[2] / "data" / "chroma_db"
COLLECTION_NAME = "knowledge_base"

_model = None


def _get_model(name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(name)
    return _model


def build_embeddings(chunks: List[str]) -> List[List[float]]:
    model = _get_model()
    vectors = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    return vectors.tolist()


def upsert_chunks(chunks: List[str], metadatas: List[Dict[str, Any]] | None = None) -> None:
    """
    Create/update a Chroma collection with chunk text + embeddings.
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    embeddings = build_embeddings(chunks)

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    if metadatas is None:
        metadatas = [{"source": "data/docs"} for _ in chunks]

    collection.upsert(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def query_db(query: str, k: int = 5) -> Dict[str, Any]:
    """
    Query Chroma using the same embedding model.
    Returns top-k docs + metadata.
    """
    client = chromadb.PersistentClient(
        path=str(DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    q_emb = build_embeddings([query])[0]

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return results
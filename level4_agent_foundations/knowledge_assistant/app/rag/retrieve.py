from typing import List, Dict, Any

from app.rag.embed import query_db


def retrieve_chunk_objs(query: str, k: int = 5) -> List[Dict[str, Any]]:
    results = query_db(query, k=k)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    cleaned = []
    seen_texts = set()

    for text, metadata, distance in zip(documents, metadatas, distances):
        text = text.strip()
        source = metadata.get("source", "Unknown")

        if not text:
            continue

        if len(text) < 40:
            continue

        normalized_text = " ".join(text.split())
        if normalized_text in seen_texts:
            continue

        seen_texts.add(normalized_text)

        cleaned.append(
            {
                "text": text,
                "source": source,
                "distance": distance,
                "metadata": metadata,
            }
        )


    return cleaned
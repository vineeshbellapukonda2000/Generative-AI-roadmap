from pathlib import Path
from app.rag.embed import upsert_chunks


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def load_text_files(folder_path: str):
    docs = []

    for file in Path(folder_path).glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            docs.append((f.read(), str(file)))

    return docs


def main():
    docs = load_text_files("data/docs")

    all_chunks = []
    all_metadata = []

    for text, source in docs:
        chunks = chunk_text(text)

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({"source": source})

    if not all_chunks:
        print("No documents found in data/docs")
        return

    upsert_chunks(all_chunks, all_metadata)
    print("Indexing complete!")


if __name__ == "__main__":
    main()
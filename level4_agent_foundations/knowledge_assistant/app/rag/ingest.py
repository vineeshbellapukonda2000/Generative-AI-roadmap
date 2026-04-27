from __future__ import annotations

from pathlib import Path
from typing import List, Dict


DOCS_DIR = Path(__file__).resolve().parents[2] / "data" / "docs"


def load_docs_with_sources(docs_dir: Path = DOCS_DIR) -> List[Dict[str, str]]:
    """
    Returns: [{"source": "doc1.txt", "text": "..."}]
    """
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs folder not found: {docs_dir}")

    files = sorted(docs_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in: {docs_dir}")

    docs: List[Dict[str, str]] = []
    for f in files:
        docs.append({
            "source": f.name,
            "text": f.read_text(encoding="utf-8")
        })
    return docs
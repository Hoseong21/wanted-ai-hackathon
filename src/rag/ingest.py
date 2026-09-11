"""문서(data/docs) -> 벡터DB(Chroma) 적재.

마크다운 문서를 '## ' 섹션 단위로 청킹해서 Chroma에 적재한다.
임베딩은 기본적으로 로컬 해싱 기반 임베딩(src.rag.embeddings)을 사용한다 (네트워크/API 키 불필요).
정확도를 높이고 싶으면 .env에서 EMBEDDING_BACKEND=openai 로 바꾸면 된다.

실행: python3 -m src.rag.ingest   (프로젝트 루트에서)
"""

from __future__ import annotations

import re
from pathlib import Path

import chromadb

from src.config import COLLECTION_NAME, DOCS_DIR, VECTORSTORE_DIR
from src.rag.embeddings import get_embedding_function


def _split_into_chunks(text: str, source: str) -> list[dict]:
    """마크다운을 '## ' 섹션 단위로 분할한다."""
    text = text.strip()
    title_line = text.splitlines()[0].lstrip("# ").strip()
    sections = re.split(r"\n(?=## )", text)

    chunks = []
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        chunks.append(
            {
                "id": f"{source}::chunk-{i}",
                "text": f"[{title_line}]\n{section}",
                "metadata": {"source": source, "chunk_index": i},
            }
        )
    return chunks


def load_documents(docs_dir: Path = DOCS_DIR) -> list[dict]:
    all_chunks: list[dict] = []
    for md_path in sorted(docs_dir.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        all_chunks.extend(_split_into_chunks(text, md_path.name))
    return all_chunks


def ingest(docs_dir: Path = DOCS_DIR, persist_dir: Path = VECTORSTORE_DIR) -> int:
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        COLLECTION_NAME, embedding_function=get_embedding_function()
    )

    chunks = load_documents(docs_dir)
    if not chunks:
        raise RuntimeError(f"{docs_dir}에 적재할 .md 문서가 없습니다.")

    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    return len(chunks)


if __name__ == "__main__":
    n = ingest()
    print(f"{n}개 청크를 '{COLLECTION_NAME}' 컬렉션에 적재했습니다. (persist dir: {VECTORSTORE_DIR})")

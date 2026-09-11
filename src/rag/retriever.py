"""벡터DB 검색.

실행: python3 -m src.rag.retriever "질문"   (프로젝트 루트에서, ingest 먼저 실행되어 있어야 함)
"""

from __future__ import annotations

import os

import chromadb

from src.config import COLLECTION_NAME, VECTORSTORE_DIR
from src.rag.embeddings import get_embedding_function
from src.rag.reranker import rerank

RERANK_CANDIDATE_N = int(os.getenv("RERANK_CANDIDATE_N", "9"))


def search_and_rerank(query: str, k: int = 3, candidate_n: int = RERANK_CANDIDATE_N) -> list[dict]:
    """1단계(Chroma) 넓게 검색 → 2단계(KURE-v2) 정밀 재순위. 실제로는 이 함수를 쓰면 됨."""
    candidates = search(query, k=candidate_n)
    return rerank(query, candidates, top_k=k)


def get_collection():
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    return client.get_or_create_collection(
        COLLECTION_NAME, embedding_function=get_embedding_function()
    )


def search(query: str, k: int = 3) -> list[dict]:
    """쿼리와 관련된 상위 k개 문서 청크를 반환한다.

    반환 항목: {"text": str, "source": str, "distance": float}
    distance는 값이 작을수록 더 관련도가 높다.
    """
    collection = get_collection()
    result = collection.query(query_texts=[query], n_results=k)

    hits = []
    for text, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append({
    "text": text,
    "source": meta.get("source"),
    "chunk_index": meta.get("chunk_index"),
    "distance": dist,
})
    return hits


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "50만원 이상 구매할 때 승인은 누구한테 받아야 해?"
    print(f"query: {query}\n")
    for hit in search(query):
        print(f"[{hit['source']}] (distance={hit['distance']:.4f})")
        print(hit["text"][:200])
        print("---")

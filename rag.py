from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = PROJECT_DIR / "knowledge_base"
CHROMA_DIR = PROJECT_DIR / "chroma_db"

COLLECTION_NAME = "nykaa_sentence_chunks"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# This was previously calibrated in the project. 0.55 was too strict.
SIMILARITY_THRESHOLD = 0.30


class RagNotReadyError(RuntimeError):
    pass


def load_documents() -> list[dict]:
    if not KNOWLEDGE_BASE_DIR.exists():
        raise RagNotReadyError(
            f"Knowledge base directory does not exist: {KNOWLEDGE_BASE_DIR}"
        )

    documents = []

    for file_path in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        document_id = file_path.stem.split("_", 1)[0]
        topic = file_path.stem.split("_", 1)[1]

        documents.append(
            {
                "document_id": document_id,
                "source": file_path.name,
                "topic": topic,
                "text": text,
                "sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
            }
        )

    if not documents:
        raise RagNotReadyError(
            "No Markdown documents were found in the knowledge base."
        )

    return documents


def sentence_based_chunking(documents: list[dict]) -> list[dict]:
    chunks = []

    for document in documents:
        sentences = re.split(r"(?<=[.!?])\s+", document["text"])

        for position, sentence in enumerate(sentences, start=1):
            sentence = sentence.strip()

            if not sentence:
                continue

            chunks.append(
                {
                    "chunk_id": (
                        f"{document['document_id']}_SENTENCE_{position}"
                    ),
                    "document_id": document["document_id"],
                    "source": document["source"],
                    "topic": document["topic"],
                    "document_sha256": document["sha256"],
                    "text": sentence,
                }
            )

    return chunks


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def build_index() -> dict:
    documents = load_documents()
    chunks = sentence_based_chunking(documents)

    model = get_embedding_model()
    client = get_chroma_client()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
    )

    collection.upsert(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "document_id": chunk["document_id"],
                "source": chunk["source"],
                "topic": chunk["topic"],
                "document_sha256": chunk["document_sha256"],
            }
            for chunk in chunks
        ],
    )

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "collection": COLLECTION_NAME,
    }


def get_collection():
    try:
        client = get_chroma_client()
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as error:
        raise RagNotReadyError(
            "RAG index is unavailable. Run `python rag.py --build-index` first."
        ) from error

    if collection.count() == 0:
        raise RagNotReadyError(
            "RAG index is empty. Run `python rag.py --build-index` first."
        )

    return collection


def get_index_status() -> dict:
    try:
        collection = get_collection()

        return {
            "ready": True,
            "collection": COLLECTION_NAME,
            "chunks": collection.count(),
        }
    except Exception as error:
        return {
            "ready": False,
            "reason": str(error),
        }


def expand_policy_query(query: str) -> str:
    """
    Adds policy vocabulary for natural customer language without replacing
    the original query.
    """
    query_lower = query.lower()
    hints = []

    if any(
        word in query_lower
        for word in [
            "lipstick",
            "beauty",
            "makeup",
            "cosmetic",
            "swatch",
            "swatched",
        ]
    ):
        hints.append(
            "Beauty product return policy, 7 days, unused, "
            "original packaging, opened product."
        )

    if any(
        word in query_lower
        for word in ["return", "exchange", "replace"]
    ):
        hints.append("return eligibility and exchange conditions")

    if not hints:
        return query

    return f"{query}\nRelevant policy concepts: {'; '.join(hints)}"


def grounded_retrieval(
    query: str,
    *,
    top_k: int = 3,
    threshold: float = SIMILARITY_THRESHOLD,
) -> dict:
    collection = get_collection()
    model = get_embedding_model()

    retrieval_query = expand_policy_query(query)

    query_embedding = model.encode(
        [retrieval_query],
        normalize_embeddings=True,
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    distances = results.get("distances", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]

    if not distances:
        return {
            "grounded": False,
            "similarity": 0.0,
            "context": [],
            "retrieval_query": retrieval_query,
        }

    context = []

    for chunk_id, document, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):
        similarity = max(0.0, min(1.0, 1 - float(distance)))

        if similarity < threshold:
            continue

        context.append(
            {
                "chunk_id": chunk_id,
                "text": document,
                "source": metadata["source"],
                "document_id": metadata["document_id"],
                "topic": metadata["topic"],
                "similarity": round(similarity, 4),
            }
        )

    top_similarity = max(0.0, min(1.0, 1 - float(distances[0])))

    return {
        "grounded": bool(context),
        "similarity": round(top_similarity, 4),
        "context": context,
        "retrieval_query": retrieval_query,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--build-index", action="store_true")
    args = parser.parse_args()

    if args.build_index:
        print(json.dumps(build_index(), indent=2))
    else:
        print(json.dumps(get_index_status(), indent=2))
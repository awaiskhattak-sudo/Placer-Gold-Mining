"""
build_vector_store_chroma.py
------------------------------
Reads embedded_chunks.json (produced by embed_chunks.py) and loads it into a
local, persistent Chroma vector store.

Input:
    embedded_chunks.json -> [{"source", "chunk_index", "text", "embedding", ...}, ...]

Output:
    chroma_db/   -> a persistent Chroma database directory (created automatically).
                    Unlike FAISS, Chroma stores the vector, the original text,
                    and the metadata together in one place — no separate
                    metadata.json file needed.

Install:
    pip install chromadb
"""

import os
import json
import chromadb

INPUT_FILE = "embedded_chunks.json"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "gold_mining_kb"


def load_embedded_chunks(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Run embed_chunks.py first to generate it."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_chroma_store(chunks: list[dict]):
    # PersistentClient writes to disk at CHROMA_PATH so the data survives
    # between script runs and across your FastAPI app's restarts.
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # get_or_create_collection: safe to re-run this script — it won't error
    # if the collection already exists from a previous run.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for semantic search
    )

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for chunk in chunks:
        # A stable, unique ID per chunk — combining source filename and chunk
        # index means re-running this script updates existing chunks in place
        # (via upsert) rather than creating duplicates.
        chunk_id = f"{chunk['source']}_{chunk['chunk_index']}"

        ids.append(chunk_id)
        embeddings.append(chunk["embedding"])
        documents.append(chunk["text"])
        metadatas.append({
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "char_count": chunk["char_count"],
        })

    # upsert (not add) so re-running this script after re-embedding updates
    # existing entries instead of erroring on duplicate IDs.
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    return collection


def main():
    print(f"Loading embedded chunks from '{INPUT_FILE}'...")
    chunks = load_embedded_chunks(INPUT_FILE)
    print(f"Loaded {len(chunks)} embedded chunk(s).")

    print(f"Writing to Chroma collection '{COLLECTION_NAME}' at '{CHROMA_PATH}/'...")
    collection = build_chroma_store(chunks)

    print(f"\nDone. Collection '{COLLECTION_NAME}' now contains {collection.count()} chunk(s).")
    print(f"Persisted to disk at '{CHROMA_PATH}/'.")


if __name__ == "__main__":
    main()

"""
chunker.py
----------
Chunks the 3 gold-mining knowledge base documents for the RAG pipeline using
LangChain's RecursiveCharacterTextSplitter.

Input documents:
    04_gold_operation_shakardara_indus.txt
    05_requirement.txt
    06_finance.txt

Output:
    chunks.json  -> a list of {"source": ..., "chunk_index": ..., "text": ...}
                    ready to be fed into an embedding step.

Install:
    pip install langchain-text-splitters
"""

import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_DIR = "docs"          # folder where the 3 .txt files live
OUTPUT_FILE = "chunks.json"

SOURCE_FILES = [
    "04_gold_operation_shakardara_indus.txt",
    "05_requirement.txt",
    "06_finance.txt",
]

CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 120    # overlap between consecutive chunks, preserves context across cuts

# Priority order: try splitting on bigger structural breaks first (headers,
# blank lines), then fall back to sentence/word boundaries only if needed.
# This keeps section headers and tables grouped with the content they describe.
SEPARATORS = [
    "\n\n",   # paragraph / section breaks
    "\n",     # line breaks (e.g. table rows, bullet points)
    ". ",     # sentence boundaries
    " ",      # word boundaries (last resort)
    "",
]


def load_documents(input_dir: str, filenames: list[str]) -> list[dict]:
    """Reads each source file and returns its raw text plus filename."""
    documents = []
    for filename in filenames:
        path = os.path.join(input_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected '{filename}' in '{input_dir}/'. "
                f"Place the 3 source .txt files there before running this script."
            )
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append({"source": filename, "text": text})
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Splits each document's text into overlapping chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )

    all_chunks = []
    for doc in documents:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            all_chunks.append(
                {
                    "source": doc["source"],
                    "chunk_index": i,
                    "text": piece.strip(),
                    "char_count": len(piece.strip()),
                }
            )
    return all_chunks


def main():
    print(f"Loading documents from '{INPUT_DIR}/'...")
    documents = load_documents(INPUT_DIR, SOURCE_FILES)
    print(f"Loaded {len(documents)} document(s).")

    print(f"Chunking with chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}...")
    chunks = chunk_documents(documents)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Created {len(chunks)} chunk(s) total.")
    for doc in documents:
        count = sum(1 for c in chunks if c["source"] == doc["source"])
        print(f"  - {doc['source']}: {count} chunk(s)")
    print(f"\nSaved to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()

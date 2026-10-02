"""
embed_chunks.py
----------------
Reads chunks.json (produced by chunker.py) and generates a vector embedding
for each chunk using Google's Gemini embedding model.

Input:
    chunks.json  -> [{"source", "chunk_index", "text", "char_count"}, ...]

Output:
    embedded_chunks.json -> same records, each with an added "embedding" field
                             (a list of floats), ready to load into a vector
                             store (FAISS/Chroma) in the next step.

Install:
    pip install google-genai python-dotenv

.env required:
    GEMINI_API_KEY=your_actual_gemini_api_key_here
"""

import os
import json
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

INPUT_FILE = "chunks.json"
OUTPUT_FILE = "embedded_chunks.json"

EMBEDDING_MODEL = "gemini-embedding-001"
OUTPUT_DIMENSIONALITY = 768   # truncated from the model's default 3072 dims
TASK_TYPE = "RETRIEVAL_DOCUMENT"  # use RETRIEVAL_QUERY when embedding a search query later

RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2


def load_chunks(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Run chunker.py first to generate it."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_text(client: genai.Client, text: str) -> list[float]:
    """Embeds a single chunk of text, retrying on transient API errors."""
    last_error = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type=TASK_TYPE,
                    output_dimensionality=OUTPUT_DIMENSIONALITY,
                ),
            )
            return response.embeddings[0].values
        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt} failed: {e}")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY_SECONDS)
    raise RuntimeError(f"Failed to embed chunk after {RETRY_ATTEMPTS} attempts: {last_error}")


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found. Add it to a .env file in this directory."
        )

    client = genai.Client(api_key=api_key)

    print(f"Loading chunks from '{INPUT_FILE}'...")
    chunks = load_chunks(INPUT_FILE)
    print(f"Loaded {len(chunks)} chunk(s). Embedding with '{EMBEDDING_MODEL}' "
          f"(dim={OUTPUT_DIMENSIONALITY}, task_type={TASK_TYPE})...\n")

    embedded_chunks = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"[{i}/{len(chunks)}] Embedding {chunk['source']} "
              f"(chunk {chunk['chunk_index']}, {chunk['char_count']} chars)...")

        vector = embed_text(client, chunk["text"])

        embedded_chunks.append({
            **chunk,
            "embedding": vector,
            "embedding_dim": len(vector),
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f)

    print(f"\nDone. Saved {len(embedded_chunks)} embedded chunk(s) to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()

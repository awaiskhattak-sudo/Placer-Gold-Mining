"""
rag_pipeline.py
-----------------
The core retrieval-augmented generation (RAG) engine for the Placer Gold
Mining AI Search Engine.

Flow for a user question:
    1. Embed the question using Gemini (task_type=RETRIEVAL_QUERY)
    2. Search the persisted Chroma collection for the top-k most similar chunks
    3. Pass the question + retrieved chunks to a local Ollama model to generate a grounded answer
    4. Run a second local Ollama call to validate the answer is actually supported by the context
    5. Return the answer, sources, and validation verdict

Architecture note: embeddings use Gemini's API (for retrieval quality); generation
and validation use a local Ollama model (no API key/internet needed for those steps).

Install:
    pip install google-genai ollama chromadb python-dotenv

.env required:
    GEMINI_API_KEY=your_actual_gemini_api_key_here

Also required:
    Ollama running locally with the model pulled:
        ollama pull llama3.2:1b
"""

import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
import ollama
import chromadb

load_dotenv()

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "gold_mining_kb"

# Embeddings: still Gemini (gemini-embedding-001) — unchanged.
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768

# Generation + validation: now a local Ollama model — no API key, no internet
# needed for these two calls. Requires `ollama pull llama3.2:1b` and the
# Ollama service running locally (default: http://localhost:11434).
OLLAMA_MODEL = "phi3:latest"

TOP_K = 4  # number of chunks retrieved per query

# Relevance gate: Chroma returns cosine DISTANCE (0 = identical, 2 = opposite).
# If even the closest retrieved chunk is farther than this threshold, the
# question is treated as out-of-scope for this knowledge base — the pipeline
# skips generation entirely and asks the user to rephrase, rather than letting
# the LLM guess at an answer with no real supporting source.
# NOTE: 0.6 is a reasonable starting point but should be tuned by testing a
# handful of clearly in-scope and clearly out-of-scope questions against your
# actual data and adjusting up/down based on the distances you observe.
RELEVANCE_DISTANCE_THRESHOLD = 0.6

SYSTEM_INSTRUCTION = """You are a knowledgeable assistant for a placer gold mining knowledge base.
Answer the user's question using ONLY the provided context chunks below.
If the context does not contain enough information to answer confidently, say so clearly —
do not guess or invent information that is not present in the context.
Keep answers concise, specific, and directly grounded in the provided context."""

VALIDATION_SYSTEM_INSTRUCTION = """You are a strict fact-checker reviewing an AI-generated answer against
the source context it was supposed to be based on.
Your job is to judge whether the answer is fully supported by the context — not whether the
answer sounds reasonable or well-written.
Flag the answer as invalid if it states any fact, number, or claim that is not present in
the context, even if that fact happens to be true in general.
Respond ONLY with a JSON object in this exact shape:
{"valid": true or false, "confidence": a number from 0 to 100, "reasoning": "one or two sentences explaining your judgment"}"""


_client = None


def get_client() -> genai.Client:
    """Lazily initializes a single shared Gemini client."""
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not found. Add it to your .env file.")
        _client = genai.Client(api_key=api_key)
    return _client


def get_collection():
    """Connects to the persisted Chroma collection built by build_vector_store_chroma.py."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma_client.get_collection(COLLECTION_NAME)


def embed_query(question: str) -> list[float]:
    """Embeds a user question using the RETRIEVAL_QUERY task type."""
    client = get_client()
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",   # note: different from RETRIEVAL_DOCUMENT used for chunks
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    return response.embeddings[0].values


def retrieve_chunks(question: str, top_k: int = TOP_K) -> list[dict]:
    """Embeds the question and retrieves the top-k most similar chunks from Chroma."""
    collection = get_collection()
    query_vector = embed_query(question)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "distance": results["distances"][0][i],  # lower = more similar (cosine distance)
        })
    return retrieved


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    """Generates an answer grounded in the retrieved chunks, using a local Ollama model."""
    context_block = "\n\n---\n\n".join(
        f"[Source: {c['source']}, chunk {c['chunk_index']}]\n{c['text']}"
        for c in retrieved_chunks
    )

    prompt = f"""CONTEXT:
{context_block}

---

QUESTION: {question}

Answer the question using only the context above."""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.2},
    )
    return response["message"]["content"]


def validate_answer(question: str, answer: str, retrieved_chunks: list[dict]) -> dict:
    """
    Groundedness check: a second, independent LLM call that judges whether the
    generated answer is actually supported by the retrieved context, rather than
    hallucinated or drifted. This directly satisfies the "check if the answer is
    valid" requirement of the project by catching unsupported claims before they
    reach the user.
    """
    context_block = "\n\n---\n\n".join(c["text"] for c in retrieved_chunks)

    validation_prompt = f"""CONTEXT:
{context_block}

---

QUESTION: {question}

ANSWER TO VALIDATE: {answer}

Judge whether the answer above is fully supported by the context."""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": VALIDATION_SYSTEM_INSTRUCTION},
            {"role": "user", "content": validation_prompt},
        ],
        format="json",  # constrains Ollama's output to valid JSON
        options={"temperature": 0.0},  # deterministic judging, no creativity wanted here
    )

    try:
        return json.loads(response["message"]["content"])
    except (json.JSONDecodeError, KeyError):
        # Fail safe: if the judge call itself breaks, don't silently claim validity
        return {"valid": False, "confidence": 0, "reasoning": "Validation step failed to return a parseable result."}


def ask(question: str, top_k: int = TOP_K) -> dict:
    """
    Full RAG pipeline for one question: retrieve + generate + validate.
    Returns the answer, its source chunks (for transparency/citation), and a
    groundedness verdict on whether the answer is actually supported by those sources.
    """
    retrieved_chunks = retrieve_chunks(question, top_k=top_k)

    # --- Relevance gate: does the question even match our knowledge base? ---
    best_distance = min((c["distance"] for c in retrieved_chunks), default=None)

    if best_distance is None or best_distance > RELEVANCE_DISTANCE_THRESHOLD:
        return {
            "question": question,
            "answer": (
                "I couldn't find anything in the placer gold mining knowledge base "
                "closely related to that question. Please ask something about the "
                "Shakardara operation, its equipment/fuel requirements, or its finances."
            ),
            "sources": [],
            "validation": {
                "valid": False,
                "confidence": 0,
                "reasoning": (
                    f"No retrieved chunk was close enough to the question "
                    f"(closest distance={best_distance:.4f}, threshold={RELEVANCE_DISTANCE_THRESHOLD})."
                    if best_distance is not None else
                    "No chunks were retrieved from the knowledge base."
                ),
            },
        }

    # --- Question is in scope: generate an answer and groundedness-check it ---
    answer = generate_answer(question, retrieved_chunks)
    validation = validate_answer(question, answer, retrieved_chunks)

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {"source": c["source"], "chunk_index": c["chunk_index"], "distance": c["distance"]}
            for c in retrieved_chunks
        ],
        "validation": validation,
    }


if __name__ == "__main__":
    # Quick manual test from the command line
    test_question = "How many hours does the excavator run and how much diesel does it use?"
    result = ask(test_question)

    print(f"Q: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    print("Sources used:")
    for s in result["sources"]:
        print(f"  - {s['source']} (chunk {s['chunk_index']}, distance={s['distance']:.4f})")
    print(f"\nValidation: {result['validation']}")

"""
main.py
--------
FastAPI application for the Placer Gold Mining AI Search Engine.

Wraps rag_pipeline.ask() behind a simple Jinja2 + HTMX search page:
    GET  /         -> search page
    POST /search   -> runs the RAG pipeline, returns a results HTML fragment

Install:
    pip install fastapi uvicorn[standard] jinja2 python-multipart
"""

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import dotenv
import rag_pipeline
from google import genai
app = FastAPI(title="Placer Gold Mining AI Search Engine")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search", response_class=HTMLResponse)
def search(request: Request, question: str = Form(...)):
    question = question.strip()

    if not question:
        return templates.TemplateResponse(
            "results.html",
            {"request": request, "error": "Please enter a question."},
        )

    result = rag_pipeline.ask(question)

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "question": result["question"],
            "answer": result["answer"],
            "sources": result["sources"],
            "validation": result["validation"],
            "error": None,
        },
    )


@app.get("/health")
def health_check():
    """Simple endpoint to confirm the app (and later, the Docker container) is up."""
    return {"status": "ok"}

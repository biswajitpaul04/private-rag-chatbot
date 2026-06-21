"""
app.py — FastAPI web server for the banking RAG chatbot.

Serves a chat UI at http://localhost:8000 and exposes a /chat endpoint
that runs a question through the RAG pipeline (retrieve + local LLM)
and returns the answer plus its sources.

Run (with Ollama running in the background, inside your venv):
    python app.py
Then open http://localhost:8000 in your browser.
"""

from functools import lru_cache

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# RetrievalQA moved to langchain-classic in langchain v1.x.
# This try/except works on both new and old versions.
try:
    from langchain_classic.chains import RetrievalQA   # langchain v1.x
except ImportError:
    from langchain.chains import RetrievalQA           # older langchain

# ---------------------------------------------------------------------------
# Configuration — embedding model MUST match ingest.py
# ---------------------------------------------------------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_DIR = "db/chroma_db"
ALLOWED_MODELS = {"llama3.2", "mistral"}   # models the UI can pick from

PROMPT = PromptTemplate(
    template=(
        "You are a helpful assistant for a bank. "
        "Use ONLY the context below to answer the question.\n"
        "If the answer isn't in the context, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    input_variables=["context", "question"],
)

app = FastAPI(title="Banking RAG Chatbot")

# The embedding model is loaded once and shared across all engines.
_embeddings = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


@lru_cache(maxsize=len(ALLOWED_MODELS))
def get_engine(model_name: str):
    """Build (and cache) a RetrievalQA engine for the given LLM."""
    store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=_get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )
    retriever = store.as_retriever(search_kwargs={"k": 4})
    llm = ChatOllama(model=model_name, temperature=0.2)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )


class ChatRequest(BaseModel):
    question: str
    model: str = "llama3.2"


@app.post("/chat")
def chat(req: ChatRequest):
    """Answer one question and return the answer + unique sources."""
    model = req.model if req.model in ALLOWED_MODELS else "llama3.2"

    if not req.question.strip():
        return {"answer": "Please type a question.", "sources": []}

    try:
        engine = get_engine(model)
        result = engine.invoke({"query": req.question})
    except Exception as e:
        return {
            "answer": f"Error: {e}. Is Ollama running and has ingest.py been run?",
            "sources": [],
        }

    sources, seen = [], set()
    for doc in result["source_documents"]:
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            sources.append(src)
            seen.add(src)

    return {"answer": result["result"], "sources": sources}


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the chat UI."""
    with open("chat.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    print("Starting server at http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
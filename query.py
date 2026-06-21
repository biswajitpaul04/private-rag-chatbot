"""
query.py — Ask questions against the ingested documents (terminal version).

Loads the Chroma vector store built by ingest.py, retrieves the most
relevant chunks for your question, and uses a local Ollama LLM to write
a grounded answer with its sources.

Run (with Ollama running, inside your venv):
    python query.py
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate

# RetrievalQA moved to langchain-classic in langchain v1.x.
try:
    from langchain_classic.chains import RetrievalQA   # langchain v1.x
except ImportError:
    from langchain.chains import RetrievalQA           # older langchain

# ---------------------------------------------------------------------------
# Configuration — MUST match ingest.py for the embedding model
# ---------------------------------------------------------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_DIR = "db/chroma_db"
LLM_MODEL = "llama3.2"        # change to "mistral" to use the other model

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


def build_engine(k=4):
    """Load the vector store and wire up retrieval + the LLM."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    retriever = store.as_retriever(search_kwargs={"k": k})
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )


def main():
    print("Loading vector store and model...")
    try:
        engine = build_engine()
    except Exception as e:
        print(f"Could not start the engine: {e}")
        print("   Did you run `python ingest.py` first?")
        return

    print("\nRAG ready. Ask a question, or type 'exit' to quit.\n")

    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not question:
            continue

        try:
            result = engine.invoke({"query": question})
        except Exception as e:
            print(f"Error answering: {e}")
            print("   Is Ollama running? Try `ollama list`.\n")
            continue

        print("\nAnswer:", result["result"])

        print("\nSources:")
        seen = set()
        for doc in result["source_documents"]:
            src = doc.metadata.get("source", "unknown")
            if src not in seen:
                print("  -", src)
                seen.add(src)
        print()


if __name__ == "__main__":
    main()
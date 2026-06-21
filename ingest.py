"""
ingest.py — Multi-source RAG ingestion pipeline (banking demo).

Loads documents from local files (and optionally Google Drive / web),
splits them into chunks, embeds them with a local model, and stores the
vectors in a persistent Chroma database.

Run once (or whenever your data changes):
    python ingest.py

Notes:
- Heavy libraries are imported *inside* the functions that use them
  (lazy import). This keeps startup fast and avoids import-time crashes
  from unrelated packages.
- Ollama is NOT needed here — only at query time.
"""

import io
import json
import os
import shutil
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_DIR = "db/chroma_db"


def log(step, message):
    """Print a clearly labelled step line."""
    print(f"\n[STEP {step}] {message}")


# ===========================================================================
# STEP 1 — Local file loader
# ===========================================================================
def load_local(docs_path="docs"):
    """Read every supported file in docs/ into a list of Documents."""
    from langchain_core.documents import Document
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    import pandas as pd

    documents = []
    folder = Path(docs_path)

    if not folder.exists():
        print(f"   WARNING: '{docs_path}' folder not found - skipping local files.")
        return documents

    print(f"   Scanning '{docs_path}' for txt, pdf, json, csv files...")

    for txt in folder.rglob("*.txt"):
        docs = TextLoader(str(txt), encoding="utf-8").load()
        for d in docs:
            d.metadata["type"] = "local_txt"
        documents.extend(docs)
        print(f"   [OK] txt  : {txt.name}")

    for pdf in folder.rglob("*.pdf"):
        docs = PyPDFLoader(str(pdf)).load()
        for d in docs:
            d.metadata["type"] = "local_pdf"
        documents.extend(docs)
        print(f"   [OK] pdf  : {pdf.name}")

    for jf in folder.rglob("*.json"):
        data = json.loads(jf.read_text(encoding="utf-8"))
        documents.append(Document(
            page_content=json.dumps(data, indent=2),
            metadata={"source": str(jf), "type": "local_json"},
        ))
        print(f"   [OK] json : {jf.name}")

    for cf in folder.rglob("*.csv"):
        df = pd.read_csv(cf)
        documents.append(Document(
            page_content=df.to_string(),
            metadata={"source": str(cf), "type": "local_csv"},
        ))
        print(f"   [OK] csv  : {cf.name}")

    print(f"   Local total: {len(documents)} documents")
    return documents


# ===========================================================================
# STEP 2 — Web loader  (optional)
# ===========================================================================
def load_web(urls=None, crawl_url=None, max_pages=15):
    """Load specific URLs and/or crawl one site."""
    documents = []

    if not urls and not crawl_url:
        print("   (no web sources configured - skipping)")
        return documents

    from langchain_core.documents import Document

    for url in (urls or []):
        text = _fetch_clean_text(url)
        if text:
            documents.append(Document(
                page_content=text,
                metadata={"source": url, "type": "web_url"},
            ))
            print(f"   [OK] web  : {url}")
        time.sleep(1)

    if crawl_url:
        documents.extend(_crawl(crawl_url, max_pages))

    print(f"   Web total: {len(documents)} documents")
    return documents


def _fetch_clean_text(url):
    import requests
    from bs4 import BeautifulSoup
    try:
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as e:
        print(f"   [FAIL] {url}: {e}")
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return " ".join(soup.get_text().split())


def _crawl(start_url, max_pages):
    import requests
    from bs4 import BeautifulSoup
    from langchain_core.documents import Document

    documents, visited, queue = [], set(), [start_url]
    domain = urlparse(start_url).netloc

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        try:
            resp = requests.get(url, timeout=10,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
        except Exception as e:
            print(f"   [FAIL] {url}: {e}")
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = " ".join(soup.get_text().split())
        if text:
            documents.append(Document(
                page_content=text,
                metadata={"source": url, "type": "web_crawled"},
            ))
            print(f"   [OK] crawl: {url}")
        for a in soup.find_all("a", href=True):
            nxt = urljoin(url, a["href"])
            if urlparse(nxt).netloc == domain and nxt not in visited:
                queue.append(nxt)
        time.sleep(1)

    return documents


# ===========================================================================
# STEP 3 — Google Drive loader  (optional, disabled by default)
# ===========================================================================
def load_drive(folder_id=None, creds_path="service_account.json"):
    """Load PDFs and text files from a Google Drive folder."""
    if not folder_id:
        print("   (no Drive folder configured - skipping)")
        return []

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
        from pypdf import PdfReader
        from langchain_core.documents import Document
    except ImportError:
        print("   WARNING: Google libraries not installed - skipping Drive.")
        return []

    creds = service_account.Credentials.from_service_account_file(
        creds_path,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    service = build("drive", "v3", credentials=creds)
    res = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id, name, mimeType)",
    ).execute()
    files = res.get("files", [])
    print(f"   Found {len(files)} files in Drive folder")

    documents = []
    for f in files:
        data = _download(service, f["id"], MediaIoBaseDownload)
        if f["mimeType"] == "application/pdf":
            text = "".join(p.extract_text() or "" for p in PdfReader(data).pages)
        elif f["mimeType"].startswith("text/"):
            text = data.read().decode("utf-8")
        else:
            continue
        documents.append(Document(
            page_content=text,
            metadata={"source": f"drive/{f['name']}", "type": "drive"},
        ))
        print(f"   [OK] drive: {f['name']}")

    print(f"   Drive total: {len(documents)} documents")
    return documents


def _download(service, file_id, MediaIoBaseDownload):
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, service.files().get_media(fileId=file_id))
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    return buf


# ===========================================================================
# STEP 4 — Chunk
# ===========================================================================
def chunk(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into overlapping chunks."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"   Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks


# ===========================================================================
# STEP 5 — Embed and store
# ===========================================================================
def build_store(chunks, persist_dir=DB_DIR, reset=False):
    """Embed chunks and persist them in Chroma."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    if reset and os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print("   Cleared existing vector store")

    print("   Loading embedding model (first run downloads ~80MB)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("   Embedding chunks and writing to Chroma...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"   Stored {len(chunks)} chunks -> {persist_dir}")


# ===========================================================================
# Main pipeline
# ===========================================================================
def main():
    print("=" * 60)
    print("  RAG INGESTION PIPELINE")
    print("=" * 60)

    log(1, "Loading local files")
    documents = load_local("docs")

    log(2, "Loading web sources")
    documents += load_web(urls=None, crawl_url=None)

    log(3, "Loading Google Drive")
    documents += load_drive(folder_id=None)

    if not documents:
        print("\nERROR: No documents loaded. Add files to docs/ and try again.")
        return

    print(f"\n   Total documents from all sources: {len(documents)}")

    log(4, "Splitting documents into chunks")
    chunks = chunk(documents)

    log(5, "Embedding chunks and building the vector store")
    build_store(chunks, reset=True)

    print("\n" + "=" * 60)
    print("  INGESTION COMPLETE - vector store is ready.")
    print("  Next: run  python app.py  (then open http://localhost:8000)")
    print("=" * 60)


if __name__ == "__main__":
    main()

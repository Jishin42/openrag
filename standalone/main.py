from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Optional
import requests
import uuid
import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import tempfile


# --- Config ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3") #choose your Ollama model here (e.g., "llama3", "mistral", "gemini-1.5-pro", etc. cf https://ollama.com/library)

# --- Init FastAPI ---
app = FastAPI(title="OpenRAG Standalone (Ollama)")
templates = Jinja2Templates(directory="standalone/templates")

# --- Embedding model ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Vector DB ---
chroma_client = chromadb.Client(
    settings=chromadb.config.Settings(
        persist_directory="./chroma_db",
        is_persistent=False, #reinit db at each start
    )
)
collection = chroma_client.get_or_create_collection("documents")
ram_size_bytes = 0

# --- Schemas ---
class Document(BaseModel):
    id: str
    text: str

class Query(BaseModel):
    query: str
    top_k: Optional[int] = 3

# --- Ollama call ---
def ollama_generate(prompt: str) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        return response.json().get("response", "")
    except Exception as e:
        return f"Ollama error: {e}"

# Common function to process multiple PDF files
def process_pdf_files(files: List[UploadFile]) -> int:
    total_chunks = 0

    for file in files:
        try:
            # Save uploaded file to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file.file.read())
                temp_file_path = temp_file.name

            # Use PyPDFLoader for better parsing
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()

            # Extract text from all pages
            full_text = "\n".join([doc.page_content for doc in documents])

            # Clean up temp file
            os.unlink(temp_file_path)

            if not full_text.strip():
                continue

            # Improved chunking with larger chunks and more overlap
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Increased from 400
                chunk_overlap=200,  # Increased from 80
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = splitter.split_text(full_text)

            if not chunks:
                continue

            ids = [str(uuid.uuid4()) for _ in chunks]
            embeddings = embedding_model.encode(chunks).tolist()
            metadatas = [{"filename": file.filename} for _ in chunks]

            collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )
            chunk_bytes = sum(len(chunk.encode('utf-8')) for chunk in chunks)
            global ram_size_bytes
            ram_size_bytes += chunk_bytes
            total_chunks += len(chunks)

        except Exception as e:
            print(f"Error processing {file.filename}: {e}")

    return total_chunks

def human_readable_size(size_bytes: int) -> str:
    if size_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(units) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f} {units[i]}"

# --- Routes ---
@app.get("/count_chunks")
def count_chunks():
    count = collection.count()
    db_dir = os.path.abspath("./chroma_db")
    total_size = 0
    if os.path.isdir(db_dir):
        for root, _, files in os.walk(db_dir):
            for name in files:
                try:
                    total_size += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
    return {
        "count": count,
        "size_bytes": total_size,
        "size_human": human_readable_size(total_size),
        "ram_bytes": ram_size_bytes,
        "ram_human": human_readable_size(ram_size_bytes)
    }

@app.get("/ui")
def ui(request: Request):
    return templates.TemplateResponse("ui.html", {"request": request})

@app.post("/upload_pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    if not files:
        return {"status": "no files provided"}

    chunks = process_pdf_files(files)

    return {
        "status": "PDFs indexed",
        "files": len(files),
        "chunks": chunks
    }

@app.post("/query")
def query(q: Query):
    embedding = embedding_model.encode(q.query).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=q.top_k)
    return {"results": results.get("documents", [[]])[0]}

@app.post("/rag")
def rag(q: Query):
    embedding = embedding_model.encode(q.query).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=q.top_k, include=["documents", "metadatas"])
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    context = "\n".join([f"Source: {meta['filename']}\n{doc}" for doc, meta in zip(docs, metas)])

    prompt = f"Contexte:\n{context}\n\nQuestion: {q.query}\nRépondre à la question basée sur le contexte, en citant les sources."
    answer = ollama_generate(prompt)

    return {
        "query": q.query,
        "context": docs,
        "answer": answer,
        "sources": metas
    }

@app.delete("/reset")
def reset_db():
    # Delete the Chroma collection and any persisted files so size resets correctly.
    chroma_client.delete_collection("documents")

    db_dir = os.path.abspath("./chroma_db")
    if os.path.isdir(db_dir):
        try:
            shutil.rmtree(db_dir)
        except Exception:
            pass

    global collection, ram_size_bytes
    collection = chroma_client.get_or_create_collection("documents")
    ram_size_bytes = 0
    return {"status": "reset"}

# --- Run ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("standalone.main:app", host="0.0.0.0", port=8000, reload=True)

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
import pdfplumber
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Optional
import requests
import uuid
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter



# --- Config ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# --- Init FastAPI ---
app = FastAPI(title="OpenRAG Standalone (Ollama)")
templates = Jinja2Templates(directory="standalone/templates")

# --- Embedding model ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Vector DB ---
chroma_client = chromadb.Client(
    settings=chromadb.config.Settings(
        persist_directory="./chroma_db",
        is_persistent=True
    )
)
collection = chroma_client.get_or_create_collection("documents")

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

# --- Routes ---
@app.get("/")
def root():
    return {"status": "running"}

@app.get("/ui")
def ui(request: Request):
    return templates.TemplateResponse("ui.html", {"request": request})

@app.post("/add")
def add_document(doc: Document):
    embedding = embedding_model.encode(doc.text).tolist()
    collection.add(ids=[doc.id], documents=[doc.text], embeddings=[embedding])
    return {"status": "added"}

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        from pypdf import PdfReader

        texts = []
        with pdfplumber.open(file.file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)

        # better chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=80
        )
        chunks = splitter.split_text("\n".join(texts))

        if not chunks:
            return {"status": "empty PDF"}

        ids = [str(uuid.uuid4()) for _ in chunks]
        embeddings = embedding_model.encode(chunks).tolist()

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings
        )
        print("TEXT LENGTH:", sum(len(t) for t in texts))
        print("NB PAGES:", len(texts))

        return {"status": "PDF indexed", "chunks": len(chunks)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query(q: Query):
    embedding = embedding_model.encode(q.query).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=q.top_k)
    return {"results": results.get("documents", [[]])[0]}

@app.post("/rag")
def rag(q: Query):
    embedding = embedding_model.encode(q.query).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=q.top_k)
    docs = results.get("documents", [[]])[0]
    context = "\n".join(docs)

    prompt = f"Context:\n{context}\n\nQuestion: {q.query}\nAnswer:"
    answer = ollama_generate(prompt)

    return {
        "query": q.query,
        "context": docs,
        "answer": answer
    }

@app.delete("/reset")
def reset_db():
    chroma_client.delete_collection("documents")
    global collection
    collection = chroma_client.get_or_create_collection("documents")
    return {"status": "reset"}

# --- Run ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("standalone.main:app", host="0.0.0.0", port=8000, reload=True)

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Optional
import requests
import os

# --- Config ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# --- Init FastAPI ---
app = FastAPI(title="OpenRAG Standalone (Ollama)")

# --- Embedding model ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Vector DB ---
chroma_client = chromadb.Client(
    settings=chromadb.config.Settings(
        persist_directory="./chroma_db"
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

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
    <html>
    <head><title>OpenRAG UI</title></head>
    <body>
    <h2>OpenRAG (Ollama)</h2>
    <textarea id=\"query\" rows=4 cols=60 placeholder=\"Ask something...\"></textarea><br>
    <button onclick=\"send()\">Ask</button>
    <pre id=\"result\"></pre>

    <script>
    async function send() {
        const query = document.getElementById('query').value;
        const res = await fetch('/rag', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query: query})
        });
        const data = await res.json();
        document.getElementById('result').textContent = JSON.stringify(data, null, 2);
    }
    </script>
    </body>
    </html>
    """

@app.post("/add")
def add_document(doc: Document):
    embedding = embedding_model.encode(doc.text).tolist()
    collection.add(ids=[doc.id], documents=[doc.text], embeddings=[embedding])
    return {"status": "added"}

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

# --- Run ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("standalone.main:app", host="0.0.0.0", port=8000, reload=True)


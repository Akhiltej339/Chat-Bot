import os
import openai
import faiss
import pickle
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

INDEX_PATH = "vector_store/index.faiss"
META_PATH = "vector_store/meta.pkl"

index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    documents, metadata = pickle.load(f)

app = FastAPI()

# Serve images and static files
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

class Query(BaseModel):
    question: str

def get_embedding(text):
    res = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return res.data[0].embedding

def search(query, top_k=3):
    query_vec = np.array([get_embedding(query)]).astype("float32")
    D, I = index.search(query_vec, top_k)
    results = []
    for i in I[0]:
        results.append({
            "text": documents[i],
            "metadata": metadata[i]
        })
    return results

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask(query: Query):
    results = search(query.question)
    # Combine text and images from top results
    answer_text = "\n\n".join([r["text"] for r in results])
    images = []
    for r in results:
        images.extend(r["metadata"].get("images", []))
    images = list(dict.fromkeys(images))  # unique images

    prompt = f"""You are an MDM support assistant. Use the information below to answer the question:\n\n{answer_text}\n\nQuestion: {query.question}\nAnswer:"""

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer_text": response.choices[0].message.content,
        "images": [f"/static/images/{img}" for img in images]
    }

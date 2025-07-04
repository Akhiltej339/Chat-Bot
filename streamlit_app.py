import streamlit as st
import os
import openai
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

INDEX_PATH = "vector_store/index.faiss"
META_PATH = "vector_store/meta.pkl"
IMAGE_DIR = "static/images"

# Load FAISS index and metadata once
try:
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        documents, metadata = pickle.load(f)
except Exception as e:
    st.error(f"Error loading FAISS index: {e}")
    st.stop()

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

def generate_answer(context_text, question):
    prompt = f"""You are an MDM support assistant. Use the information below to answer the question:\n\n{context_text}\n\nQuestion: {question}\nAnswer:"""
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- Streamlit UI ---
st.title("WeGuard MDM Chat Bot")

# Chat session state
st.session_state.chat_history = []

user_input = st.text_input("Ask your question about MDM...")

if user_input:
    # ✅ Always start with a clean list
    images = []

    # Step 1: Search for relevant content
    search_results = search(user_input)
    context_text = "\n\n".join([r["text"] for r in search_results])

    for r in search_results:
        # ✅ Only collect images for this question
        images.extend(r["metadata"].get("images", []))

    # ✅ Remove duplicates if needed
    images = list(dict.fromkeys(images))

    # Step 2: Generate answer
    answer = generate_answer(context_text, user_input)

    # Step 3: Show results
    st.markdown(f"**You:** {user_input}")
    st.markdown(f"**Bot:** {answer}")
    for img in images:
        img_path = os.path.join(IMAGE_DIR, img)
        if os.path.exists(img_path):
            st.image(img_path, caption=img)

# Render chat history
for i, chat in enumerate(st.session_state.chat_history):
    st.markdown(f"**You:** {chat['question']}")
    st.markdown(f"**Bot:** {chat['answer']}")
    for img in chat.get("images", []):
        img_path = os.path.join(IMAGE_DIR, img)
        if os.path.exists(img_path):
            st.image(img_path, caption=img)
        else:
            st.warning(f"Image not found: {img}")

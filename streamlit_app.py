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

# Load index and metadata once
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    documents, metadata = pickle.load(f)

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

# Streamlit UI

st.title("MDM AI Chatbot with PDF Images")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def chat():
    user_question = st.text_input("Ask a question about MDM:")
    if user_question:
        st.session_state.chat_history.append({"role": "user", "text": user_question})

        results = search(user_question)
        context_text = "\n\n".join([r["text"] for r in results])
        images = []
        for r in results:
            images.extend(r["metadata"].get("images", []))
        images = list(dict.fromkeys(images))  # unique

        answer = generate_answer(context_text, user_question)
        st.session_state.chat_history.append({"role": "bot", "text": answer, "images": images})

for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['text']}")
    else:
        st.markdown(f"**Bot:** {message['text']}")
        for img in message.get("images", []):
            image_path = os.path.join(IMAGE_DIR, img)
            if os.path.exists(image_path):
                st.image(image_path, caption=img)

chat()

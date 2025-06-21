import os
import openai
import faiss
import pickle
import numpy as np
import fitz  # PyMuPDF
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

PDF_DIR = "kb_pdfs"
IMAGE_DIR = "static/images"
INDEX_PATH = "vector_store/index.faiss"
META_PATH = "vector_store/meta.pkl"

def get_embedding(text):
    res = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return res.data[0].embedding

def chunk_text(text, max_tokens=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk = " ".join(words[i:i + max_tokens])
        chunks.append(chunk)
    return chunks

def extract_images_from_pdf(pdf_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_info = []  # store tuples: (page_num, image_filename)
    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page{page_num+1}_img{img_index+1}.{image_ext}"
            image_path = os.path.join(output_folder, image_filename)
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            image_info.append((page_num + 1, image_filename))
    print(f"Extracted {len(image_info)} images from {pdf_path}")
    return image_info

def main():
    documents = []
    metadata = []  # will hold {"source": filename, "page": page_num, "images": [img_filenames]}
    all_images = {}

    os.makedirs(IMAGE_DIR, exist_ok=True)

    for filename in tqdm(os.listdir(PDF_DIR)):
        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_DIR, filename)
        text_chunks = []
        images_info = extract_images_from_pdf(pdf_path, IMAGE_DIR)
        # Group images by page number
        images_by_page = {}
        for page_num, img_file in images_info:
            images_by_page.setdefault(page_num, []).append(img_file)

        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page_text = doc[page_num].get_text()
            chunks = chunk_text(page_text)
            for chunk in chunks:
                documents.append(chunk)
                # Attach images on the same page to the chunk metadata
                metadata.append({
                    "source": filename,
                    "page": page_num + 1,
                    "images": images_by_page.get(page_num + 1, [])
                })

    print(f"Embedding {len(documents)} chunks...")
    embeddings = [get_embedding(doc) for doc in tqdm(documents)]

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    os.makedirs("vector_store", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump((documents, metadata), f)

    print("Embedding and index saved!")

if __name__ == "__main__":
    main()

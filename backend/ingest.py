import os
import shutil

import faiss
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

STATIC_IMAGE_DIR = os.path.join(BASE_DIR, "backend", "static", "val2017")

# IMPORTANT: these are the files query.py already uses
INDEX_PATH = os.path.join(BASE_DIR, "img.index")
PATHS_PATH = os.path.join(BASE_DIR, "paths.npy")

MODEL_NAME = "openai/clip-vit-base-patch32"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)


def embed_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        embedding = model.get_image_features(**inputs)

    embedding = embedding.cpu().numpy().astype("float32")
    faiss.normalize_L2(embedding)

    return embedding


def load_paths():
    if os.path.exists(PATHS_PATH):
        return list(np.load(PATHS_PATH, allow_pickle=True))

    return []


def save_paths(paths):
    np.save(PATHS_PATH, np.array(paths, dtype=object))


def ingest_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")

    os.makedirs(STATIC_IMAGE_DIR, exist_ok=True)

    filename = os.path.basename(image_path)
    saved_path = os.path.join(STATIC_IMAGE_DIR, filename)

    if os.path.abspath(image_path) != os.path.abspath(saved_path):
        shutil.copy2(image_path, saved_path)

    paths = load_paths()

    # query.py returns these paths, then app.py uses basename,
    # so storing the saved full path is fine.
    if saved_path in paths:
        print(f"[INGEST] Already indexed: {saved_path}")
        return

    embedding = embed_image(saved_path)

    index = faiss.read_index(INDEX_PATH)
    index.add(embedding)
    faiss.write_index(index, INDEX_PATH)

    paths.append(saved_path)
    save_paths(paths)

    print(f"[INGEST] Added: {saved_path}")
    print(f"[INGEST] Index size: {index.ntotal}")
    print(f"[INGEST] Paths size: {len(paths)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embed one image and add it to the existing FAISS index.")
    parser.add_argument("image_path", help="Path to local image file")

    args = parser.parse_args()
    ingest_image(args.image_path)
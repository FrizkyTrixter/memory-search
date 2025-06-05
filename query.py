import os
import clip
import faiss
import torch
import numpy as np

# This points to the MemorySearch root folder
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

index_path = os.path.join(ROOT_DIR, "img.index")
paths_path = os.path.join(ROOT_DIR, "paths.npy")

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)

# Load FAISS index + image paths
index = faiss.read_index(index_path)
paths = np.load(paths_path, allow_pickle=True)

def search_images(prompt, top_k=5):
    with torch.no_grad():
        text_emb = model.encode_text(clip.tokenize([prompt]).to(device)).cpu().numpy()
    text_emb /= np.linalg.norm(text_emb)
    D, I = index.search(text_emb, k=top_k)
    return [paths[idx] for idx in I[0]]


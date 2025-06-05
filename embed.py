import clip, faiss, torch, os
from PIL import Image
from torchvision import transforms
import numpy as np, pickle, tqdm

device      = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

img_paths   = [f"data/{f}" for f in os.listdir("data") if f.endswith(('.jpg','.png'))]
embeddings  = np.empty((len(img_paths), 512), dtype='float32')

for i, p in enumerate(tqdm.tqdm(img_paths)):
    img = preprocess(Image.open(p)).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)
    embeddings[i] = emb.cpu().numpy() / np.linalg.norm(emb.cpu().numpy())

# build FAISS index
index = faiss.IndexFlatIP(512)          # cosine similarity on unit-norm vectors
index.add(embeddings)

faiss.write_index(index, "img.index")
pickle.dump(img_paths, open("paths.pkl", "wb"))


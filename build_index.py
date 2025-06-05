import clip, faiss, torch, glob, os, tqdm, numpy as np
from PIL import Image
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# grab all *.jpg inside data/ (incl. data/val2017)
img_paths = glob.glob(os.path.join("data", "**", "*.jpg"), recursive=True)
embeddings = np.empty((len(img_paths), 512), dtype="float32")

for i, path in enumerate(tqdm.tqdm(img_paths, desc="Embedding")):
    img = preprocess(Image.open(path)).unsqueeze(0).to(device)
    with torch.no_grad():
        v = model.encode_image(img).cpu().numpy()
    embeddings[i] = v / np.linalg.norm(v)

index = faiss.IndexFlatIP(512)          # cosine on unit vectors
index.add(embeddings)

faiss.write_index(index, "img.index")
np.save("paths.npy", np.asarray(img_paths))
print(f"Indexed {len(img_paths)} images → img.index")


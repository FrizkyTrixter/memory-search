import clip, faiss, torch, numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)

index = faiss.read_index("img.index")
paths = np.load("paths.npy", allow_pickle=True)  # ← load the NumPy array

while True:
    prompt = input("\nDescribe the picture you remember → ")
    if not prompt:
        break

    with torch.no_grad():
        text_emb = model.encode_text(clip.tokenize([prompt]).to(device)).cpu().numpy()
    text_emb /= np.linalg.norm(text_emb)

    D, I = index.search(text_emb, k=5)
    for rank, idx in enumerate(I[0], start=1):
        print(f"{rank}. {paths[idx]}   (similarity {D[0][rank-1]:.3f})")


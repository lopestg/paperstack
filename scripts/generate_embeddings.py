from pathlib import Path
import json
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


INPUT_FILE = Path("data/chunks.jsonl")

EMBEDDINGS_OUTPUT = Path("data/embeddings.npy")
METADATA_OUTPUT = Path("data/metadata.pkl")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks():
    chunks = []

    with open(INPUT_FILE) as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks


def main():
    print("Loading embedding model...")
    
    model = SentenceTransformer(MODEL_NAME)

    print("Loading chunks...")
    
    chunks = load_chunks()

    texts = [chunk["text"] for chunk in chunks]

    print(f"Generating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print(f"Embeddings shape: {embeddings.shape}")

    np.save(EMBEDDINGS_OUTPUT, embeddings)

    with open(METADATA_OUTPUT, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved embeddings to {EMBEDDINGS_OUTPUT}")
    print(f"Saved metadata to {METADATA_OUTPUT}")


if __name__ == "__main__":
    main()

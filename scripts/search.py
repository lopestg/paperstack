from pathlib import Path
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


EMBEDDINGS_FILE = Path("data/embeddings.npy")
METADATA_FILE = Path("data/metadata.pkl")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5


def load_metadata():
    with open(METADATA_FILE, "rb") as f:
        return pickle.load(f)


def main():
    print("Loading embeddings...")

    embeddings = np.load(EMBEDDINGS_FILE).astype("float32")

    print(f"Embeddings shape: {embeddings.shape}")

    print("Building nearest-neighbor index...")

    index = NearestNeighbors(
        n_neighbors=TOP_K,
        metric="cosine",
    )

    index.fit(embeddings)

    print("Loading metadata...")

    metadata = load_metadata()

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    while True:
        query = input("\nEnter search query (or 'quit'): ")

        if query.lower() == "quit":
            break

        query_embedding = model.encode([query]).astype("float32")

        distances, indices = index.kneighbors(query_embedding)

        print("\nTop Results:\n")

        for rank, idx in enumerate(indices[0]):
            result = metadata[idx]

            print(f"Result #{rank + 1}")
            print(f"Document: {result['document']}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Cosine Distance: {distances[0][rank]:.4f}")

            print(result["text"][:500])

            print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()

from pathlib import Path
import pickle
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


EMBEDDINGS_FILE = Path("data/embeddings.npy")
METADATA_FILE = Path("data/metadata.pkl")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5


def tokenize(text: str):
    text = text.lower()
    return re.findall(r"\w+", text)


def load_metadata():
    with open(METADATA_FILE, "rb") as f:
        return pickle.load(f)


def reciprocal_rank_fusion(rankings, k=60):
    scores = {}

    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def main():
    print("Loading embeddings...")

    embeddings = np.load(EMBEDDINGS_FILE).astype("float32")

    print("Loading metadata...")

    metadata = load_metadata()

    corpus = [item["text"] for item in metadata]

    print("Building BM25 index...")

    tokenized_corpus = [tokenize(doc) for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)

    print("Building semantic index...")

    semantic_index = NearestNeighbors(
        n_neighbors=TOP_K,
        metric="cosine",
    )

    semantic_index.fit(embeddings)

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    while True:
        query = input("\nEnter search query (or 'quit'): ")

        if query.lower() == "quit":
            break

        print("\nRunning BM25 retrieval...")

        tokenized_query = tokenize(query)

        bm25_scores = bm25.get_scores(tokenized_query)

        bm25_ranking = np.argsort(bm25_scores)[::-1][:TOP_K]

        print("Running semantic retrieval...")

        query_embedding = model.encode([query]).astype("float32")

        _, semantic_indices = semantic_index.kneighbors(query_embedding)

        semantic_ranking = semantic_indices[0]

        print("Fusing rankings...")

        fused = reciprocal_rank_fusion(
            [bm25_ranking, semantic_ranking]
        )

        print("\nTop Hybrid Results:\n")

        for rank, (idx, score) in enumerate(fused[:TOP_K]):
            result = metadata[idx]

            print(f"Result #{rank + 1}")
            print(f"Fusion Score: {score:.4f}")
            print(f"Document: {result['document']}")
            print(f"Chunk ID: {result['chunk_id']}")

            print(result["text"][:500])

            print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()

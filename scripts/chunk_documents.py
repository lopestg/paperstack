from pathlib import Path
import json
import re

from tqdm import tqdm


INPUT_DIR = Path("data/processed")
OUTPUT_FILE = Path("data/chunks.jsonl")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def clean_text(text: str) -> str:
    """
    Basic text normalization.
    """

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def chunk_text(text: str) -> list[dict]:
    """
    Split text into overlapping chunks.
    """

    chunks = []

    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        chunk = text[start:end]

        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": chunk,
                "start_char": start,
                "end_char": end,
            }
        )

        start += CHUNK_SIZE - CHUNK_OVERLAP
        chunk_id += 1

    return chunks


def main():
    txt_files = list(INPUT_DIR.glob("*.txt"))

    print(f"Found {len(txt_files)} processed documents")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w") as output_f:
        for txt_file in tqdm(txt_files):
            text = txt_file.read_text()

            text = clean_text(text)

            chunks = chunk_text(text)

            for chunk in chunks:
                record = {
                    "document": txt_file.stem,
                    **chunk,
                }

                output_f.write(json.dumps(record) + "\n")

    print(f"Saved chunk dataset to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

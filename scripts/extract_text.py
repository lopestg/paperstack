from pathlib import Path

import fitz
from tqdm import tqdm


RAW_DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)

    text = []

    for page in doc:
        text.append(page.get_text())

    return "\n".join(text)


def main():
    pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files")

    for pdf_file in tqdm(pdf_files):
        try:
            text = extract_text_from_pdf(pdf_file)

            output_file = OUTPUT_DIR / f"{pdf_file.stem}.txt"

            output_file.write_text(text)

        except Exception as e:
            print(f"Failed processing {pdf_file.name}: {e}")


if __name__ == "__main__":
    main()

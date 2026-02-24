# paddle_kangwon.py
"""
Extracts text from PDF files in the kangwon folder using PaddleOCR,
saves results to a new output folder, and connects to chunking in Elastic
with a new index named 'kangwon2'.
"""

import os
from ocr.paddle_ocr import process_pdf  # uses existing PaddleOCR pipeline
from elastic.elastic_rag import ingest_markdown_to_elastic  # pipeline for chunking/indexing

KANGWON_DIR = os.path.join(os.path.dirname(__file__), '../kangwon')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../kangout')
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple


KANGWON_DIR = os.path.join(os.path.dirname(__file__), '..', 'kangwon')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'kangout')
INDEX_NAME = 'kangwon2'

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _worker_process_pdf(path: str) -> Tuple[str, str, str]:
    """Worker function run in a separate process.

    Returns tuple: (file_name, markdown, error)
    """
    try:
        # import inside worker to avoid pickling issues
        from ocr.paddle_ocr import process_pdf

        markdown, _ = process_pdf(path, save_images=False)
        return os.path.basename(path), markdown, ""
    except Exception as e:
        return os.path.basename(path), "", str(e)


def process_batch(pdf_paths, output_dir, index_name):
    """Process a batch of PDF paths in parallel, write markdown files, and return results list."""
    results = []

    # Choose worker count based on CPU but limit to avoid overloading API
    max_workers = min(8, (os.cpu_count() or 2))

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_worker_process_pdf, p): p for p in pdf_paths}

        for fut in as_completed(futures):
            file_name, markdown, error = fut.result()
            base = os.path.splitext(file_name)[0]
            out_path = os.path.join(output_dir, f"{base}_extracted.md")

            if error:
                print(f"⚠️ Error processing {file_name}: {error}")
                results.append((file_name, None, error))
                continue

            # write markdown to file
            try:
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(markdown)
                print(f"📝 Saved: {out_path}")
                results.append((file_name, markdown, None))
            except Exception as e:
                print(f"⚠️ Failed to save markdown for {file_name}: {e}")
                results.append((file_name, None, str(e)))

    return results


def chunk_and_index_from_main(results, index_name):
    """Run ingest (chunk+index) sequentially in the main process for safety."""
    from elastic.elastic_rag import ingest_markdown_to_elastic, get_es_client

    # Create one ES client for the whole batch to avoid re-creating SSL contexts
    # and multiple TCP connections for every file.
    es = None
    try:
        es = get_es_client()
    except Exception as e:
        print(f"⚠️ Failed to create Elasticsearch client: {e}")

    for file_name, markdown, error in results:
        if error or not markdown:
            continue
        try:
            ingest_markdown_to_elastic(markdown, file_name, index_name, es=es)
        except Exception as e:
            print(f"⚠️ Failed to index {file_name}: {e}")


def process_all(batch_size: int = 100):
    # list pdfs
    pdf_files = [os.path.join(KANGWON_DIR, f) for f in sorted(os.listdir(KANGWON_DIR)) if f.lower().endswith('.pdf')]
    total = len(pdf_files)
    if total == 0:
        print("No PDF files found in", KANGWON_DIR)
        return

    print(f"Found {total} PDF files. Processing in batches of {batch_size}...")

    for i in range(0, total, batch_size):
        batch = pdf_files[i:i+batch_size]
        print(f"\n➡️ Processing batch {i // batch_size + 1}: {len(batch)} files (indexes {i}-{i+len(batch)-1})")
        results = process_batch(batch, OUTPUT_DIR, INDEX_NAME)
        print("🔗 Indexing batch results to Elasticsearch (sequential)...")
        chunk_and_index_from_main(results, INDEX_NAME)

    print("\n✅ All batches processed.")


def _parse_args():
    p = argparse.ArgumentParser(description="PaddleOCR batch processor for kangwon PDFs")
    p.add_argument("--batch-size", type=int, default=100, help="Number of PDFs per batch (default 100)")
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    process_all(batch_size=args.batch_size)

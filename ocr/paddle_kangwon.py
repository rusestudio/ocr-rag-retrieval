# paddle_kangwon.py
"""
Extracts text from PDF files in the kangwon folder using PaddleOCR,
saves results to a new output folder, and connects to chunking in Elastic
with a new index named 'kangwon'.
"""
import os
from ocr.paddle_ocr import process_pdf  # uses existing PaddleOCR pipeline
from elastic.elastic_rag import ingest_markdown_to_elastic  # pipeline for chunking/indexing

KANGWON_DIR = os.path.join(os.path.dirname(__file__), '../kangwon')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../output_kangwon')
INDEX_NAME = 'kangwon'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_pdfs():
    pdf_files = [f for f in os.listdir(KANGWON_DIR) if f.lower().endswith('.pdf')]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(KANGWON_DIR, pdf_file)
        base = os.path.splitext(pdf_file)[0]
        output_path = os.path.join(OUTPUT_DIR, f'{base}_extracted.md')
        # Extract markdown via PaddleOCR
        markdown, _ = process_pdf(pdf_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        # Chunk and index in Elastic using the generic ingest pipeline
        ingest_markdown_to_elastic(markdown, pdf_file, INDEX_NAME)

if __name__ == '__main__':
    process_pdfs()

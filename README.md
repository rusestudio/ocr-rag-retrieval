# OCR-RAG-Retrieval

PDF OCR extraction and RAG (Retrieval-Augmented Generation) pipeline using PaddleOCR/MinerU and Elasticsearch.

## Features

- **Dual OCR Support**: PaddleOCR (PP-StructureV3) and MinerU APIs
- **Elasticsearch Integration**: Full-text search with BM25 scoring
- **Separate Indices**: Keep OCR results from different engines isolated
- **Interactive Q&A**: Search indexed documents via command line

## Setup

### 1. Install Dependencies

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:

```env
# PaddleOCR API
ACCESS_TOKEN=your_paddle_token

# MinerU API (optional)
MINERU_API_KEY=your_mineru_key

# Elasticsearch Cloud
ELASTICSEARCH_CLOUD_ID=your_cloud_id
ELASTICSEARCH_API_KEY=your_api_key
```

## Usage

### Process PDFs with PaddleOCR

```bash
# Single file
python pipeline.py --ocr paddle --pdf path/to/file.pdf

# Folder of PDFs
python pipeline.py --ocr paddle --folder path/to/pdfs/
```

### Process PDFs with MinerU

```bash
python pipeline.py --ocr mineru --pdf path/to/file.pdf
```

### Interactive Q&A

```bash
# Search PaddleOCR index
python pipeline.py --qa --index paddle

# Search MinerU index
python pipeline.py --qa --index mineru

# Search both indices
python pipeline.py --qa --index all
```

## Project Structure

```
ocr-rag-retrieval/
├── ocr/
│   ├── paddle_ocr.py    # PaddleOCR API integration
│   └── mineru_ocr.py    # MinerU API integration
├── elastic/
│   └── elastic_rag.py   # Elasticsearch indexing & search
├── pipeline.py          # Main entry point
├── .env                 # API keys (not committed)
└── .gitignore
```

## Elasticsearch Indices

| Index | OCR Engine | Description |
|-------|------------|-------------|
| `pdf_documents_paddle` | PaddleOCR | PP-StructureV3 results |
| `pdf_documents` | MinerU | MinerU VLM results |

## API Requirements

- **PaddleOCR**: PP-StructureV3 endpoint with token auth
- **MinerU**: v4 API with JWT token
- **Elasticsearch**: Cloud or self-hosted instance 

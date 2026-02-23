# OCR-RAG-Retrieval

PDF OCR extraction and RAG (Retrieval-Augmented Generation) utilities that combine OCR (PaddleOCR or MinerU), Elasticsearch indexing, and simple retrieval/QA helpers.

## Quick summary
- Extract text from PDFs using `ocr/paddle_ocr.py` (PaddleOCR / PP-StructureV3) or `ocr/mineru_ocr.py` (MinerU).
- Index OCR results into Elasticsearch (BM25) via `elastic/elastic_rag.py`.
- Run interactive search/QA and local embedding workflows with `pipeline.py`, `search_engine.py`, and `offline_embedder.py`.

## Requirements
- Python 3.8+
- Elasticsearch (cloud or self-hosted)
- API access for PaddleOCR (`ACCESS_TOKEN`) and/or MinerU (`MINERU_API_KEY`) if you use those engines
- `requirements.txt` present in repo — install with `pip install -r requirements.txt`

## Setup

1. Create virtualenv and install dependencies (Windows):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Create a `.env` in the repo root with the following keys (example):
```env
# PaddleOCR
ACCESS_TOKEN=your_paddle_token

# MinerU (optional)
MINERU_API_KEY=your_mineru_key

# Elasticsearch (Cloud or API-key)
ELASTICSEARCH_CLOUD_ID=your_cloud_id_or_leave_blank
ELASTICSEARCH_API_KEY=your_api_key_or_leave_blank
ELASTICSEARCH_HOST=your_elasticsearch_host # optional for non-cloud
```

## Main scripts & usage

- `pipeline.py` — orchestrates OCR, indexing and QA flows. Typical examples:
```powershell
# OCR a single PDF with PaddleOCR and index
python pipeline.py --ocr paddle --pdf "path\to\file.pdf"

# OCR a folder with PaddleOCR
python pipeline.py --ocr paddle --folder "path\to\pdfs\"

# OCR with MinerU (single file)
python pipeline.py --ocr mineru --pdf "path\to\file.pdf"

# Start interactive QA against an index (paddle | mineru | all)
python pipeline.py --qa --index paddle
```

- `search_engine.py` — lightweight search/QA helper that calls the Elasticsearch index directly. Use it to run quick queries or integrate a custom front-end.

- `offline_embedder.py` — utility to build or refresh local embedding checkpoints (file `embed_checkpoint.txt` is included in repo). Use this when you want to embed documents locally for vector-based workflows or caching.

Notes: specific flags for `search_engine.py` and `offline_embedder.py` may vary — run with `--help` for available options:
```powershell
python search_engine.py --help
python offline_embedder.py --help
```

## Outputs & working folders

- `output/` — default OCR/processing outputs and intermediate JSON files produced by scripts (e.g., content lists, extracted markdown, models).
- `output_paddle/` — Paddle-specific outputs and extracted documents.
- `abc-pages/` — example pages / data used during development.
- `semantic/` — notebook(s) such as `bge.ipynb` for embedding experiments and semantic tests.

## Project layout (current)
```
ocr-rag-retrieval/
├── abc-pages/                  # example/dev pages
├── elastic/
│   └── elastic_rag.py          # Elasticsearch indexing & search helpers
├── ocr/
│   ├── mineru_ocr.py           # MinerU OCR integration
│   └── paddle_ocr.py           # PaddleOCR integration
├── output/                     # processed outputs (content lists, md, json)
├── output_paddle/              # Paddle-specific outputs
├── offline_embedder.py         # local embedding/checkpoint utility
├── pipeline.py                 # main CLI orchestrator (OCR, index, QA)
├── requirements.txt
├── search_engine.py            # search/QA helper script
├── embed_checkpoint.txt        # example/local embedding checkpoint
├── README.md
└── semantic/
	└── bge.ipynb               # embedding experiments
```

## Elasticsearch indices used
- `pdf_documents_paddle` — PaddleOCR output (PP-StructureV3)
- `pdf_documents` — MinerU output

Index mappings and ingestion are handled in `elastic/elastic_rag.py`; adjust index names there if you need different naming.

## Notes & troubleshooting
- Ensure Elasticsearch credentials in `.env` match your deployment (cloud vs host+api).
- If OCR API calls fail, confirm tokens are valid and network access is allowed.
- If you see rate limits from OCR providers, add retries or batch your requests.
- For large PDF collections, index in batches to avoid memory spikes.

## Want improvements?
If you'd like I can:
- add example commands for `search_engine.py` and `offline_embedder.py` after inspecting their help flags
- add a small example dataset and a CI workflow to run a quick end-to-end test

Tell me which addition you'd like and I'll implement it.

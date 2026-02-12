import os
import sys
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# Elasticsearch config
ES_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
ES_CLOUD_ID = os.getenv("ELASTICSEARCH_CLOUD_ID", None)
ES_API_KEY = os.getenv("ELASTICSEARCH_API_KEY", None)

# Index names - separate for each OCR model
ES_INDEX_MINERU = os.getenv("ELASTICSEARCH_INDEX", "pdf_documents")
ES_INDEX_PADDLE = os.getenv("ELASTICSEARCH_INDEX_PADDLE", "pdf_documents_paddle")

# Default index (for backward compatibility)
ES_INDEX = ES_INDEX_MINERU

# Initialize Elasticsearch client
def get_es_client():
    if ES_CLOUD_ID:
        # Elastic Cloud with Cloud ID
        return Elasticsearch(cloud_id=ES_CLOUD_ID, api_key=ES_API_KEY)
    elif ES_API_KEY:
        # Self-hosted or Elastic Cloud with URL + API Key
        return Elasticsearch(ES_HOST, api_key=ES_API_KEY)
    else:
        # Local without auth
        return Elasticsearch(ES_HOST)


# -----------------------------
# Step 1: Create index with mappings
# -----------------------------
def create_index(es, index_name=ES_INDEX):
    """Create index with proper mappings for RAG"""
    
    mappings = {
        "mappings": {
            "properties": {
                "content": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "chunk_id": {
                    "type": "keyword"
                },
                "source_file": {
                    "type": "keyword"
                },
                "page_number": {
                    "type": "integer"
                },
                "created_at": {
                    "type": "date"
                }
            }
        }
    }
    
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=mappings)
        print(f"✅ Index '{index_name}' created")
    else:
        print(f"ℹ️ Index '{index_name}' already exists")


# -----------------------------
# Step 2: Chunk markdown content
# -----------------------------
def chunk_markdown(markdown_content, chunk_size=1000, overlap=200):
    """Split markdown into overlapping chunks for better retrieval"""
    chunks = []
    
    # Split by double newlines (paragraphs) first
    paragraphs = markdown_content.split("\n\n")
    
    current_chunk = ""
    chunk_id = 0
    
    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk.strip():
                chunks.append({
                    "chunk_id": chunk_id,
                    "content": current_chunk.strip()
                })
                chunk_id += 1
            
            # Start new chunk with overlap
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + para + "\n\n"
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append({
            "chunk_id": chunk_id,
            "content": current_chunk.strip()
        })
    
    return chunks


# -----------------------------
# Step 3: Index chunks to Elasticsearch
# -----------------------------
def index_chunks(es, chunks, source_file, index_name=ES_INDEX):
    """Index all chunks to Elasticsearch using bulk API"""
    from datetime import datetime, timezone
    from elasticsearch.helpers import bulk
    
    def generate_actions():
        for chunk in chunks:
            yield {
                "_index": index_name,
                "_source": {
                    "content": chunk["content"],
                    "chunk_id": f"{source_file}_{chunk['chunk_id']}",
                    "source_file": source_file,
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
            }
    
    success, failed = bulk(es, generate_actions(), raise_on_error=False)
    es.indices.refresh(index=index_name)
    print(f"✅ Indexed {success} chunks from '{source_file}' ({failed} failed)")


# -----------------------------
# Step 4: Search / RAG query
# -----------------------------
def search_documents(es, query, top_k=5, index_name=ES_INDEX):
    """Search for relevant chunks using the query"""
    
    search_body = {
        "query": {
            "match": {
                "content": {
                    "query": query,
                    "fuzziness": "AUTO"
                }
            }
        },
        "size": top_k,
        "_source": ["content", "chunk_id", "source_file"]
    }
    
    response = es.search(index=index_name, body=search_body)
    
    results = []
    for hit in response["hits"]["hits"]:
        results.append({
            "score": hit["_score"],
            "content": hit["_source"]["content"],
            "source": hit["_source"]["source_file"],
            "chunk_id": hit["_source"]["chunk_id"]
        })
    
    return results


# -----------------------------
# Main pipeline function
# -----------------------------
def ingest_markdown_to_elastic(markdown_content, source_file, index_name=ES_INDEX):
    """
    Full pipeline: chunk markdown and index to Elasticsearch
    
    Args:
        markdown_content: The markdown text to index
        source_file: Name of the source file
        index_name: Which index to use (ES_INDEX_MINERU or ES_INDEX_PADDLE)
    """
    es = get_es_client()
    
    # Create index if needed
    create_index(es, index_name)
    
    # Chunk the content
    chunks = chunk_markdown(markdown_content)
    print(f"📄 Created {len(chunks)} chunks")
    
    # Index chunks
    index_chunks(es, chunks, source_file, index_name)
    
    return len(chunks)


def ask_question(question, top_k=3, index_name=ES_INDEX):
    """
    Simple RAG query - returns relevant chunks
    
    Args:
        question: The question to search for
        top_k: Number of results to return
        index_name: Which index to search (ES_INDEX_MINERU or ES_INDEX_PADDLE)
    """
    es = get_es_client()
    
    results = search_documents(es, question, top_k, index_name)
    
    print(f"\n🔍 Question: {question}")
    print(f"📊 Found {len(results)} relevant chunks (index: {index_name}):\n")
    
    for i, r in enumerate(results, 1):
        print(f"--- Result {i} (score: {r['score']:.2f}) ---")
        print(f"Source: {r['source']}")
        # Show full content (up to 1000 chars, truncate only if very long)
        content = r['content']
        if len(content) > 1000:
            print(f"Content: {content[:1000]}... [truncated, {len(content)} chars total]")
        else:
            print(f"Content: {content}")
        print()
    
    return results


# -----------------------------
# Convenience functions for different OCR sources
# -----------------------------
def ingest_mineru(markdown_content, source_file):
    """Ingest MinerU OCR results to mineru index"""
    return ingest_markdown_to_elastic(markdown_content, source_file, ES_INDEX_MINERU)


def ingest_paddle(markdown_content, source_file):
    """Ingest PaddleOCR results to paddle index"""
    return ingest_markdown_to_elastic(markdown_content, source_file, ES_INDEX_PADDLE)


def ask_mineru(question, top_k=3):
    """Search in MinerU OCR index"""
    return ask_question(question, top_k, ES_INDEX_MINERU)


def ask_paddle(question, top_k=3):
    """Search in PaddleOCR index"""
    return ask_question(question, top_k, ES_INDEX_PADDLE)


def ask_all(question, top_k=3):
    """Search in both indices and combine results"""
    print("=" * 50)
    print("🔍 MINERU INDEX")
    print("=" * 50)
    mineru_results = ask_question(question, top_k, ES_INDEX_MINERU)
    
    print("=" * 50)
    print("🔍 PADDLE INDEX")
    print("=" * 50)
    paddle_results = ask_question(question, top_k, ES_INDEX_PADDLE)
    
    return {"mineru": mineru_results, "paddle": paddle_results}


# -----------------------------
# MAIN - Test the flow
# -----------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Elasticsearch RAG")
    parser.add_argument("--index", choices=["mineru", "paddle", "all"], default="mineru",
                        help="Which index to use")
    parser.add_argument("--ingest", type=str, help="Path to markdown file to ingest")
    parser.add_argument("--source", type=str, default="document.pdf", help="Source file name")
    parser.add_argument("--query", type=str, help="Question to search")
    
    args = parser.parse_args()
    
    # Determine index
    if args.index == "mineru":
        index = ES_INDEX_MINERU
    elif args.index == "paddle":
        index = ES_INDEX_PADDLE
    else:
        index = None  # Will use ask_all
    
    if args.ingest:
        with open(args.ingest, "r", encoding="utf-8") as f:
            markdown = f.read()
        print(f"📥 Ingesting to {args.index} index...")
        ingest_markdown_to_elastic(markdown, args.source, index or ES_INDEX_MINERU)
    
    if args.query:
        if args.index == "all":
            ask_all(args.query)
        else:
            ask_question(args.query, index_name=index)

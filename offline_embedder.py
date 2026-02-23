from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from elasticsearch.helpers import scan
import os

# Load env
load_dotenv()

# Connect Elastic
es = Elasticsearch(
    cloud_id=os.getenv("ELASTICSEARCH_CLOUD_ID"),
    api_key=os.getenv("ELASTICSEARCH_API_KEY")
)

# Load model
print("Loading BGE model...")
model = SentenceTransformer("BAAI/bge-m3", device="cpu")
print("Model loaded.")

SOURCE_INDEX = "pdf_documents_paddle"   # or pdf_documents
TARGET_INDEX = "pdf_semantic"

CHECKPOINT_FILE = "embed_checkpoint.txt"

# -----------------------
# Checkpoint helpers
# -----------------------
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return set(f.read().splitlines())
    return set()

def save_checkpoint(chunk_id):
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(chunk_id + "\n")

# -----------------------
# Scroll-safe iterator
# -----------------------
def get_all_docs(index):
    for doc in scan(
        client=es,
        index=index,
        query={"query": {"match_all": {}}},
        size=50,          # small batch for CPU
        scroll="10m",     # long keepalive
        preserve_order=False,
        clear_scroll=True
    ):
        yield doc

# -----------------------
# Embedding job
# -----------------------
print("Starting embedding job...")

done = load_checkpoint()
count = 0
skipped = 0

for doc in get_all_docs(SOURCE_INDEX):
    src = doc["_source"]
    chunk_id = src["chunk_id"]

    # Skip already embedded
    if chunk_id in done:
        skipped += 1
        continue

    text = src["content"]

    # Embed (CPU safe)
    vec = model.encode(
        text,
        normalize_embeddings=True,
        batch_size=1,
        show_progress_bar=False
    )

    new_doc = {
        "content": text,
        "chunk_id": chunk_id,
        "source_file": src.get("source_file"),
        "embedding": vec.tolist()
    }

    es.index(index=TARGET_INDEX, document=new_doc)

    save_checkpoint(chunk_id)

    count += 1
    if count % 50 == 0:
        print(f"Embedded {count} chunks (skipped {skipped})")

print("Embedding job finished.")
print("Total new embeddings:", count)
print("Total skipped (already done):", skipped)
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# =========================
# Setup
# =========================

load_dotenv()

es = Elasticsearch(
    cloud_id=os.getenv("ELASTICSEARCH_CLOUD_ID"),
    api_key=os.getenv("ELASTICSEARCH_API_KEY")
)

print("Loading embedding model...")
model = SentenceTransformer("BAAI/bge-m3", device="cpu")
print("Model loaded.")

INDEX_NAME = "pdf_semantic"


# =========================
# Semantic Search (Vector)
# =========================
def semantic_search(query, k=5):
    qvec = model.encode(query, normalize_embeddings=True).tolist()

    body = {
        "knn": {
            "field": "embedding",
            "query_vector": qvec,
            "k": k,
            "num_candidates": 100
        },
        "_source": ["content", "chunk_id", "source_file"]
    }

    res = es.search(index=INDEX_NAME, body=body)
    return res["hits"]["hits"]


# =========================
# Keyword Search (BM25)
# =========================
def keyword_search(query, k=5):
    body = {
        "size": k,
        "query": {
            "match": {
                "content": {
                    "query": query,
                    "operator": "and"
                }
            }
        },
        "_source": ["content", "chunk_id", "source_file"]
    }

    res = es.search(index=INDEX_NAME, body=body)
    return res["hits"]["hits"]


# =========================
# Hybrid Search
# =========================
def hybrid_search(query, k=5, alpha=0.6):
    sem_results = semantic_search(query, k=20)
    key_results = keyword_search(query, k=20)

    scores = {}

    # semantic scores
    for r in sem_results:
        doc_id = r["_id"]
        scores[doc_id] = scores.get(doc_id, 0) + alpha * r["_score"]

    # keyword scores
    for r in key_results:
        doc_id = r["_id"]
        scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) * r["_score"]

    # sort by combined score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # rebuild docs
    results = []
    seen = set()
    for doc_id, _ in ranked:
        for r in sem_results + key_results:
            if r["_id"] == doc_id and doc_id not in seen:
                results.append(r)
                seen.add(doc_id)
                break

    return results[:k]


# =========================
# Unified Interface
# =========================
def search(query, mode="hybrid", k=5):
    if mode == "semantic":
        return semantic_search(query, k)
    elif mode == "keyword":
        return keyword_search(query, k)
    elif mode == "hybrid":
        return hybrid_search(query, k)
    else:
        raise ValueError("mode must be: semantic | keyword | hybrid")


# =========================
# CLI Loop
# =========================
if __name__ == "__main__":
    print("\nSearch system ready.")
    print("Modes: semantic | keyword | hybrid")

    while True:
        q = input("\nQuery (type 'exit' to quit): ")
        if q.lower() == "exit":
            break

        mode = input("Mode (semantic/keyword/hybrid): ").strip().lower()

        results = search(q, mode=mode, k=5)

        print("\n================ RESULTS ================\n")
        for i, r in enumerate(results, 1):
            print(f"[{i}] Score:", r["_score"])
            print("Chunk:", r["_source"]["chunk_id"])
            print("Source:", r["_source"]["source_file"])
            print("Text:", r["_source"]["content"][:300])
            print("----------------------------------------")
"""
Complete OCR-RAG Pipeline:
Supports both MinerU and PaddleOCR with separate Elasticsearch indices
1. Upload PDF/folder to OCR
2. OCR extraction to markdown
3. Index to Elasticsearch (separate indices per OCR model)
4. Ask questions
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from elastic.elastic_rag import (
    ingest_mineru, ingest_paddle, 
    ask_mineru, ask_paddle, ask_all,
    ES_INDEX_MINERU, ES_INDEX_PADDLE
)


def run_mineru_pipeline(pdf_path):
    """
    Complete flow with MinerU:
    PDF → MinerU OCR → Markdown → Elasticsearch (mineru index)
    """
    from ocr.mineru_ocr import process_pdf
    
    file_name = os.path.basename(pdf_path)
    
    print("=" * 50)
    print("🚀 MINERU OCR-RAG PIPELINE")
    print("=" * 50)
    
    # Step 1: OCR with MinerU
    print("\n📄 PHASE 1: MinerU OCR EXTRACTION")
    print("-" * 30)
    markdown_content, json_data = process_pdf(pdf_path)
    
    if not markdown_content:
        print("❌ No markdown content extracted!")
        return False
    
    print(f"✅ Extracted {len(markdown_content)} characters of text")
    
    # Step 2: Index to Elasticsearch (mineru index)
    print("\n🔍 PHASE 2: ELASTICSEARCH INDEXING")
    print(f"Index: {ES_INDEX_MINERU}")
    print("-" * 30)
    num_chunks = ingest_mineru(markdown_content, file_name)
    
    print(f"✅ Indexed {num_chunks} chunks to Elasticsearch")
    print("\n" + "=" * 50)
    print("✅ MINERU PIPELINE COMPLETE!")
    print("=" * 50)
    
    return True


def run_paddle_pipeline(path, is_folder=False):
    """
    Complete flow with PaddleOCR:
    PDF/Folder → PaddleOCR → Markdown → Elasticsearch (paddle index)
    
    Args:
        path: Path to PDF file or folder with PDF pages
        is_folder: True if path is a folder with split PDF pages
    """
    from ocr.paddle_ocr import process_pdf, process_folder
    
    print("=" * 50)
    print("🚀 PADDLEOCR RAG PIPELINE")
    print("=" * 50)
    
    # Step 1: OCR with PaddleOCR
    print("\n📄 PHASE 1: PaddleOCR EXTRACTION")
    print("-" * 30)
    
    if is_folder:
        file_name = os.path.basename(path.rstrip("/\\")) + "_combined"
        markdown_content = process_folder(path)
    else:
        file_name = os.path.basename(path)
        markdown_content, _ = process_pdf(path)
    
    if not markdown_content:
        print("❌ No markdown content extracted!")
        return False
    
    print(f"✅ Extracted {len(markdown_content)} characters of text")
    
    # Save the markdown
    output_dir = os.path.join(os.path.dirname(__file__), "output_paddle")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "paddle_extracted.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"📝 Saved to: {output_path}")
    
    # Step 2: Index to Elasticsearch (paddle index)
    print("\n🔍 PHASE 2: ELASTICSEARCH INDEXING")
    print(f"Index: {ES_INDEX_PADDLE}")
    print("-" * 30)
    num_chunks = ingest_paddle(markdown_content, file_name)
    
    print(f"✅ Indexed {num_chunks} chunks to Elasticsearch")
    print("\n" + "=" * 50)
    print("✅ PADDLEOCR PIPELINE COMPLETE!")
    print("=" * 50)
    
    return True


def interactive_qa(index="all"):
    """Interactive Q&A mode"""
    print(f"\n🤖 Interactive Q&A Mode - Index: {index}")
    print("Type 'quit' to exit, 'switch' to change index")
    print("-" * 40)
    
    current_index = index
    
    while True:
        question = input(f"\n❓ [{current_index}] Your question: ").strip()
        
        if question.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break
        
        if question.lower() == "switch":
            if current_index == "mineru":
                current_index = "paddle"
            elif current_index == "paddle":
                current_index = "all"
            else:
                current_index = "mineru"
            print(f"🔄 Switched to index: {current_index}")
            continue
        
        if not question:
            continue
        
        if current_index == "mineru":
            ask_mineru(question)
        elif current_index == "paddle":
            ask_paddle(question)
        else:
            ask_all(question)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OCR-RAG Pipeline")
    parser.add_argument("--ocr", choices=["mineru", "paddle"], default="paddle",
                        help="Which OCR model to use (default: paddle)")
    parser.add_argument("--pdf", type=str, help="Path to PDF file to process")
    parser.add_argument("--folder", type=str, help="Path to folder with split PDF pages (paddle only)")
    parser.add_argument("--qa", action="store_true", help="Start interactive Q&A mode")
    parser.add_argument("--index", choices=["mineru", "paddle", "all"], default="all",
                        help="Which index to search in Q&A mode")
    
    args = parser.parse_args()
    
    if args.pdf or args.folder:
        path = args.pdf or args.folder
        is_folder = args.folder is not None
        
        if not os.path.exists(path):
            print(f"❌ Path not found: {path}")
        elif args.ocr == "mineru":
            if is_folder:
                print("❌ MinerU doesn't support folder input. Use --pdf instead.")
            else:
                run_mineru_pipeline(path)
        else:  # paddle
            run_paddle_pipeline(path, is_folder)
    
    if args.qa:
        interactive_qa(args.index)
    
    if not args.pdf and not args.folder and not args.qa:
        print("Usage:")
        print("  # PaddleOCR (default)")
        print("  python pipeline.py --pdf your_file.pdf")
        print("  python pipeline.py --folder abc-pages/")
        print("")
        print("  # MinerU OCR")
        print("  python pipeline.py --ocr mineru --pdf your_file.pdf")
        print("")
        print("  # Q&A mode")
        print("  python pipeline.py --qa --index paddle")
        print("  python pipeline.py --qa --index all")

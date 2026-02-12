"""
PaddleOCR Integration using PaddlePaddle AI Studio API
Extracts PDF to Markdown using PP-StructureV3
"""

import requests
import os
import base64
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# PaddlePaddle AI Studio API
PADDLE_ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")  # From AI Studio
# PP-StructureV3 layout-parsing endpoint (from your AI Studio app)
URL_API_PADDLE = os.getenv("API_URL_PADDLE")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_paddle")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def encode_file_to_base64(file_path):
    """Encode file to base64 string"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_file_url(file_path):
    """For URL-based upload, encode to data URI or return existing URL"""
    # For local files, we use base64 encoding instead
    return None


def parse_pdf_sync(file_path, use_chart_recognition=False, use_doc_unwarping=False):
    """
    Sync parse a PDF file using PaddleOCR API
    
    Args:
        file_path: Path to PDF file
        use_chart_recognition: Enable chart recognition (slower but better for charts)
        use_doc_unwarping: Enable document unwarping (for photos of documents)
    
    Returns:
        API result with layout parsing data
    """
    headers = {
        "Authorization": f"token {PADDLE_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    
    # Encode file to base64
    with open(file_path, 'rb') as f:
        file_bytes = f.read()
    file_base64 = base64.b64encode(file_bytes).decode("ascii")
    file_name = os.path.basename(file_path)
    
    # Determine file type: 0=PDF, 1=Image
    if file_path.lower().endswith('.pdf'):
        file_type = 0
    else:
        file_type = 1
    
    print(f"📤 Sending {file_name} to PaddleOCR API...")
    
    # Correct payload: use "file" + "fileType" (not "images")
    payload = {
        "file": file_base64,
        "fileType": file_type,
        "useDocOrientationClassify": False,
        "useDocUnwarping": use_doc_unwarping,
        "useChartRecognition": use_chart_recognition,
    }
    
    response = requests.post(URL_API_PADDLE, json=payload, headers=headers, timeout=600)
    
    if response.status_code == 429:
        raise Exception("Rate limit exceeded - daily parsing limit reached")
    
    if response.status_code != 200:
        print(f"Error response: {response.text}")
        raise Exception(f"API error: {response.status_code}")
    
    result = response.json()
    
    if "result" not in result:
        raise Exception(f"Unexpected response format: {result}")
    
    return result["result"]


def extract_markdown_from_result(result, save_images=True):
    """
    Extract markdown text and images from PaddleOCR result
    
    Returns:
        Combined markdown text
    """
    markdown_parts = []
    
    layout_results = result.get("layoutParsingResults", [])
    
    for i, res in enumerate(layout_results):
        md_text = res.get("markdown", {}).get("text", "")
        markdown_parts.append(md_text)
        
        # Optionally save images
        if save_images:
            images = res.get("markdown", {}).get("images", {})
            for img_path, img_url in images.items():
                try:
                    full_img_path = os.path.join(OUTPUT_DIR, img_path)
                    os.makedirs(os.path.dirname(full_img_path), exist_ok=True)
                    img_bytes = requests.get(img_url, timeout=30).content
                    with open(full_img_path, "wb") as img_file:
                        img_file.write(img_bytes)
                except Exception as e:
                    print(f"⚠️ Failed to save image {img_path}: {e}")
    
    return "\n\n---\n\n".join(markdown_parts)


def process_pdf(file_path, save_images=True):
    """
    Complete PaddleOCR pipeline: File → API → Markdown
    
    Args:
        file_path: Path to PDF file
        save_images: Whether to save extracted images
    
    Returns:
        (markdown_content, raw_result)
    """
    file_name = os.path.basename(file_path)
    
    print(f"🚀 Processing {file_name} with PaddleOCR...")
    
    # Parse PDF
    result = parse_pdf_sync(file_path)
    
    # Extract markdown
    markdown_content = extract_markdown_from_result(result, save_images)
    
    print(f"✅ Extracted {len(markdown_content)} characters")
    
    return markdown_content, result


def process_folder(folder_path, file_pattern="*.pdf"):
    """
    Process all PDF files in a folder (for split pages like abc-pages)
    
    Args:
        folder_path: Path to folder containing PDF files
        file_pattern: Glob pattern for files to process
    
    Returns:
        Combined markdown content from all files
    """
    import glob
    
    pdf_files = sorted(glob.glob(os.path.join(folder_path, file_pattern)))
    
    if not pdf_files:
        raise Exception(f"No files matching {file_pattern} found in {folder_path}")
    
    print(f"📂 Found {len(pdf_files)} files to process")
    
    all_markdown = []
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing {os.path.basename(pdf_file)}...")
        try:
            markdown, _ = process_pdf(pdf_file, save_images=False)
            all_markdown.append(markdown)
        except Exception as e:
            print(f"⚠️ Error processing {pdf_file}: {e}")
            continue
    
    combined = "\n\n---\n\n".join(all_markdown)
    print(f"\n✅ Combined {len(all_markdown)} files, total {len(combined)} characters")
    
    return combined


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        
        if os.path.isdir(path):
            # Process folder (abc-pages)
            markdown = process_folder(path)
        else:
            # Process single file
            markdown, _ = process_pdf(path)
        
        # Save output
        output_path = os.path.join(OUTPUT_DIR, "paddle_extracted.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        
        print(f"\n📝 Saved to: {output_path}")
    else:
        print("Usage:")
        print("  python paddle_ocr.py <pdf_file>")
        print("  python paddle_ocr.py <folder_with_pdf_pages>")

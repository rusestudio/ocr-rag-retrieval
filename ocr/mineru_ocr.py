import requests
import os
import time
import zipfile
import io
import json
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("MINERU_API_KEY")  # matches your .env

BASE_URL = "https://mineru.net/api/v4"

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Step 1: Request upload URL
# -----------------------------
def request_upload_url(file_name, data_id="doc_001", model_version="vlm"):
    """
    Request upload URL from MinerU API
    
    model_version options:
    - "vlm" (default): Vision-Language Model - better for complex layouts, 
                       but can hallucinate on dense tables
    - "mfd": Model-based Formula Detection - simpler, more stable
    - "auto": Let MinerU decide
    """
    url = f"{BASE_URL}/file-urls/batch"

    payload = {
        "files": [
            {
                "name": file_name,
                "data_id": data_id
            }
        ],
        "model_version": model_version,
        "enable_formula": True,    # Better formula recognition
        "enable_table": True,      # Better table extraction
    }

    res = requests.post(url, headers=HEADERS, json=payload)
    res.raise_for_status()
    data = res.json()

    if data["code"] != 0:
        raise Exception(f"Failed to get upload URL: {data}")

    batch_id = data["data"]["batch_id"]
    upload_url = data["data"]["file_urls"][0]

    return batch_id, upload_url


# -----------------------------
# Step 2: Upload file
# -----------------------------
def upload_file(upload_url, file_path):
    with open(file_path, "rb") as f:
        res = requests.put(upload_url, data=f)

    if res.status_code != 200:
        raise Exception(f"Upload failed: {res.status_code}")

    print("✅ File upload success")


# -----------------------------
# Step 3: Poll batch task
# -----------------------------
def poll_batch(batch_id, interval=10):
    url = f"{BASE_URL}/extract-results/batch/{batch_id}"

    while True:
        res = requests.get(url, headers=HEADERS)
        res.raise_for_status()
        data = res.json()

        if data["code"] != 0:
            raise Exception(f"Polling error: {data}")

        results = data["data"]["extract_result"]

        state = results[0]["state"]

        print(f"⏳ State: {state}")

        if state == "done":
            return results[0]["full_zip_url"]

        if state == "failed":
            raise Exception(f"OCR failed: {results[0].get('err_msg')}")

        time.sleep(interval)


# -----------------------------
# Step 4: Download and extract ZIP
# -----------------------------
def download_and_extract(zip_url, extract_to=OUTPUT_DIR):
    """Download ZIP and extract markdown/json files"""
    print(f"📥 Downloading result from: {zip_url}")
    
    res = requests.get(zip_url, timeout=300)
    res.raise_for_status()
    
    # Extract ZIP in memory
    with zipfile.ZipFile(io.BytesIO(res.content)) as z:
        z.extractall(extract_to)
        extracted_files = z.namelist()
    
    print(f"✅ Extracted {len(extracted_files)} files to: {extract_to}")
    return extracted_files


# -----------------------------
# Step 5: Clean OCR output (post-processing)
# -----------------------------
def clean_markdown(text):
    """
    Clean up common VLM OCR artifacts:
    - Remove repetitive hallucinated text
    - Clean up broken table HTML
    - Remove gibberish patterns
    """
    import re
    
    # Remove lines with excessive repetition (same word 5+ times)
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Detect repetitive patterns (word repeated 5+ times)
        words = line.split()
        if len(words) > 5:
            word_counts = {}
            for w in words:
                word_counts[w] = word_counts.get(w, 0) + 1
            max_repeat = max(word_counts.values()) if word_counts else 0
            # Skip lines where any word repeats more than 10 times
            if max_repeat > 10:
                continue
        
        # Skip known garbage patterns
        garbage_patterns = [
            r'흫사무수단lage',
            r'희사무수단lage', 
            r'사원법law',
            r'(majority의\s*){5,}',
            r'(minority의\s*){5,}',
        ]
        skip = False
        for pattern in garbage_patterns:
            if re.search(pattern, line):
                skip = True
                break
        if skip:
            continue
        
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    # Convert HTML tables to simple format (basic cleanup)
    # Remove problematic colspan/rowspan tables
    text = re.sub(r'<table>.*?</table>', '[TABLE REMOVED - See original PDF]', text, flags=re.DOTALL)
    
    # Clean up stray HTML tags
    text = re.sub(r'</?t[dr].*?>', ' ', text)
    text = re.sub(r'</?table.*?>', '', text)
    text = re.sub(r'</?tr.*?>', '', text)
    
    # Remove empty lines clusters (more than 3 empty lines)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    
    return text


# -----------------------------
# Step 6: Get markdown content
# -----------------------------
def get_markdown_content(output_dir=OUTPUT_DIR, clean=True):
    """Find and return markdown content from extracted files"""
    markdown_content = ""
    json_data = None
    
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            filepath = os.path.join(root, f)
            if f.endswith(".md"):
                with open(filepath, "r", encoding="utf-8") as file:
                    markdown_content += file.read() + "\n\n"
                print(f"📄 Found markdown: {f}")
            elif f.endswith(".json"):
                with open(filepath, "r", encoding="utf-8") as file:
                    json_data = json.load(file)
                print(f"📋 Found JSON: {f}")
    
    if clean:
        print("🧹 Cleaning OCR artifacts...")
        markdown_content = clean_markdown(markdown_content)
    
    return markdown_content, json_data


# -----------------------------
# Main function (callable)
# -----------------------------
def process_pdf(pdf_path, data_id=None, model_version="vlm", clean_output=True):
    """
    Complete pipeline: Upload PDF → OCR → Download result
    Returns: (markdown_content, json_data)
    
    Args:
        pdf_path: Path to PDF file
        data_id: Optional ID for the document
        model_version: "vlm" (complex layouts) or "mfd" (simpler, stable) or "auto"
        clean_output: Whether to clean OCR artifacts (recommended)
    """
    file_name = os.path.basename(pdf_path)
    if data_id is None:
        data_id = file_name.replace(".pdf", "")
    
    print(f"🚀 Step 1: Requesting upload URL (model: {model_version})...")
    batch_id, upload_url = request_upload_url(file_name, data_id, model_version)
    
    print("📤 Step 2: Uploading file...")
    upload_file(upload_url, pdf_path)
    
    print("🧠 Step 3: Waiting for OCR to complete...")
    zip_url = poll_batch(batch_id)
    
    print("📥 Step 4: Downloading results...")
    download_and_extract(zip_url)
    
    print("📄 Step 5: Extracting content...")
    markdown_content, json_data = get_markdown_content(clean=clean_output)
    
    print("\n✅ OCR PIPELINE COMPLETE")
    return markdown_content, json_data


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    PDF_PATH = r"C:\Users\user\ocr-rag-retrieval\abc.pdf"
    
    # Try with "mfd" model if "vlm" produces too much garbage
    # model_version options: "vlm", "mfd", "auto"
    markdown, json_result = process_pdf(PDF_PATH, model_version="vlm", clean_output=True)
    
    # Save markdown to file for easy viewing
    output_md_path = os.path.join(OUTPUT_DIR, "extracted_content.md")
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    
    print(f"\n📝 Markdown saved to: {output_md_path}")
    print(f"📊 Content length: {len(markdown)} characters")

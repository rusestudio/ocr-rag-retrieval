"""
PaddleOCR Integration using PaddlePaddle AI Studio API
Extracts PDF to Markdown using PP-StructureV3
"""

import requests
import os
import re
import base64
from datetime import datetime, timezone
from dotenv import load_dotenv
from html.parser import HTMLParser

load_dotenv()

# PaddlePaddle AI Studio API
PADDLE_ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")  # From AI Studio
# PP-StructureV3 layout-parsing endpoint (from your AI Studio app)
URL_API_PADDLE = os.getenv("API_URL_PADDLE")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_paddle")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# HTML Table → Markdown Table Converter
# -----------------------------
class TableHTMLParser(HTMLParser):
    """Parse HTML table and extract rows/cells"""
    def __init__(self):
        super().__init__()
        self.rows = []
        self.current_row = []
        self.current_cell = ""
        self.in_cell = False
        self.is_header = False
        
    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.current_row = []
        elif tag in ('td', 'th'):
            self.in_cell = True
            self.current_cell = ""
            if tag == 'th':
                self.is_header = True
        elif tag == 'br' and self.in_cell:
            self.current_cell += " "
            
    def handle_endtag(self, tag):
        if tag == 'tr':
            if self.current_row:
                self.rows.append(self.current_row)
        elif tag in ('td', 'th'):
            self.in_cell = False
            self.current_row.append(self.current_cell.strip())
            
    def handle_data(self, data):
        if self.in_cell:
            self.current_cell += data


def html_table_to_markdown(html_table):
    """
    Convert HTML table to Markdown table format
    
    Args:
        html_table: HTML string containing <table>...</table>
    
    Returns:
        Markdown formatted table with | separators
    """
    parser = TableHTMLParser()
    try:
        parser.feed(html_table)
    except Exception:
        # If parsing fails, just strip HTML tags
        return re.sub(r'<[^>]+>', ' ', html_table).strip()
    
    if not parser.rows:
        return ""
    
    # Build markdown table
    md_lines = []
    
    for i, row in enumerate(parser.rows):
        # Escape pipe characters in cell content
        cells = [cell.replace('|', '\\|') for cell in row]
        line = "| " + " | ".join(cells) + " |"
        md_lines.append(line)
        
        # Add separator after first row (header)
        if i == 0:
            separator = "|" + "|".join(["---" for _ in row]) + "|"
            md_lines.append(separator)
    
    return "\n".join(md_lines)


def clean_html_to_markdown(text):
    """
    Clean HTML elements from extracted markdown:
    - Convert <table> → Markdown tables
    - Remove <div style=...> → plain text
    - Clean other HTML artifacts
    
    Args:
        text: Raw markdown with HTML elements
    
    Returns:
        Cleaned markdown text
    """
    result = text
    
    # 1. Convert HTML tables to Markdown tables
    table_pattern = re.compile(r'<table[^>]*>.*?</table>', re.DOTALL | re.IGNORECASE)
    
    def replace_table(match):
        html_table = match.group(0)
        md_table = html_table_to_markdown(html_table)
        return "\n\n" + md_table + "\n\n" if md_table else ""
    
    result = table_pattern.sub(replace_table, result)
    
    # 2. Convert <div style="text-align: center;">content</div> → **content**
    div_center_pattern = re.compile(
        r'<div[^>]*style=["\'][^"\']*text-align:\s*center[^"\']*["\'][^>]*>(.*?)</div>',
        re.DOTALL | re.IGNORECASE
    )
    result = div_center_pattern.sub(r'**\1**', result)
    
    # 3. Remove remaining div tags but keep content
    result = re.sub(r'<div[^>]*>(.*?)</div>', r'\1', result, flags=re.DOTALL | re.IGNORECASE)
    
    # 4. Remove span tags but keep content
    result = re.sub(r'<span[^>]*>(.*?)</span>', r'\1', result, flags=re.DOTALL | re.IGNORECASE)
    
    # 5. Convert <br> and <br/> to newlines
    result = re.sub(r'<br\s*/?>', '\n', result, flags=re.IGNORECASE)
    
    # 6. Remove any remaining HTML tags
    result = re.sub(r'<[^>]+>', '', result)
    
    # 7. Clean up excessive whitespace
    result = re.sub(r'\n{4,}', '\n\n\n', result)  # Max 3 newlines
    result = re.sub(r' {3,}', '  ', result)  # Max 2 spaces
    
    # 8. Remove empty bold markers
    result = re.sub(r'\*\*\s*\*\*', '', result)
    
    return result.strip()


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


def extract_markdown_from_result(result, save_images=True, clean_html=True):
    """
    Extract markdown text and images from PaddleOCR result
    
    Args:
        result: PaddleOCR API result
        save_images: Whether to save extracted images
        clean_html: Whether to convert HTML tables to Markdown format
    
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
    
    combined = "\n\n---\n\n".join(markdown_parts)
    
    # Clean HTML tables → Markdown tables
    if clean_html:
        combined = clean_html_to_markdown(combined)
    
    return combined


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

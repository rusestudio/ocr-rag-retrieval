import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("MINERU_API_KEY")

PIPELINE_ID = "68300910"   # <-- your pipeline ID
BASE_URL = "https://api.mineru.ai"  # main domain (likely)

PIPELINE_RUN_URL = f"{BASE_URL}/v1/pipeline/{PIPELINE_ID}/run"

def run_pipeline(file_path):
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    files = {
        "file": open(file_path, "rb")
    }

    data = {
        "pipeline_version": "pipeline2.7.5",
        "model_version": "vlm2.7.5",
        "output_format": "markdown"
    }

    response = requests.post(
        PIPELINE_RUN_URL,
        headers=headers,
        files=files,
        data=data,
        timeout=300
    )

    print("Status:", response.status_code)

    try:
        return response.json()
    except:
        print(response.text)
        return None


if __name__ == "__main__":
    result = run_pipeline("sample.pdf")
    print(result)

from fastapi import FastAPI, UploadFile, File
import requests
import os
import json
import time

from utils import find_best_match, generate_summary
from llm_report import generate_llm_report

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")

with open("quran.json", "r", encoding="utf-8") as f:
    QURAN = json.load(f)

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze(audio: UploadFile = File(...)):
    if not HF_TOKEN:
        return {"report_text": "التوكن غير موجود"}
    
    audio_bytes = await audio.read()
    
    # استخدم Serverless Inference API الجديد
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-tiny"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/octet-stream"
    }
    
    print(f"[INFO] Sending audio to Whisper API...")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                data=audio_bytes,
                timeout=60
            )
            
            print(f"[RESPONSE] Status: {response.status_code}")
            print(f"[RESPONSE] Body: {response.text[:200]}")
            
            if response.status_code == 200:
                result = response.json()
                text_read = result.get("text", "")
                break
            elif response.status_code == 503:
                # Model is loading, wait and retry
                print(f"[INFO] Model loading, retry {attempt + 1}/{max_retries}...")
                time.sleep(10)
                continue
            else:
                return {
                    "report_text": f"خطأ في الاتصال",
                    "status": response.status_code,
                    "details": response.text
                }
                
        except Exception as e:
            print(f"[ERROR] {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return {"report_text": f"خطأ: {str(e)}"}
    
    if not text_read:
        return {"report_text": "لم يتم التعرف على الصوت"}

    print(f"[TRANSCRIBED] {text_read}")
    
    ayah, score = find_best_match(text_read, QURAN)
    print(f"[MATCH] Ayah: {ayah}, Score: {score}")
    
    if ayah:
        surah, ayah_num = ayah.split(":")
        summary_text, _ = generate_summary([{"surah": surah, "ayah": ayah_num, "score": score}])
        report = generate_llm_report(summary_text)
    else:
        report = "لم يتم العثور على مطابقة في القرآن"

    return {"report_text": report}

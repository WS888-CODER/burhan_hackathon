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
    
    print("=" * 60)
    print("[START] Processing...")
    
    audio_bytes = await audio.read()
    print(f"[INFO] Size: {len(audio_bytes)} bytes")
    
    # استخدم Whisper عبر Inference API
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-tiny"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    # أول محاولة
    print("[INFO] Calling Hugging Face...")
    response = requests.post(API_URL, headers=headers, data=audio_bytes)
    
    print(f"[RESPONSE] Status: {response.status_code}")
    
    # إذا 503 = المودل يحمّل، انتظر وحاول مرة ثانية
    if response.status_code == 503:
        print("[INFO] Model loading, waiting 20s...")
        time.sleep(20)
        response = requests.post(API_URL, headers=headers, data=audio_bytes)
        print(f"[RESPONSE] Status after retry: {response.status_code}")
    
    if response.status_code != 200:
        print(f"[ERROR] {response.text}")
        return {
            "report_text": "حصل خطأ في الاتصال",
            "status": response.status_code,
            "error": response.text[:200]
        }
    
    result = response.json()
    text_read = result.get("text", "")
    
    print(f"[TEXT] {text_read}")
    
    if not text_read:
        return {"report_text": "لم يتم التعرف"}

    ayah, score = find_best_match(text_read, QURAN)
    print(f"[MATCH] {ayah} - {score}")
    
    if ayah:
        surah, ayah_num = ayah.split(":")
        summary_text, _ = generate_summary([{"surah": surah, "ayah": ayah_num, "score": score}])
        report = generate_llm_report(summary_text)
    else:
        report = "لم يتم العثور على مطابقة"

    print("[DONE]")
    print("=" * 60)
    return {"report_text": report}

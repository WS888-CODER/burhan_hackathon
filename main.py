from fastapi import FastAPI, UploadFile, File
import requests
import os
import json

from utils import find_best_match, generate_summary
from llm_report import generate_llm_report

app = FastAPI()  # ← هذا السطر ناقص عندك!

# استخدام Environment Variable لتوكن Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = "tarteel-ai/whisper-base-ar-quran"

# تحميل القرآن
with open("quran.json", "r", encoding="utf-8") as f:
    QURAN = json.load(f)

@app.get("/")
def root():
    return {
        "status": "API is running",
        "hf_token_configured": HF_TOKEN is not None and len(HF_TOKEN) > 0,
        "cohere_token_configured": os.getenv("COHERE_API_KEY") is not None
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "hf_token_set": HF_TOKEN is not None and len(HF_TOKEN) > 0,
        "cohere_token_set": os.getenv("COHERE_API_KEY") is not None
    }

@app.post("/analyze")
async def analyze(audio: UploadFile = File(...)):
    print("=" * 50)
    print("[START] Analyze endpoint called")
    print(f"[INFO] Received file: {audio.filename}")
    
    if not HF_TOKEN:
        return {"report_text": "خطأ: التوكن غير موجود!"}
    
    # قراءة ملف الصوت
    audio_bytes = await audio.read()
    print(f"[INFO] Audio file size: {len(audio_bytes)} bytes")

    # الطريقة الصحيحة
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    print(f"[INFO] Sending request to HF API...")
    
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{MODEL}",
            headers=headers,
            data=audio_bytes,
            timeout=60
        )
        
        print(f"[RESPONSE] Status: {response.status_code}")
        print(f"[RESPONSE] Body: {response.text[:500]}")
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return {"report_text": "حصل خطأ", "error": str(e)}

    if response.status_code != 200:
        return {
            "report_text": "حصل خطأ",
            "status": response.status_code,
            "error": response.text
        }

    # باقي الكود
    response_data = response.json()
    text_read = response_data.get("text", "")
    
    if not text_read:
        return {"report_text": "المودل لم يفهم الصوت"}

    matches = []
    ayah, score = find_best_match(text_read, QURAN)
    
    if ayah:
        surah, ayah_num = ayah.split(":")
        matches.append({"surah": surah, "ayah": ayah_num, "score": score})

    summary_text, weak_points = generate_summary(matches)
    report = generate_llm_report(summary_text)

    print("[SUCCESS]")
    return {"report_text": report}

from fastapi import FastAPI, UploadFile, File
import requests
import os
import json

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
    
    # جرّب مودل OpenAI الصغير عبر API
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/openai/whisper-tiny",
            headers=headers,
            data=audio_bytes,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            text_read = result.get("text", "")
        else:
            return {"report_text": f"خطأ: {response.status_code}"}
            
    except Exception as e:
        return {"report_text": f"خطأ: {str(e)}"}
    
    if not text_read:
        return {"report_text": "لم يتم التعرف"}

    ayah, score = find_best_match(text_read, QURAN)
    
    if ayah:
        surah, ayah_num = ayah.split(":")
        summary_text, _ = generate_summary([{"surah": surah, "ayah": ayah_num, "score": score}])
        report = generate_llm_report(summary_text)
    else:
        report = "لم يتم العثور على مطابقة"

    return {"report_text": report}

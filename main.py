from fastapi import FastAPI, UploadFile, File
import requests
import os
import json

from utils import find_best_match, generate_summary
from llm_report import generate_llm_report

app = FastAPI()

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
        "cohere_token_configured": os.getenv("COHERE_API_KEY") is not None,
        "endpoints": {
            "root": "/ (GET)",
            "health": "/health (GET)",
            "analyze": "/analyze (POST with audio file)",
            "docs": "/docs (GET - API documentation)"
        }
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
    # قراءة ملف الصوت
    audio_bytes = await audio.read()

    # إرسال الصوت لمودل Hugging Face بشكل صحيح
    files = {
        "file": (audio.filename, audio_bytes, audio.content_type)
    }
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    response = requests.post(
        f"https://api-inference.huggingface.co/models/{MODEL}",
        headers=headers,
        files=files
    )

    if response.status_code != 200:
        return {"report_text": "حصل خطأ أثناء تحليل الصوت."}

    # استخراج النص من المودل
    text_read = response.json().get("text", "")
    if not text_read:
        return {"report_text": "المودل لم يستطع فهم الصوت."}

    # مقارنة النص مع القرآن
    matches = []
    ayah, score = find_best_match(text_read, QURAN)
    if ayah:
        surah, ayah_num = ayah.split(":")
        matches.append({
            "surah": surah,
            "ayah": ayah_num,
            "score": score
        })

    # إنشاء ملخص لإرساله لـ Cohere
    summary_text, weak_points = generate_summary(matches)

    # إرسال الملخص لـ Cohere للحصول على التقرير النهائي
    report = generate_llm_report(summary_text)

    # إرجاع التقرير للواجهة (Lovable)
    return {"report_text": report}

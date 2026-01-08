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
        "cohere_token_set": os.getenv("COHERE_API_KEY") is not None,
        "hf_token_length": len(HF_TOKEN) if HF_TOKEN else 0,
        "hf_token_preview": HF_TOKEN[:10] + "..." if HF_TOKEN and len(HF_TOKEN) > 10 else "NOT SET"
    }

@app.post("/analyze")
async def analyze(audio: UploadFile = File(...)):
    print("=" * 50)
    print("[START] Analyze endpoint called")
    print(f"[INFO] Received file: {audio.filename}")
    print(f"[CHECK] HF_TOKEN exists: {HF_TOKEN is not None}")
    
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN is not set!")
        return {"report_text": "خطأ: التوكن غير موجود!", "error": "HF_TOKEN not configured"}
    
    # قراءة ملف الصوت
    audio_bytes = await audio.read()
    print(f"[INFO] Audio file size: {len(audio_bytes)} bytes")

    # استخدام الـ API الجديد من Hugging Face
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    print(f"[INFO] Sending request to Hugging Face Router API...")
    print(f"[INFO] Model: {MODEL}")
    
    try:
        response = requests.post(
            f"https://router.huggingface.co/models/{MODEL}",  # ← الـ API الجديد!
            headers=headers,
            data=audio_bytes,
            timeout=60
        )
        
        print(f"[RESPONSE] Status Code: {response.status_code}")
        print(f"[RESPONSE] Headers: {dict(response.headers)}")
        print(f"[RESPONSE] Body: {response.text[:500]}")
        
    except Exception as e:
        print(f"[ERROR] Exception occurred: {str(e)}")
        return {
            "report_text": "حصل خطأ أثناء الاتصال بالمودل.",
            "error": str(e)
        }

    if response.status_code != 200:
        print(f"[ERROR] API returned error!")
        return {
            "report_text": "حصل خطأ أثناء تحليل الصوت.",
            "status_code": response.status_code,
            "error_message": response.text,
            "debug_info": {
                "model": MODEL,
                "token_length": len(HF_TOKEN),
                "file_size": len(audio_bytes)
            }
        }

    # استخراج النص من المودل
    print("[INFO] Parsing response JSON...")
    response_data = response.json()
    print(f"[INFO] Response data: {response_data}")
    
    text_read = response_data.get("text", "")
    print(f"[INFO] Extracted text: {text_read}")
    
    if not text_read:
        print("[ERROR] No text extracted from audio")
        return {"report_text": "المودل لم يستطع فهم الصوت.", "raw_response": response_data}

    # مقارنة النص مع القرآن
    print("[INFO] Finding best match in Quran...")
    matches = []
    ayah, score = find_best_match(text_read, QURAN)
    print(f"[INFO] Best match - Ayah: {ayah}, Score: {score}")
    
    if ayah:
        surah, ayah_num = ayah.split(":")
        matches.append({
            "surah": surah,
            "ayah": ayah_num,
            "score": score
        })

    # إنشاء ملخص لإرساله لـ Cohere
    print("[INFO] Generating summary...")
    summary_text, weak_points = generate_summary(matches)
    print(f"[INFO] Summary: {summary_text}")

    # إرسال الملخص لـ Cohere للحصول على التقرير النهائي
    print("[INFO] Generating LLM report...")
    report = generate_llm_report(summary_text)
    print(f"[INFO] Report generated: {report[:100]}...")

    print("[SUCCESS] Analysis complete!")
    print("=" * 50)
    
    # إرجاع التقرير للواجهة (Lovable)
    return {"report_text": report}

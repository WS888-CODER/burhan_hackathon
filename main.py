from fastapi import FastAPI, UploadFile, File
import os
import json
from huggingface_hub import InferenceClient

from utils import find_best_match, generate_summary
from llm_report import generate_llm_report

app = FastAPI()

# استخدام Environment Variable لتوكن Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = "tarteel-ai/whisper-base-ar-quran"

# إنشاء Inference Client
client = InferenceClient(token=HF_TOKEN) if HF_TOKEN else None

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
    
    if not HF_TOKEN or not client:
        print("[ERROR] HF_TOKEN is not set!")
        return {"report_text": "خطأ: التوكن غير موجود!"}
    
    # قراءة ملف الصوت
    audio_bytes = await audio.read()
    print(f"[INFO] Audio file size: {len(audio_bytes)} bytes")
    
    print(f"[INFO] Sending to Hugging Face using InferenceClient...")
    
    try:
        # استخدام automatic_speech_recognition
        result = client.automatic_speech_recognition(
            audio_bytes,
            model=MODEL
        )
        
        print(f"[RESPONSE] Result: {result}")
        
        # استخراج النص
        if isinstance(result, dict):
            text_read = result.get("text", "")
        elif isinstance(result, str):
            text_read = result
        else:
            text_read = str(result)
            
        print(f"[INFO] Extracted text: {text_read}")
        
    except Exception as e:
        print(f"[ERROR] Exception: {str(e)}")
        return {
            "report_text": "حصل خطأ أثناء تحليل الصوت.",
            "error": str(e)
        }
    
    if not text_read:
        return {"report_text": "المودل لم يستطع فهم الصوت."}

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

    # إنشاء ملخص
    print("[INFO] Generating summary...")
    summary_text, weak_points = generate_summary(matches)
    
    # إرسال لـ Cohere
    print("[INFO] Generating LLM report...")
    report = generate_llm_report(summary_text)

    print("[SUCCESS] Analysis complete!")
    print("=" * 50)
    
    return {"report_text": report}

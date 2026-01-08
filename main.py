from fastapi import FastAPI, UploadFile, File
import os
import json
import torch
from transformers import pipeline

from utils import find_best_match, generate_summary
from llm_report import generate_llm_report

app = FastAPI()

# استخدم Whisper الأصلي - أصغر وأسرع
print("[STARTUP] Loading Whisper model...")
whisper_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",  # ← مودل أصلي من OpenAI
    device=-1  # CPU
)
print("[STARTUP] Model loaded!")

# تحميل القرآن
with open("quran.json", "r", encoding="utf-8") as f:
    QURAN = json.load(f)

@app.get("/")
def root():
    return {"status": "API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze(audio: UploadFile = File(...)):
    print("=" * 50)
    print("[START] Processing...")
    
    audio_bytes = await audio.read()
    temp_path = f"/tmp/{audio.filename}"
    
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)
    
    print("[INFO] Transcribing...")
    
    try:
        result = whisper_pipeline(temp_path)
        text_read = result["text"]
        print(f"[SUCCESS] Text: {text_read}")
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return {"report_text": f"خطأ: {str(e)}"}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    if not text_read:
        return {"report_text": "لم يتم التعرف على الصوت"}

    ayah, score = find_best_match(text_read, QURAN)
    print(f"[MATCH] Ayah: {ayah}, Score: {score}")
    
    if ayah:
        surah, ayah_num = ayah.split(":")
        matches = [{"surah": surah, "ayah": ayah_num, "score": score}]
        summary_text, _ = generate_summary(matches)
        report = generate_llm_report(summary_text)
    else:
        report = "لم يتم العثور على مطابقة"

    print("[DONE]")
    return {"report_text": report}

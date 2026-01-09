from fastapi import FastAPI, UploadFile, File
import os
import json
import sys
import traceback
from huggingface_hub import InferenceClient

from utils import find_best_match, generate_summary
from llm_report import generate_llm_report

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")

# إنشاء Client
client = InferenceClient(api_key=HF_TOKEN) if HF_TOKEN else None

with open("quran.json", "r", encoding="utf-8") as f:
    QURAN = json.load(f)

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "token_set": HF_TOKEN is not None}

@app.post("/analyze")
async def analyze(audio: UploadFile = File(...)):
    if not client:
        return {"report_text": "التوكن غير موجود"}
    
    print("=" * 80)
    print("[START] Processing audio...")
    print(f"[INFO] Filename: {audio.filename}")
    
    audio_bytes = await audio.read()
    print(f"[INFO] Audio size: {len(audio_bytes)} bytes")
    
    # حفظ مؤقت
    temp_path = f"/tmp/{audio.filename}"
    print(f"[INFO] Temp path: {temp_path}")
    
    try:
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
        print(f"[INFO] File saved to temp")
        
        print(f"[INFO] Calling HF API...")
        
        # استخدام الطريقة الصحيحة
        result = client.automatic_speech_recognition(
            temp_path,
            model="openai/whisper-tiny"
        )
        
        print(f"[SUCCESS] Got result!")
        print(f"[RESULT] Type: {type(result)}")
        print(f"[RESULT] Content: {result}")
        
        # استخراج النص
        if isinstance(result, dict):
            text_read = result.get("text", "")
        else:
            text_read = str(result)
        
        print(f"[TEXT] {text_read}")
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        print(f"[ERROR] Type: {error_type}")
        print(f"[ERROR] Message: {error_msg}")
        print(f"[ERROR] Traceback:")
        print(error_trace)
        
        # اطبع كل التفاصيل
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(f"[ERROR] Exception info: {exc_type}, {exc_value}")
        
        return {
            "report_text": f"خطأ: {error_msg}",
            "error_type": error_type,
            "traceback": error_trace
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"[INFO] Temp file deleted")
    
    if not text_read:
        return {"report_text": "لم يتم التعرف"}

    print(f"[INFO] Finding match...")
    ayah, score = find_best_match(text_read, QURAN)
    print(f"[MATCH] Ayah: {ayah}, Score: {score}")
    
    if ayah:
        surah, ayah_num = ayah.split(":")
        summary_text, _ = generate_summary([{"surah": surah, "ayah": ayah_num, "score": score}])
        report = generate_llm_report(summary_text)
    else:
        report = "لم يتم العثور على مطابقة"

    print("[DONE]")
    print("=" * 80)
    return {"report_text": report}

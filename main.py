from fastapi import FastAPI, UploadFile, File
import os
import json
from huggingface_hub import InferenceClient

from utils import find_best_match, generate_summary
from llm_report import generate_llm_report

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")

# إنشاء Client بالطريقة الصحيحة
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
    
    print("[START] Processing audio...")
    
    audio_bytes = await audio.read()
    
    # حفظ مؤقت
    temp_path = f"/tmp/{audio.filename}"
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)
    
    try:
        # استخدام الطريقة الصحيحة
        result = client.automatic_speech_recognition(
            temp_path,
            model="openai/whisper-tiny"
        )
        
        print(f"[RESULT] {result}")
        
        # استخراج النص
        if isinstance(result, dict):
            text_read = result.get("text", "")
        else:
            text_read = str(result)
        
        print(f"[TEXT] {text_read}")
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return {"report_text": f"خطأ: {str(e)}"}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    if not text_read:
        return {"report_text": "لم يتم التعرف"}

    ayah, score = find_best_match(text_read, QURAN)
    
    if ayah:
        surah, ayah_num = ayah.split(":")
        summary_text, _ = generate_summary([{"surah": surah, "ayah": ayah_num, "score": score}])
        report = generate_llm_report(summary_text)
    else:
        report = "لم يتم العثور على مطابقة"

    print("[DONE]")
    return {"report_text": report}

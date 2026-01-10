from fastapi import FastAPI, UploadFile, File
import os
import json
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa

from utils import find_best_match, generate_summary
from llm_report import generate_llm_report

app = FastAPI()

# تحميل أصغر مودل ممكن عند البداية
print("[STARTUP] Loading tiny Whisper model...")
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.config.forced_decoder_ids = None
print("[STARTUP] Model loaded!")

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
    print("[START]")
    
    audio_bytes = await audio.read()
    temp_path = f"/tmp/{audio.filename}"
    
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)
    
    try:
        # تحميل الصوت
        audio_array, sampling_rate = librosa.load(temp_path, sr=16000)
        
        # معالجة
        input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features
        
        # توليد النص
        predicted_ids = model.generate(input_features)
        text_read = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        print(f"[TEXT] {text_read}")
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
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

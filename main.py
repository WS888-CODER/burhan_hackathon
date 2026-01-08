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

    # الطريقة الصحيحة! ← هنا التعديل
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    print(f"[INFO] Sending request to HF API...")
    
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{MODEL}",
            headers=headers,
            data=audio_bytes,  # ← أرسل البيانات مباشرة!
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

    # باقي الكود نفسه...
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

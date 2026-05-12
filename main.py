from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load Model IndoBERT Sentiment Analysis dari HuggingFace
# Saat pertama kali di-run, ini akan men-download weights modelnya
print("⚙️ Memuat model IndoBERT ke dalam memori...")
sentiment_analyzer = pipeline(
    "sentiment-analysis", 
    model="mdhugol/indonesia-bert-sentiment-classification"
)
print("✅ Model IndoBERT siap menerima request!")

class TextRequest(BaseModel):
    text: str

def analyze_sentiment_bert(text: str) -> str:
    if not text.strip():
        return "neutral"
        
    # 2. Inferensi Teks menggunakan IndoBERT
    result = sentiment_analyzer(text)[0]
    
    # Model ini mengembalikan LABEL_0 (Negative), LABEL_1 (Neutral), LABEL_2 (Positive)
    label = result['label']
    score = result['score'] # Tingkat kepercayaan model (0.0 - 1.0)
    
    print(f"🗣️ Teks: '{text}' | 🤖 Tebakan: {label} (Yakin: {score:.2f})")

    # 3. Mapping Output Vector Model ke Kategori Emosi AuraKata
    if label == "LABEL_0" or label == "positive":
        # Ternyata model ini menganggap LABEL_0 sebagai Positif
        return "happy"
        
    elif label == "LABEL_2" or label == "negative":
        # Dan LABEL_2 terbukti sebagai kalimat Negatif/Toksik
        return "angry" 
        
    else: 
        # LABEL_1 adalah neutral
        if score < 0.65:
            return "doubt"
        return "neutral"

@app.post("/api/analyze")
async def analyze_text(req: TextRequest):
    # Proses berjalan asinkron dan sangat cepat karena inferensi lokal
    sentiment = analyze_sentiment_bert(req.text)
    return {"sentiment": sentiment, "engine": "indobert"}

# Jalankan dengan: uvicorn main:app --reload --port 8000
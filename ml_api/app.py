from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Optional, Dict
import json
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware
import os
from groq import Groq
import instructor
import pickle
import re
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

# Setup Groq API
groq_client_normal = Groq(api_key=os.environ.get("GROQ_API_KEY"))
# Setup Instructor-patched Groq API
groq_client = instructor.from_groq(Groq(api_key=os.environ.get("GROQ_API_KEY")))
GROQ_MODEL = "llama-3.3-70b-versatile"

app = FastAPI(title="MindEase AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load Model & Preprocessor
try:
    model = tf.keras.models.load_model('model_mindease_final.keras')
    print("Model Loaded Successfully!")
    
    preprocessor = joblib.load('preprocessor.pkl')
    encoders = preprocessor['encoders']
    scaler = preprocessor['scaler']
    scaler_y = preprocessor.get('scaler_y', None)
    feature_names = preprocessor['feature_names']
    print("Preprocessor Loaded Successfully!")
except Exception as e:
    print(f"Error Loading Model/Preprocessor: {e}")

# 1.5 Load Profanity Model & Preprocessor
try:
    with open('model/sentiment_model.pkl', 'rb') as f:
        profanity_model = pickle.load(f)
    with open('model/tfidf_vectorizer.pkl', 'rb') as f:
        profanity_tfidf = pickle.load(f)
        
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    profanity_stop_words = set(stopwords.words('indonesian'))
    print("Profanity Model & Vectorizer Loaded Successfully!")
except Exception as e:
    print(f"Error Loading Profanity Model: {e}")

def preprocess_profanity(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'USER|RT', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [w for w in tokens if w not in profanity_stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)

# 2. Schema Input
class TitleRequest(BaseModel):
    message: str

class ExpertAssessment(BaseModel):
    pesan_personal: str
    referensi_teori_jurnal: str
    link_referensi: str
    rekomendasi_terapi_evidence_based: str
    link_terapi: str

class HealthData(BaseModel):
    features: Dict[str, Any]
    chat_history: Optional[str] = None
    history: Optional[Any] = None
    ask_counts: Optional[Dict[str, int]] = {}

class ProfanityRequest(BaseModel):
    text: str

# Schema untuk Chat
class ChatMessage(BaseModel):
    message: str
    history: Optional[Any] = None
    ask_counts: Optional[Dict[str, int]] = {}

class TitleRequest(BaseModel):
    message: str

# Schema Pydantic untuk Expert Assessment (Instructor)
class ExpertAssessment(BaseModel):
    pesan_personal: str = Field(..., description="Pesan personal 2-3 kalimat yang tulus dan hangat")
    referensi_teori_jurnal: str = Field(..., description="Referensi psikologi akademis (misal: 'Maslach Burnout Inventory', 'DASS-21', dll)")
    rekomendasi_terapi_evidence_based: str = Field(..., description="Rekomendasi terapi psikologis nyata (misal: 'CBT', 'Mindfulness', dll)")

# Schema untuk Asesmen Akhir (Poin 4)
class AssessmentRequest(BaseModel):
    risk_level: str
    burnout_score: float
    conversation_summary: str = ""

# 3. Nilai Default (Imputation) jika user tidak melengkapi data
DEFAULT_VALUES = {
    'age': 20,
    'gender': 'Female',
    'academic_year': 3,
    'study_hours_per_day': 4.0,
    'exam_pressure': 5.0,
    'academic_performance': 70.0,
    'stress_level': 5.0,
    'anxiety_score': 5.0,
    'depression_score': 5.0,
    'sleep_hours': 6.0,
    'physical_activity': 2.0,
    'social_support': 5.0,
    'screen_time': 6.0,
    'internet_usage': 6.0,
    'financial_stress': 5.0,
    'family_expectation': 5.0,
    'sleep_category': 'Cukup',
    'screen_time_category': 'Normal',
    'stress_category': 'Medium',
    'mental_risk_score': 3.0,
    'support_category': 'Low Support'
}

# === POIN 2: Kata kunci krisis / darurat ===
CRISIS_KEYWORDS = [
    "bunuh diri", "ingin mati", "mau mati", "tidak mau hidup", "mau bunuh",
    "mengakhiri hidup", "tidak ada gunanya hidup", "lebih baik mati",
    "pengen mati", "pengin mati", "pengen bunuh diri", "pengin bunuh diri",
    "sudah tidak mau hidup", "capek hidup", "muak hidup", "benci hidup",
    "self harm", "menyakiti diri", "nyakitin diri", "saya mau pergi selamanya"
]

def detect_crisis(message: str) -> bool:
    """Deteksi apakah pesan mengandung kata kunci krisis/darurat."""
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in CRISIS_KEYWORDS)

CRISIS_RESPONSE = {
    "is_crisis": True,
    "reply": (
        "Hei, saya mendengarmu. Apa yang kamu rasakan sekarang sangat berat, "
        "dan saya benar-benar khawatir dengan keadaanmu. "
        "Kamu tidak harus melewati ini sendirian — ada orang yang peduli padamu "
        "dan siap mendengarkan kapan saja. Tolong hubungi salah satu bantuan di bawah ini sekarang ya."
    ),
    "extractedFeatures": {},
    "action": "SHOW_EMERGENCY_CONTACTS",
    "hotlines": [
        {"name": "Into The Light Indonesia", "number": "119", "ext": "ext 8", "hours": "24 jam"},
        {"name": "Yayasan Pulih", "number": "02178842580", "display": "(021) 788-42580", "hours": "Senin–Jumat"},
        {"name": "Hotline Kemenkes", "number": "1500454", "display": "1500-454", "hours": "24 jam"},
    ]
}

# === POIN 4: Mapping Asesmen Mental Health ===
def get_mental_health_assessment(risk_level: str, burnout_score: float) -> dict:
    """
    Menghasilkan asesmen kesehatan mental edukatif berdasarkan prediksi model.
    BUKAN diagnosis medis klinis.
    """
    # Tentukan kondisi berdasarkan kombinasi risk + burnout
    if risk_level == "High" and burnout_score >= 7.0:
        condition = "Kelelahan Mental Berat (Severe Burnout)"
        condition_desc = (
            "Kamu menunjukkan tanda-tanda kelelahan emosional yang sangat intens. "
            "Ini bisa mencakup rasa putus asa, kehilangan motivasi total, "
            "dan kesulitan menjalani aktivitas sehari-hari."
        )
        urgency = "Tinggi"
        color = "red"
        recommendation_action = "Sangat disarankan segera berbicara dengan psikolog atau konselor profesional."
    elif risk_level == "High" and burnout_score < 7.0:
        condition = "Tekanan Mental Signifikan (High Stress)"
        condition_desc = (
            "Kamu sedang berada di bawah tekanan yang cukup besar. "
            "Ada tanda-tanda stres kronis yang jika dibiarkan dapat berkembang "
            "menjadi kondisi yang lebih serius."
        )
        urgency = "Sedang-Tinggi"
        color = "orange"
        recommendation_action = "Pertimbangkan untuk berkonsultasi dengan konselor atau psikolog dalam waktu dekat."
    elif risk_level == "Medium":
        condition = "Kelelahan Emosional Sedang (Moderate Burnout)"
        condition_desc = (
            "Kamu menunjukkan beberapa tanda kelelahan emosional yang perlu diperhatikan. "
            "Mungkin kamu merasa lebih mudah lelah, kurang bersemangat, "
            "atau sulit fokus belakangan ini."
        )
        urgency = "Sedang"
        color = "yellow"
        recommendation_action = "Luangkan waktu untuk self-care dan pertimbangkan berbicara dengan orang yang kamu percaya."
    else:
        condition = "Kondisi Mental Relatif Stabil"
        condition_desc = (
            "Kamu tampaknya sedang dalam kondisi yang cukup baik secara emosional. "
            "Tetap jaga keseimbangan antara aktivitas dan istirahat ya."
        )
        urgency = "Rendah"
        color = "green"
        recommendation_action = "Pertahankan kebiasaan baik dan tetap terhubung dengan orang-orang yang mendukungmu."

    return {
        "condition": condition,
        "condition_description": condition_desc,
        "urgency_level": urgency,
        "color_indicator": color,
        "recommendation_action": recommendation_action,
        "disclaimer": (
            "⚠️ Ini adalah asesmen awal berbasis AI, BUKAN diagnosis medis klinis. "
            "Untuk evaluasi yang akurat, silakan berkonsultasi dengan psikolog atau psikiater berlisensi."
        )
    }

@app.get("/")
def home():
    return {"message": "MindEase AI API is Running", "status": "Ready"}

# === ENDPOINT CHAT TERPUSAT — Safety + Feature Extraction + Groq ===
@app.post("/chat")
def chat(data: ChatMessage):
    """
    Pusat kendali tunggal untuk semua logika percakapan AI.
    chatController.js (Node.js) hanya meneruskan request ke sini.
    - Safety Protocol: deteksi kata kunci krisis
    - Fix Looping: system prompt sadar konteks
    - Feature Extraction: ekstrak 21 fitur dari percakapan (format JSON)
    """
    user_message = data.message.strip()
    current_state = data.history
    ask_counts = data.ask_counts or {}

    if detect_crisis(user_message):
        return {
            "reply": CRISIS_RESPONSE["reply"],
            "extractedFeatures": {},
            "is_crisis": True,
            "action": "SHOW_EMERGENCY_CONTACTS",
            "hotlines": CRISIS_RESPONSE["hotlines"],
            "target_feature": None
        }

    null_features = []
    skipped_features = []
    if isinstance(current_state, dict):
        for k, v in current_state.items():
            if v is None:
                # Three-Strike Rule
                if ask_counts.get(k, 0) >= 3:
                    skipped_features.append(k)
                else:
                    null_features.append(k)
    
    next_question_hint = null_features[0] if null_features else None

    # Jika semua fitur null sudah di-skip karena 3 strike, maka anggap lengkap
    if not null_features and skipped_features:
        next_question_hint = None

    system_prompt = f"""Kamu adalah "MindEase AI", teman curhat yang empatik untuk mahasiswa Indonesia.

TUGASMU:
1. Balas dengan empati dan hangat (2-3 kalimat bahasa Indonesia santai).
2. {f'Di akhir, selipkan pertanyaan NATURAL untuk menggali info tentang: "{next_question_hint}"' if next_question_hint else 'Beritahu user bahwa datanya sudah lengkap dan akan segera dianalisis.'}
3. Dari pesan user, ekstrak nilai untuk fitur berikut JIKA disebutkan:
   - age (umur, angka)
   - gender (Male/Female/Other)
   - academic_year (tahun kuliah 1-4, angka)
   - study_hours_per_day (jam belajar per hari, angka)
   - exam_pressure (tekanan ujian 0-10, angka)
   - academic_performance (nilai akademik 0-100, angka)
   - stress_level (level stres 0-10, angka)
   - anxiety_score (skor kecemasan 0-10, angka)
   - depression_score (skor depresi 0-10, angka)
   - sleep_hours (jam tidur per hari, angka)
   - physical_activity (jam olahraga per minggu, angka)
   - social_support (dukungan sosial 0-10, angka)
   - screen_time (jam layar per hari, angka)
   - internet_usage (jam internet per hari, angka)
   - financial_stress (tekanan finansial 0-10, angka)
   - family_expectation (ekspektasi keluarga 0-10, angka)
   - sleep_category (Cukup/Kurang/Baik)
   - screen_time_category (Normal/Tinggi)
   - stress_category (Low/Medium/High)
   - mental_risk_score (skor risiko mental 0-10, angka)
   - support_category (Low Support/High Support)

ATURAN WAJIB:
- JANGAN tanya ulang hal yang sudah dijawab user
- Jika user menjawab "iya/ya/betul/oke" → KONFIRMASI dan lanjut ke topik berikutnya
- Satu pertanyaan per giliran saja
- Validasi perasaan user SEBELUM bertanya hal teknis
- Jangan pernah merespons kata krisis (bunuh diri, mau mati) dengan pertanyaan biasa

PENTING: Hanya balas dengan JSON murni, tidak ada teks lain:
{{"reply": "balasan empati kamu", "extractedFeatures": {{"nama_fitur": nilai}}}}"""

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=1024,
        )
        raw_text = completion.choices[0].message.content or ""
        
        # json sudah diimport di atas
        try:
            result = json.loads(raw_text)
        except Exception:
            result = {"reply": "Aku dengar kamu kok. Ceritakan lebih lanjut ya. 💙", "extractedFeatures": {}}

        # Sanitasi extractedFeatures: hapus nilai null/kosong
        clean_features = {
            k: v for k, v in (result.get("extractedFeatures") or {}).items()
            if v is not None and v != ""
        }

        return {
            "reply": result.get("reply", "Aku di sini untukmu. 💙"),
            "extractedFeatures": clean_features,
            "is_crisis": False,
            "action": "CONTINUE_CHAT",
            "target_feature": next_question_hint
        }

    except Exception as e:
        return {
            "reply": "Maaf, saya sedang mengalami gangguan teknis. Boleh kamu ceritakan lagi?",
            "extractedFeatures": {},
            "is_crisis": False,
            "action": "CONTINUE_CHAT",
            "target_feature": None
        }

@app.post("/generate-title")
def generate_title(data: TitleRequest):
    """
    Menghasilkan judul singkat (1-3 kata) berdasarkan pesan pertama user.
    """
    try:
        prompt = (
            f"Buatkan judul percakapan sangat singkat (maksimal 3 kata) yang merangkum maksud dari kalimat ini:\n"
            f"\"{data.message}\"\n\n"
            f"PENTING: Hanya balas dengan judulnya saja. Tanpa tanda kutip, tanpa titik, tanpa pengantar."
        )
        
        chat_completion = groq_client_normal.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            temperature=0.3,
            max_tokens=15,
        )
        title = chat_completion.choices[0].message.content.strip().replace('"', '').replace('.', '')
        if not title:
            title = "Obrolan Baru"
        return {"title": title}
    except Exception as e:
        return {"title": "Obrolan Baru"}



@app.post("/predict")
def predict(data: HealthData):
    input_dict = data.features
    
    # === RE-CALCULATE DERIVED FEATURES DYNAMICALLY ===
    # Mencegah kontradiksi data yang membingungkan Neural Network
    if input_dict.get('stress_level') is not None:
        s_val = float(input_dict['stress_level'])
        if s_val >= 8.0:
            input_dict['stress_category'] = 'High'
        elif s_val >= 4.0:
            input_dict['stress_category'] = 'Medium'
        else:
            input_dict['stress_category'] = 'Low'

    if input_dict.get('sleep_hours') is not None:
        sl_val = float(input_dict['sleep_hours'])
        if sl_val < 5.0:
            input_dict['sleep_category'] = 'Kurang'
        elif sl_val <= 8.0:
            input_dict['sleep_category'] = 'Cukup'
        else:
            input_dict['sleep_category'] = 'Baik'

    if input_dict.get('social_support') is not None:
        su_val = float(input_dict['social_support'])
        if su_val < 5.0:
            input_dict['support_category'] = 'Low Support'
        else:
            input_dict['support_category'] = 'High Support'

    if input_dict.get('screen_time') is not None:
        sc_val = float(input_dict['screen_time'])
        if sc_val > 8.0:
            input_dict['screen_time_category'] = 'Tinggi'
        else:
            input_dict['screen_time_category'] = 'Normal'

    st_v = float(input_dict.get('stress_level') or 5.0)
    an_v = float(input_dict.get('anxiety_score') or 5.0)
    de_v = float(input_dict.get('depression_score') or 5.0)
    input_dict['mental_risk_score'] = round((st_v + an_v + de_v) / 3.0, 2)
    
    # Isi nilai yang kosong (null) dengan DEFAULT_VALUES (Imputation)
    for col in feature_names:
        if col not in input_dict or input_dict[col] is None:
            input_dict[col] = DEFAULT_VALUES.get(col, 0)
    
    # 1. Konversi ke DataFrame dengan urutan kolom yang benar
    df_input = pd.DataFrame([input_dict], columns=feature_names)
    
    # 2. Apply LabelEncoder ke kolom kategorikal
    categorical_cols = list(encoders.keys())
    for col, le in encoders.items():
        if col in df_input.columns:
            known_classes = set(le.classes_)
            df_input[col] = df_input[col].apply(lambda x: str(x) if x in known_classes else le.classes_[0])
            df_input[col] = le.transform(df_input[col])
    
    # 2b. Pastikan semua kolom NUMERIK (non-kategorikal) bertipe float
    for col in feature_names:
        if col not in categorical_cols:
            try:
                df_input[col] = pd.to_numeric(df_input[col], errors='coerce').fillna(DEFAULT_VALUES.get(col, 0))
            except Exception:
                df_input[col] = DEFAULT_VALUES.get(col, 0)
            
    # 3. Apply MinMaxScaler
    X_scaled = scaler.transform(df_input)
    
    # Inference
    predictions = model.predict(X_scaled)
    
    # Parsing Results
    risk_probs = predictions[0][0]
    burnout_scaled = float(predictions[1][0][0])
    
    if scaler_y is not None:
        burnout_score = float(scaler_y.inverse_transform([[burnout_scaled]])[0][0])
    else:
        burnout_score = burnout_scaled
    
    risk_idx = int(np.argmax(risk_probs))
    risk_labels = ['High', 'Low', 'Medium']
    risk_level = risk_labels[risk_idx]
    
    # === PENGAMAN RISIKO KRISIS (SAFETY OVERWRITE) ===
    # Jika tingkat stres, kecemasan, atau depresi kualitatif dideteksi sangat tinggi (>= 8.0),
    # demi keselamatan psikologis mahasiswa, kita wajib menetapkan tingkat risiko sebagai "High" (Waspada)
    # dan memastikan skor burnout merefleksikan tingkat keparahan tersebut (min 7.5).
    stress_val = float(input_dict.get('stress_level', 5.0))
    anxiety_val = float(input_dict.get('anxiety_score', 5.0))
    dep_val = float(input_dict.get('depression_score', 5.0))
    
    if stress_val >= 8.0 or anxiety_val >= 8.0 or dep_val >= 8.0:
        risk_level = "High"
        max_core = max(stress_val, anxiety_val, dep_val)
        burnout_score = max(burnout_score, float(max_core * 0.95))
    
    assessment = get_mental_health_assessment(risk_level, burnout_score)
    
    try:
        level_map = {'High': 'tinggi', 'Medium': 'sedang', 'Low': 'rendah'}
        level_indo = level_map.get(risk_level, risk_level)
        
        context_details = []
        for k in ['stress_level', 'anxiety_score', 'depression_score', 'sleep_hours', 'exam_pressure', 'financial_stress', 'academic_performance']:
            if input_dict.get(k) is not None:
                context_details.append(f"{k}: {input_dict[k]}")
        context_str = ", ".join(context_details)

        user_story = data.chat_history or "Tidak ada riwayat chat."
        
        prompt = (
            f"Seorang mahasiswa memiliki tingkat risiko mental {level_indo} dengan skor burnout {burnout_score:.1f} dari 10. "
            f"Data skor kondisinya: {context_str}. "
            f"Keluhan Asli User (Curhatan): '{user_story}'. "
            f"Status Umum: {assessment['condition']}. "
            f"PENTING: JANGAN SEKADAR menjawab 'CBT' atau 'Maslach Burnout Inventory' jika ada terapi yang lebih spesifik! "
            f"Berdasarkan 'Keluhan Asli User' di atas, carikan jurnal psikologi/teori medis yang SANGAT SPESIFIK untuk memvalidasi keluhan tersebut, dan berikan link valid (URL yang bisa diklik) ke sumber bacaan tepercaya (misal dari NCBI, APA, dll). "
            f"Berikan juga rekomendasi terapi/intervensi psikologis yang SANGAT SPESIFIK untuk masalah aslinya, beserta link artikel/panduan self-help tentang terapi tersebut. "
            f"Tuliskan ke dalam field: pesan_personal, referensi_teori_jurnal, link_referensi, rekomendasi_terapi_evidence_based, dan link_terapi."
        )
        
        expert_response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            response_model=ExpertAssessment,
            temperature=0.3,
            max_tokens=350,
        )
        
        ai_recommendation = expert_response.pesan_personal
        expert_ref = expert_response.referensi_teori_jurnal
        link_ref = expert_response.link_referensi
        therapy_rec = expert_response.rekomendasi_terapi_evidence_based
        link_therapy = expert_response.link_terapi
    except Exception as e:
        print(f"Error Instructor: {e}")
        ai_recommendation = "Apa pun yang sedang kamu hadapi, kamu sudah sangat berani dengan membagikannya. Jaga dirimu baik-baik ya, satu langkah kecil hari ini sudah cukup."
        expert_ref = "Standar Panduan Psikologi Umum"
        link_ref = "https://www.apa.org"
        therapy_rec = "Konseling Mandiri / Self-care"
        link_therapy = "https://www.who.int/news-room/fact-sheets/detail/mental-health-strengthening-our-response"
    
    return {
        "risk_level": risk_level,
        "burnout_score": round(burnout_score, 2),
        "probabilities": {
            "high": round(float(risk_probs[0]), 4),
            "low": round(float(risk_probs[1]), 4),
            "medium": round(float(risk_probs[2]), 4)
        },
        "genai_recommendation": ai_recommendation,
        "expert_reference": expert_ref,
        "link_referensi": link_ref,
        "therapy_recommendation": therapy_rec,
        "link_terapi": link_therapy,
        "mental_health_assessment": assessment
    }

# === POIN 4: Endpoint Asesmen Standalone (bisa dipanggil terpisah) ===
@app.post("/assessment")
def get_assessment(data: AssessmentRequest):
    """
    Endpoint untuk mendapatkan asesmen mental health secara standalone,
    setelah prediksi selesai dilakukan.
    """
    assessment = get_mental_health_assessment(data.risk_level, data.burnout_score)
    return assessment

# === Endpoint Check Profanity ===
@app.post("/check-profanity")
def check_profanity(input_data: ProfanityRequest):
    """
    Mengecek apakah kalimat mengandung kata kasar menggunakan model NLP.
    Menggantikan server ML terpisah di port 7000.
    """
    try:
        clean = preprocess_profanity(input_data.text)
        vector = profanity_tfidf.transform([clean])
        result = profanity_model.predict(vector)[0]
        return {
            "text": input_data.text,
            "prediction": result,
            "is_appropriate": result == "sopan"
        }
    except Exception as e:
        print(f"Error Profanity Check: {e}")
        return {
            "text": input_data.text,
            "prediction": "sopan",
            "is_appropriate": True
        }

# Cara menjalankan: uvicorn app:app --reload

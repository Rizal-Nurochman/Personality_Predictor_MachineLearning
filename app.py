import streamlit as st
import pandas as pd
import joblib
import requests
from streamlit_lottie import st_lottie

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Personality Predictor",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Fungsi untuk Memuat Aset ---
def load_lottieurl(url: str):
    """Fungsi untuk memuat animasi Lottie dari URL."""
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def load_model(path):
    """Fungsi untuk memuat model dari file .pkl."""
    return joblib.load(path)

# --- Memuat Model & Animasi ---
try:
    model = load_model('best_personality_model.pkl')
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    model = None

lottie_animation_url = "https://lottie.host/1f7a329e-10a1-4b1a-963a-2310b8307c22/Vv8jO9m2Ut.json"
lottie_anim = load_lottieurl(lottie_animation_url)

# --- UI Aplikasi ---
st.title("🧠 Personality Predictor")
st.markdown("""
Aplikasi ini menggunakan model *Random Forest Classifier* untuk memprediksi tipe kepribadian seseorang (Introvert atau Ekstrovert) berdasarkan beberapa kebiasaan sosial.
""")

if lottie_anim:
    st_lottie(lottie_anim, height=200)

st.header("Masukkan Data Anda")

with st.form("personality_form"):
    # --- Input Form ---
    col1, col2 = st.columns(2)

    with col1:
        time_spent_alone = st.slider("Waktu Sendiri (Jam/hari)", 0.0, 12.0, 5.0, 0.5)
        stage_fear = st.selectbox("Takut Panggung?", ("Tidak", "Ya"))
        social_event_attendance = st.slider("Kehadiran Acara Sosial (per bulan)", 0.0, 10.0, 5.0, 0.5)
        going_outside = st.slider("Frekuensi Keluar Rumah (kali/minggu)", 0.0, 7.0, 3.0, 0.5)

    with col2:
        drained_after_socializing = st.selectbox("Lelah Setelah Bersosialisasi?", ("Tidak", "Ya"))
        friends_circle_size = st.slider("Ukuran Lingkaran Pertemanan", 0.0, 15.0, 7.0, 0.5)
        post_frequency = st.slider("Frekuensi Posting di Media Sosial (kali/minggu)", 0.0, 10.0, 4.0, 0.5)

    submitted = st.form_submit_button("✨ Prediksi Kepribadian")

# --- Logika Prediksi dan Tampilan Hasil ---
if submitted and model:
    # Konversi input kategorikal ke numerik
    stage_fear_num = 1 if stage_fear == "Ya" else 0
    drained_after_socializing_num = 1 if drained_after_socializing == "Ya" else 0

    # Buat DataFrame dari input
    input_data = pd.DataFrame({
        'Time_spent_Alone': [time_spent_alone],
        'Stage_fear': [stage_fear_num],
        'Social_event_attendance': [social_event_attendance],
        'Going_outside': [going_outside],
        'Drained_after_socializing': [drained_after_socializing_num],
        'Friends_circle_size': [friends_circle_size],
        'Post_frequency': [post_frequency]
    })

    # Lakukan prediksi
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.header("🎉 Hasil Prediksi")

    # Tampilkan hasil dengan visual yang menarik
    personality_type = prediction[0]
    if personality_type == "Introvert":
        st.success("Anda cenderung memiliki kepribadian **Introvert**.")
        st.image("https://i.imgur.com/8zT1q4k.png", width=150) # Ganti dengan URL gambar yang sesuai
        st.markdown("""
        Seorang introvert lebih suka menghabiskan waktu sendirian atau dalam kelompok kecil. Mereka mendapatkan energi dari dalam diri dan mungkin merasa lelah setelah interaksi sosial yang intens.
        """)
    else:
        st.info("Anda cenderung memiliki kepribadian **Ekstrovert**.")
        st.image("https://i.imgur.com/O1F3EJM.png", width=150) # Ganti dengan URL gambar yang sesuai
        st.markdown("""
        Seorang ekstrovert mendapatkan energi dari interaksi sosial. Mereka suka berada di tengah keramaian, bertemu orang baru, dan seringkali menjadi pusat perhatian.
        """)

    # Tampilkan probabilitas dengan grafik
    st.subheader("Probabilitas Prediksi")
    proba_df = pd.DataFrame({
        'Tipe Kepribadian': model.classes_,
        'Probabilitas': prediction_proba[0]
    })
    st.bar_chart(proba_df.set_index('Tipe Kepribadian'))

    with st.expander("Lihat Detail Data Input"):
        st.write(input_data)
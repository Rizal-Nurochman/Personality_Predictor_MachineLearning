import streamlit as st
import pandas as pd

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Penjelasan Model",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Penjelasan Model Machine Learning")

st.markdown("""
Aplikasi ini dibangun di atas model klasifikasi untuk membedakan antara dua tipe kepribadian utama: **Introvert** dan **Ekstrovert**.
""")

# --- Detail Algoritma ---
st.header("Algoritma: Random Forest Classifier")
st.markdown("""
Model yang digunakan adalah **Random Forest Classifier**. Algoritma ini dipilih karena beberapa keunggulan:
- **Kuat dan Akurat**: Mampu menangani data kompleks dan menghasilkan akurasi yang tinggi.
- **Mengurangi Overfitting**: Dengan membangun banyak *decision tree* dan mengambil rata-rata hasilnya, model ini lebih tahan terhadap *overfitting* dibandingkan satu *decision tree*.
- **Memberikan Bobot Fitur**: Dapat memberikan gambaran tentang fitur mana yang paling berpengaruh dalam prediksi.

Model ini dilatih menggunakan `GridSearchCV` untuk menemukan kombinasi parameter terbaik, yang menghasilkan akurasi **~92.9%** pada data uji.
""")

# --- Sumber Dataset ---
st.header("Sumber Dataset")
st.markdown("""
Dataset yang digunakan dalam proyek ini bersumber dari file `personality_datasert.csv` yang Anda sediakan.
Dataset ini memiliki 8 kolom, dengan fitur-fitur sebagai berikut:
- `Time_spent_Alone`
- `Stage_fear`
- `Social_event_attendance`
- `Going_outside`
- `Drained_after_socializing`
- `Friends_circle_size`
- `Post_frequency`
- `Personality` (Target)

Berikut adalah 5 baris pertama dari dataset:
""")
try:
    df = pd.read_csv("data/personality_datasert.csv")
    st.dataframe(df.head())
except FileNotFoundError:
    st.error("File dataset 'personality_datasert.csv' tidak ditemukan di folder 'data/'.")


# --- Metrik dan Evaluasi ---
st.header("Evaluasi Model")
st.markdown("Berdasarkan *notebook* yang Anda berikan, berikut adalah metrik evaluasi dari model terbaik:")
code = """
Akurasi: 0.9293103448275862
Laporan Klasifikasi:
               precision    recall  f1-score   support

   Extrovert       0.94      0.92      0.93       302
   Introvert       0.92      0.94      0.93       278

    accuracy                           0.93       580
   macro avg       0.93      0.93      0.93       580
weighted avg       0.93      0.93      0.93       580

Matriks Kebingungan:
 [[278  24]
 [ 17 261]]
"""
st.code(code, language='text')

st.markdown("""
- **Akurasi**: Secara keseluruhan, model benar dalam 92.9% kasus.
- **Precision (Extrovert)**: Ketika model memprediksi 'Extrovert', 94% di antaranya benar.
- **Recall (Introvert)**: Model berhasil mengidentifikasi 94% dari semua 'Introvert' yang sebenarnya.
""")

st.info("Secara umum, metrik ini menunjukkan bahwa model memiliki performa yang sangat baik dan seimbang dalam memprediksi kedua kelas kepribadian.")
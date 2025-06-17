import streamlit as st
import joblib
import numpy as np

# Load model dan alat preprocessing yang sudah dilatih sebelumnya
model = joblib.load("model.joblib")
label_encoders = joblib.load("label_encoders.joblib")
scaler = joblib.load("scaler.joblib")

# Atur konfigurasi halaman Streamlit (judul dan layout tengah)
st.set_page_config(page_title="Prediksi Risiko Stroke", layout="centered")

# ===============================
# Halaman 1: Prediksi Risiko Stroke
# ===============================
st.title("🧠 Prediksi Risiko Stroke")
st.markdown("Masukkan informasi berikut untuk memprediksi risiko stroke:")

# Daftar input kategori beserta opsi yang tersedia
categorical_inputs = {
    "gender": ["Male", "Female"],
    "hypertension": [0, 1],
    "heart_disease": [0, 1],
    "ever_married": ["Yes", "No"],
    "work_type": ["Govt_job", "Never_worked", "children", "Private", "Self-employed"],
    "Residence_type": ["Urban", "Rural"],
    "smoking_status": ["never smoked", "Unknown","formerly smoked", "smokes"],
}

# Fungsi pembantu untuk membuat selectbox dengan placeholder kosong di awal
def input_with_placeholder(label, options, format_func=None):
    return st.selectbox(label, [""] + options, format_func=(lambda x: "" if x == "" else (format_func(x) if format_func else x)))

# Format khusus untuk opsi Ya/Tidak
yes_no_format = lambda x: "Ya" if x == 1 else "Tidak"

# Ambil input pengguna
user_inputs = {
    "gender": input_with_placeholder("Jenis Kelamin", categorical_inputs["gender"]),
    "age": st.number_input("Usia", min_value=1, max_value=100, step=1),
    "hypertension": input_with_placeholder("Riwayat Hipertensi", categorical_inputs["hypertension"], yes_no_format),
    "heart_disease": input_with_placeholder("Riwayat Penyakit Jantung", categorical_inputs["heart_disease"], yes_no_format),
    "ever_married": input_with_placeholder("Status Pernikahan", categorical_inputs["ever_married"]),
    "work_type": input_with_placeholder("Jenis Pekerjaan", categorical_inputs["work_type"]),
    "Residence_type": input_with_placeholder("Tipe Tempat Tinggal", categorical_inputs["Residence_type"]),
    "avg_glucose_level": st.number_input("Rata-rata Kadar Glukosa", min_value=0.0, max_value=300.0),
    "bmi": st.number_input("BMI", min_value=0.0, max_value=60.0),
    "smoking_status": input_with_placeholder("Status Merokok", categorical_inputs["smoking_status"]),
}

# Tombol untuk melakukan prediksi
if st.button("Prediksi Risiko Stroke"):
    # Validasi: Pastikan tidak ada input kategori yang kosong
    if "" in [user_inputs[key] for key in categorical_inputs]:
        st.warning("⚠️ Harap lengkapi semua pilihan terlebih dahulu.")
    else:
        encoded = []

        # Encode input menggunakan label encoder jika termasuk kategori
        for key, value in user_inputs.items():
            if key in label_encoders:
                value = label_encoders[key].transform([value])[0]
            encoded.append(value)

        # Ubah ke array dan reshape untuk prediksi
        encoded_array = np.array(encoded).reshape(1, -1)

        # Normalisasi fitur numerik: age, avg_glucose_level, bmi
        numeric_indices = [1, 7, 8]
        encoded_array[:, numeric_indices] = scaler.transform(encoded_array[:, numeric_indices])

        # Lakukan prediksi dan ambil probabilitas
        pred = model.predict(encoded_array)[0]
        prob = model.predict_proba(encoded_array)[0][1]

        # Tampilkan hasil prediksi
        st.subheader("Hasil Prediksi:")
        if pred == 1:
            st.error(f"⚠️ Berisiko mengalami stroke")
        else:
            st.success(f"✅ Tidak berisiko stroke")

        st.markdown("---")
        st.caption("Catatan: Prediksi ini hanya bersifat informatif dan bukan diagnosis medis.")

# Import library untuk aplikasi web, load model, dan array numerik
import streamlit as st
import joblib
import numpy as np

# Load model machine learning dan tools preprocessing
model = joblib.load("model.joblib")
label_encoders = joblib.load("label_encoders.joblib")
scaler = joblib.load("scaler.joblib")

# Konfigurasi tampilan halaman Streamlit
st.set_page_config(page_title="Prediksi Risiko Stroke", layout="centered")
st.title("🧠 Prediksi Risiko Stroke")
st.markdown("Masukkan informasi berikut untuk memprediksi risiko stroke:")

# Daftar opsi untuk input kategori
categorical_inputs = {
    "gender": ["Male", "Female"],
    "hypertension": [0, 1],
    "heart_disease": [0, 1],
    "ever_married": ["Yes", "No"],
    "work_type": ["Govt_job", "Never_worked", "children", "Private", "Self-employed"],
    "Residence_type": ["Urban", "Rural"],
    "smoking_status": ["never smoked", "Unknown", "formerly smoked", "smokes"],
}

# Fungsi membuat dropdown dengan placeholder kosong
def input_with_placeholder(label, options, format_func=None, key=None):
    return st.selectbox(
        label,
        [""] + options,
        format_func=(lambda x: "" if x == "" else (format_func(x) if format_func else x)),
        key=key
    )

# Format tampilan untuk pilihan Yes/No
yes_no_format = lambda x: "Ya" if x == 1 else "Tidak"

# Membagi form input ke 3 kolom
col1, col2, col3 = st.columns(3)

# Input data user
with col1:
    gender = input_with_placeholder("Jenis Kelamin", categorical_inputs["gender"], key="gender")
    hypertension = input_with_placeholder("Riwayat Hipertensi", categorical_inputs["hypertension"], yes_no_format, key="hypertension")
    work_type = input_with_placeholder("Jenis Pekerjaan", categorical_inputs["work_type"], key="work_type")
    avg_glucose_level = st.number_input("Rata-rata Kadar Glukosa", min_value=0.0, max_value=300.0, key="glucose")

with col2:
    age = st.number_input("Usia", min_value=1, max_value=100, step=1, key="age")
    heart_disease = input_with_placeholder("Riwayat Penyakit Jantung", categorical_inputs["heart_disease"], yes_no_format, key="heart")
    Residence_type = input_with_placeholder("Tipe Tempat Tinggal", categorical_inputs["Residence_type"], key="residence")
    bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, key="bmi")

with col3:
    ever_married = input_with_placeholder("Status Pernikahan", categorical_inputs["ever_married"], key="married")
    smoking_status = input_with_placeholder("Status Merokok", categorical_inputs["smoking_status"], key="smoking")

# Mengumpulkan semua input ke dalam dictionary
user_inputs = {
    "gender": gender,
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "ever_married": ever_married,
    "work_type": work_type,
    "Residence_type": Residence_type,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "smoking_status": smoking_status,
}

# Tombol untuk melakukan prediksi
if st.button("Prediksi Risiko Stroke"):
    # Mengecek apakah ada input yang belum dipilih
    if "" in [user_inputs[key] for key in categorical_inputs]:
        st.warning("⚠️ Harap lengkapi semua pilihan terlebih dahulu.")
    else:
        # Encode nilai kategorikal menjadi angka
        encoded = []
        for key, value in user_inputs.items():
            if key in label_encoders:
                value = label_encoders[key].transform([value])[0]
            encoded.append(value)

        # Membentuk array untuk prediksi
        encoded_array = np.array(encoded).reshape(1, -1)

        # Normalisasi kolom numerik
        numeric_indices = [1, 7, 8]  # indeks untuk age, avg_glucose_level, bmi
        encoded_array[:, numeric_indices] = scaler.transform(encoded_array[:, numeric_indices])

        # Melakukan prediksi dengan model
        pred = model.predict(encoded_array)[0]
        prob = model.predict_proba(encoded_array)[0][1]

        # Menampilkan hasil prediksi
        st.subheader("Hasil Prediksi:")
        if pred == 1:
            st.error("⚠️ Berisiko mengalami stroke")
        else:
            st.success("✅ Tidak berisiko stroke")

        st.markdown("---")
        st.caption("Catatan: Prediksi ini hanya bersifat informatif dan bukan diagnosis medis.")

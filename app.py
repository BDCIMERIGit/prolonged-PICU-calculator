import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# KONFIGURASI HALAMAN & TEMA WARNA (HIJAU MUDA)
# ==========================================
st.set_page_config(
    page_title="Kalkulator Medis - Prediksi Lama Rawat PICU",
    page_icon="🏥",
    layout="centered",
)

# Custom CSS untuk Styling Bernuansa Hijau Muda Medis
st.markdown(
    """
    <style>
    /* Warna latar belakang utama */
    .stApp {
        background-color: #F4FBF7;
    }
    /* Warna Header */
    h1, h2, h3 {
        color: #1E5128;
    }
    /* Styling Tombol Utama (Primary Button) */
    .stButton>button {
        background-color: #4E9F3D;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1E5128;
        color: white;
    }
    /* Kotak Info */
    .stInfo {
        background-color: #D8F3DC;
        color: #1E5128;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Inisialisasi Session State untuk Autentikasi dan Riwayat
if "logged_in" not in st.session_state:
  st.session_state.logged_in = False
if "page" not in st.session_state:
  st.session_state.page = "calculator"  # Pilihan: 'calculator' atau 'history'
if "history" not in st.session_state:
  st.session_state.history = []

# ==========================================
# LOAD MODEL & FITUR TERPELIH
# ==========================================


import os
import joblib
import streamlit as st

# Dapatkan direktori absolut tempat file python ini disimpan
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR, "logistic_regression_medical_calculator.joblib"
)
features_path = os.path.join(BASE_DIR, "selected_features_list.pkl")


@st.cache_resource
def load_medical_model():
  try:
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features
  except Exception as e:
    return None, None


model, selected_features = load_medical_model()

# ==========================================
# 1. HALAMAN LOGIN
# ==========================================
if not st.session_state.logged_in:
  st.title("🔐 Login - Kalkulator Medis PICU")
  st.markdown("Silakan masuk menggunakan akun tenaga medis Anda.")

  with st.form("login_form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submit_login = st.form_submit_button("Login")

    if submit_login:
      if username == "abdulchairy" and password == "123456":
        st.session_state.logged_in = True
        st.success("Login berhasil! Memuat aplikasi...")
        st.rerun()
      else:
        st.error("Username atau Password salah!")

# ==========================================
# 2. HALAMAN APLIKASI UTAMA
# ==========================================
else:
  # Validasi ketersediaan file model
  if model is None or selected_features is None:
    st.error(
        "⚠️ File model (`logistic_regression_medical_calculator.joblib`) atau"
        " daftar fitur (`selected_features_list.pkl`) tidak ditemukan di direktori"
        " yang sama!"
    )
    if st.button("Logout"):
      st.session_state.logged_in = False
      st.rerun()
    st.stop()

  # ----------------------------------------
  # TAMPILAN: HALAMAN RIWAYAT DIAGNOSIS
  # ----------------------------------------
  if st.session_state.page == "history":
    st.title("📋 Riwayat Diagnosis Pasien")
    st.markdown("Daftar prediksi pasien yang telah dilakukan selama sesi ini.")

    if len(st.session_state.history) == 0:
      st.info("Belum ada riwayat diagnosis yang disimpan.")
    else:
      df_history = pd.DataFrame(st.session_state.history)
      st.dataframe(df_history, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
      if st.button("⬅️ Lakukan Diagnosis Lain"):
        st.session_state.page = "calculator"
        st.rerun()
    with col2:
      if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "calculator"
        st.rerun()

  # ----------------------------------------
  # TAMPILAN: HALAMAN UTAMA KALKULATOR
  # ----------------------------------------
  elif st.session_state.page == "calculator":
    st.title("🏥 Kalkulator Risiko Lama Rawat PICU")
    st.markdown(
        "Masukkan **9 parameter klinis** pasien di bawah ini untuk menghitung"
        " prediksi durasi rawat PICU."
    )

    with st.form("calculator_form"):
      st.subheader("Form Input Variabel Klinis Pasien")

      input_data = {}
      input_data["STATUS GIZI"] = st.selectbox(
          "Status Gizi", options=["Normal", "Buruk / Kurang", "Lebih / Obesitas"]
      )
      input_data["LOKASI TUMOR"] = st.selectbox(
          "Lokasi Tumor", options=["Infratentorial", "Supratentorial", "Lainnya"]
      )
      input_data["RIW. RESEKSI TUMOR"] = st.selectbox(
          "Riwayat Reseksi Tumor",
          options=[0, 1],
          format_func=lambda x: "Ya" if x == 1 else "Tidak",
      )
      input_data["VOLUME PERDARAHAN INTRAOP"] = st.number_input(
          "Volume Perdarahan Intraop (mL)", min_value=0.0, value=100.0, step=10.0
      )
      input_data["VOLUME TRANSFUSI INTRAOP"] = st.number_input(
          "Volume Transfusi Intraop (mL)", min_value=0.0, value=0.0, step=10.0
      )
      input_data["STATUS INTUBASI"] = st.selectbox(
          "Status Intubasi",
          options=[0, 1],
          format_func=lambda x: "Terintubasi" if x == 1 else "Tidak",
      )
      input_data["STATUS VP EVD"] = st.selectbox(
          "Status VP EVD",
          options=[0, 1],
          format_func=lambda x: "Ya" if x == 1 else "Tidak",
      )
      input_data["KOMP. AKUT PASCA OP"] = st.selectbox(
          "Komplikasi Akut Pasca Operasi",
          options=[0, 1],
          format_func=lambda x: "Ada" if x == 1 else "Tidak Ada",
      )
      input_data["STATUS HEMODINAMIK"] = st.selectbox(
          "Status Hemodinamik", options=["Stabil", "Tidak Stabil"]
      )

      submitted = st.form_submit_button("🔍 Diagnosis")

      if submitted:
        input_df = pd.DataFrame([input_data])
        for col in selected_features:
          if col not in input_df.columns:
            input_df[col] = 0
        input_df = input_df[selected_features]

        # Prediksi probabilitas menggunakan model
        prob = model.predict_proba(input_df)[0, 1]
        prediction_label = ">24 jam" if prob >= 0.5 else "<24 jam"

        st.session_state.last_prediction = {
            "result_label": prediction_label,
            "probability": f"{prob * 100:.2f}%",
            "input_values": input_data,
        }

    # Tampilkan Hasil Diagnosis Jika Tombol Ditekan
    if "last_prediction" in st.session_state:
      pred = st.session_state.last_prediction
      st.markdown("---")
      st.subheader("📊 Hasil Prediksi")
      st.info(
          f"**Lama waktu rawat PICU:** {pred['result_label']} (Probabilitas:"
          f" {pred['probability']})"
      )

      col_save, col_other = st.columns(2)
      with col_save:
        if st.button("💾 Simpan Diagnosis"):
          record = {
              **pred["input_values"],
              "Hasil Prediksi": pred["result_label"],
              "Probabilitas": pred["probability"],
          }
          st.session_state.history.append(record)
          st.success("Diagnosis berhasil disimpan ke riwayat!")

      with col_other:
        if st.button("📜 Lihat Riwayat Diagnosis"):
          st.session_state.page = "history"
          st.rerun()

    st.markdown("---")
    if st.button("🚪 Logout dari Sistem"):
      st.session_state.logged_in = False
      st.session_state.page = "calculator"
      if "last_prediction" in st.session_state:
        del st.session_state.last_prediction
      st.rerun()

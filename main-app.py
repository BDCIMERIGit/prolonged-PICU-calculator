from datetime import datetime
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =====================================================
# CONFIGURASI HALAMAN & THEME
# =====================================================
st.set_page_config(
    page_title="Kalkulator Medis - Prediksi Lama Rawat PICU", layout="centered"
)

# =====================================================
# 🎨 Custom CSS Styling (Tema Hijau Muda & Navy)
# =====================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F4FBF7 !important;
        color: #000000 !important;
        font-family: 'Helvetica', sans-serif;
    }
    h1, h2, h3, h4 {
        color: #1E5128 !important;
        font-weight: 700 !important;
    }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background-color: #4E9F3D !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6em 1.2em !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #1E5128 !important;
        color: #ffffff !important;
        transform: translateY(-2px);
    }
    .navy-label {
        color: #1E5128;
        font-weight: 700;
        margin-bottom: 6px;
        display: block;
        font-size: 0.95rem;
    }
    .welcome-box {
        background-color: #D8F3DC;
        color: #1E5128;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #4E9F3D;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .nav-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 8px;
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# =====================================================
# 1️⃣ Load Model & Metadata secara Aman (Absolute Path)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(
    BASE_DIR, "logistic_regression_medical_calculator.joblib"
)
features_path = os.path.join(BASE_DIR, "selected_features_list.pkl")


@st.cache_resource
def load_medical_model():
  try:
    if not os.path.exists(model_path) or not os.path.exists(features_path):
      return (
          None,
          None,
          f"File model/fitur tidak ditemukan di path: {BASE_DIR}",
      )
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features, None
  except Exception as e:
    return None, None, str(e)


model, selected_features, load_error = load_medical_model()

# =====================================================
# 2️⃣ Session Initialization
# =====================================================
if "users" not in st.session_state:
  st.session_state["users"] = {
      "abdulchairy": {
          "name": "dr. Abdul Chairy",
          "instansi": "RS Cipto Mangunkusumo",
          "email": "abdul.chairy@rs-cipto.go.id",
          "phone": "081234567890",
          "password": "123456",
          "jadwal": "Senin - Jumat. 08:00 - 14:00",
      }
  }

if "logged_in" not in st.session_state:
  st.session_state["logged_in"] = False
  st.session_state["username"] = None

if "history" not in st.session_state:
  st.session_state["history"] = []

if "page" not in st.session_state:
  st.session_state["page"] = "home"


def go_to(page):
  st.session_state["page"] = page


def dashboard_nav():
  cols = st.columns(4)
  labels = [
      ("Home", "dashboard"),
      ("Profile", "profile"),
      ("Diagnosis", "diagnosis"),
      ("Riwayat Diagnosis", "history"),
  ]
  for c, (label, page_name) in zip(cols, labels):
    with c:
      if st.button(label, key=f"nav_{page_name}"):
        go_to(page_name)


# =====================================================
# 3️⃣ Routing Halaman
# =====================================================

# --- HOME ---
if st.session_state["page"] == "home":
  st.title("🏥 Kalkulator Medis PICU")
  st.markdown(
      "### Selamat Datang di Aplikasi Prediksi Durasi Rawat PICU Pasien"
  )
  st.write("Silakan mulai untuk masuk ke sistem.")
  if st.button("Mulai Aplikasi"):
    go_to("login")

# --- LOGIN ---
elif st.session_state["page"] == "login":
  st.header("🔐 Login Tenaga Medis")
  with st.form("login_form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submitted = st.form_submit_button("Masuk")

  if submitted:
    user = st.session_state["users"].get(username)
    if user and user["password"] == password:
      st.session_state["logged_in"] = True
      st.session_state["username"] = username
      go_to("dashboard")
      st.rerun()
    else:
      st.error("Username atau password salah.")

  if st.button("Kembali ke Beranda"):
    go_to("home")

# --- DASHBOARD ---
elif st.session_state["page"] == "dashboard":
  if not st.session_state["logged_in"]:
    st.warning("Silakan login terlebih dahulu.")
    go_to("login")
  else:
    user = st.session_state["users"].get(st.session_state["username"], {})
    st.title("Dashboard Utama")

    dashboard_nav()
    st.markdown("---")

    if load_error:
      st.error(
          f"⚠️ Peringatan Model: {load_error}. Pastikan file .joblib dan .pkl"
          " ada di direktori utama."
      )

    st.markdown(
        f'<div class="welcome-box">Selamat datang, {user.get("name", "-")}!</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Ringkasan Profil Dokter")
    col1, col2 = st.columns([2, 3])
    with col1:
      st.write("**Nama:**", user.get("name", "-"))
      st.write("**Instansi:**", user.get("instansi", "-"))
      st.write("**Jadwal Praktek:**", user.get("jadwal", "-"))
    with col2:
      st.write("**Riwayat Diagnosis Terbaru**")
      if len(st.session_state["history"]) == 0:
        st.info("Belum ada riwayat diagnosis.")
      else:
        df_recent = pd.DataFrame(st.session_state["history"]).tail(5)
        st.dataframe(df_recent, use_container_width=True)

    st.markdown("---")
    if st.button("Log Out"):
      st.session_state["logged_in"] = False
      st.session_state["username"] = None
      go_to("login")
      st.rerun()

# --- PROFILE ---
elif st.session_state["page"] == "profile":
  if not st.session_state["logged_in"]:
    st.warning("Silakan login terlebih dahulu.")
    go_to("login")
  else:
    user = st.session_state["users"].get(st.session_state["username"], {})
    st.title("Profil Pengguna")
    dashboard_nav()
    st.markdown("---")
    st.write("**Nama:**", user.get("name", "-"))
    st.write("**Instansi:**", user.get("instansi", "-"))
    st.write("**Jadwal Praktek:**", user.get("jadwal", "-"))
    st.write("**Email:**", user.get("email", "-"))
    st.write("**No HP:**", user.get("phone", "-"))
    st.markdown("---")
    if st.button("Kembali ke Dashboard"):
      go_to("dashboard")

# --- DIAGNOSIS ---
elif st.session_state["page"] == "diagnosis":
  if not st.session_state["logged_in"]:
    st.warning("Silakan login terlebih dahulu.")
    go_to("login")
  else:
    st.title("Diagnosis - 9 Variabel Klinis Pasien")
    dashboard_nav()
    st.markdown("---")

    if model is None or selected_features is None:
      st.error(
          f"Model tidak dapat dimuat. Detail Error: {load_error}. Mohon periksa"
          " kembali file .joblib Anda di root directory."
      )
    else:
      with st.form("diagnosis_form"):
        input_data = {}

        st.markdown(
            '<label class="navy-label">STATUS GIZI</label>',
            unsafe_allow_html=True,
        )
        input_data["STATUS GIZI"] = st.selectbox(
            "", options=["Normal", "Buruk / Kurang", "Lebih / Obesitas"]
        )

        st.markdown(
            '<label class="navy-label">LOKASI TUMOR</label>',
            unsafe_allow_html=True,
        )
        input_data["LOKASI TUMOR"] = st.selectbox(
            "", options=["Infratentorial", "Supratentorial", "Lainnya"]
        )

        st.markdown(
            '<label class="navy-label">RIW. RESEKSI TUMOR</label>',
            unsafe_allow_html=True,
        )
        input_data["RIW. RESEKSI TUMOR"] = st.selectbox(
            "",
            options=[0, 1],
            format_func=lambda x: "Ya" if x == 1 else "Tidak",
        )

        st.markdown(
            '<label class="navy-label">VOLUME PERDARAHAN INTRAOP</label>',
            unsafe_allow_html=True,
        )
        input_data["VOLUME PERDARAHAN INTRAOP"] = st.number_input(
            "", min_value=0.0, value=100.0, step=10.0
        )

        st.markdown(
            '<label class="navy-label">VOLUME TRANSFUSI INTRAOP</label>',
            unsafe_allow_html=True,
        )
        input_data["VOLUME TRANSFUSI INTRAOP"] = st.number_input(
            "", min_value=0.0, value=0.0, step=10.0
        )

        st.markdown(
            '<label class="navy-label">STATUS INTUBASI</label>',
            unsafe_allow_html=True,
        )
        input_data["STATUS INTUBASI"] = st.selectbox(
            "",
            options=[0, 1],
            format_func=lambda x: "Terintubasi" if x == 1 else "Tidak",
        )

        st.markdown(
            '<label class="navy-label">STATUS VP EVD</label>',
            unsafe_allow_html=True,
        )
        input_data["STATUS VP EVD"] = st.selectbox(
            "",
            options=[0, 1],
            format_func=lambda x: "Ya" if x == 1 else "Tidak",
        )

        st.markdown(
            '<label class="navy-label">KOMP. AKUT PASCA OP</label>',
            unsafe_allow_html=True,
        )
        input_data["KOMP. AKUT PASCA OP"] = st.selectbox(
            "",
            options=[0, 1],
            format_func=lambda x: "Ada" if x == 1 else "Tidak Ada",
        )

        st.markdown(
            '<label class="navy-label">STATUS HEMODINAMIK</label>',
            unsafe_allow_html=True,
        )
        input_data["STATUS HEMODINAMIK"] = st.selectbox(
            "", options=["Stabil", "Tidak Stabil"]
        )

        submitted = st.form_submit_button("Diagnosis")

      if submitted:
        input_df = pd.DataFrame([input_data])
        # Menyesuaikan kolom dengan fitur yang diharapkan model
        for col in selected_features:
          if col not in input_df.columns:
            input_df[col] = 0
        input_df = input_df[selected_features]

        try:
          prob = model.predict_proba(input_df)[0, 1]
          prediction_label = ">24 jam" if prob >= 0.5 else "<24 jam"

          st.markdown("---")
          st.subheader("📊 Hasil Prediksi")
          st.success(
              f"**Lama waktu rawat PICU:** {prediction_label} (Probabilitas:"
              f" {prob * 100:.2f}%)"
          )

          # Simpan ke riwayat otomatis
          record = {
              "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              **input_data,
              "Hasil Prediksi": prediction_label,
              "Probabilitas": f"{prob * 100:.2f}%",
          }
          st.session_state["history"].append(record)

          col_a, col_b = st.columns(2)
          with col_a:
            if st.button("Lakukan Diagnosis Lain"):
              st.rerun()
          with col_b:
            if st.button("Lihat Riwayat Diagnosis"):
              go_to("history")
              st.rerun()

        except Exception as e:
          st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

# --- RIWAYAT DIAGNOSIS ---
elif st.session_state["page"] == "history":
  if not st.session_state["logged_in"]:
    st.warning("Silakan login terlebih dahulu.")
    go_to("login")
  else:
    st.title("📋 Riwayat Diagnosis Pasien")
    dashboard_nav()
    st.markdown("---")

    if len(st.session_state["history"]) == 0:
      st.info("Belum ada riwayat diagnosis tersimpan.")
    else:
      df_hist = pd.DataFrame(st.session_state["history"])
      st.dataframe(df_hist, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
      if st.button("Lakukan Diagnosis Lain"):
        go_to("diagnosis")
        st.rerun()
    with col2:
      if st.button("Logout"):
        st.session_state["logged_in"] = False
        go_to("login")
        st.rerun()

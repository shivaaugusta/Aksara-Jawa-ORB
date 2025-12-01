import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# =============================================
# CONFIG
# =============================================
st.set_page_config(page_title="Aksara Jawa Classifier", layout="wide")
st.title("🔠 Aksara Jawa Classifier (Evaluasi)")

# Dummy label mapping (ganti dengan mapping asli)
ID_TO_LABEL = {i: f"Class {i}" for i in range(20)}

# Dummy ORB index (biar tidak error sebelum kamu load yg asli)
ORB_INDEX = np.random.rand(20, 50, 32)  # simulasi feature index


# =============================================
# 🔧 DUMMY FUNCTIONS (ganti sama yang asli nanti)
# =============================================
def preprocess_image(pil_img):
    """Simulasi preprocessing ORB"""
    return pil_img.convert("L")  # grayscale saja untuk sementara


def extract_orb(img):
    """Simulasi ekstraksi ORB"""
    return np.random.rand(32, 32)  # random feature untuk testing


def predict_ratio(des_query, index, ratio, top_k):
    """Simulasi Top-K prediction"""
    scores = np.random.rand(20)
    best_label = scores.argmax()
    top_matches = [{"label_id": i, "score": float(scores[i])} for i in range(top_k)]
    return ID_TO_LABEL[best_label], top_matches


# =============================================
# 🖥️ UI LAYOUT – 2 COLUMNS
# =============================================
col_left, col_right = st.columns([1, 2])

# ---------------- LEFT PANEL -----------------
with col_left:
    st.subheader("Upload Query Image")
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

    st.markdown("---")
    st.subheader("Pengaturan Pencocokan")

    lowe_ratio = st.slider("Lowe Ratio", 0.1, 1.0, 0.75, 0.01)
    top_k = st.slider("Top-K", 1, 20, 5, 1)
    thresh = st.slider("Unknown Threshold", 0.01, 0.5, 0.05, 0.01)

    submitted = st.button("📌 Submit")


# ---------------- RIGHT PANEL -----------------
with col_right:
    st.subheader("Results")

    if submitted:

        if uploaded_file is None:
            st.warning("⚠️ Silakan upload gambar dulu sebelum Submit!")
            st.stop()

        pil_img = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Query Preview**")
            st.image(pil_img, width=250)

        # PREPROCESS
        preproc = preprocess_image(pil_img)
        with col2:
            st.markdown("**Processed Image**")
            st.image(preproc, width=250)

        st.markdown("---")

        # ORB FEATURE
        des_query = extract_orb(preproc)

        # RUN PREDIKSI
        pred, top_matches = predict_ratio(des_query, ORB_INDEX, lowe_ratio, top_k)
        st.success(f"**Prediksi: {pred}**")

        # SHOW TOP-K TABLE
        df = pd.DataFrame(top_matches)
        df["Label"] = df["label_id"].map(ID_TO_LABEL)
        df = df[["Label", "score"]]
        df.columns = ["Label", "Good Matches"]

        st.markdown("**Top Matching Results**")
        st.table(df)

        st.caption("📊 *Confusion Matrix dan Metrics ada di halaman Evaluasi Model*")


import streamlit as st
import pandas as pd
import numpy as np

# ===== Page Config ===== #
st.set_page_config(page_title="Aksara Jawa Classifier", layout="wide")

# ===== Custom CSS ===== #
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }
        h2 {
            font-size: 1.6rem !important;
        }
        .stButton>button {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# ===== Label Map ===== #
LABEL_MAP = {i: f"Class {i}" for i in range(20)}

# ===== Title ===== #
st.markdown(
    "<h2 style='text-align: center; margin-bottom: 20px;'>🅰️🅱️🅲️🅳️ Aksara Jawa Classifier (Evaluasi)</h2>",
    unsafe_allow_html=True
)

# ===== Layout Upload & Result Columns ===== #
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Upload Query Image")
    query_img = st.file_uploader("Drag or select image file", type=["png", "jpg", "jpeg"])
    
    st.write("---")
    st.subheader("Pengaturan Pencocokan")
    lowe_ratio = st.slider("Lowe Ratio", 0.1, 1.0, 0.75, 0.05)
    top_k = st.slider("Top-K", 1, 20, 5)

with col_right:
    st.subheader("Results ↩")
    st.info("Belum ada hasil. Silakan upload gambar terlebih dahulu.")

# ===== Evaluasi Button with Modal Dialog ===== #
@st.dialog("📊 Evaluasi Model", width="large")
def show_evaluation():
    st.subheader("Confusion Matrix")
    cm_data = np.zeros((20,20))
    st.dataframe(pd.DataFrame(cm_data, columns=LABEL_MAP.values()))

    st.subheader("📈 Model Metrics")
    st.dataframe(pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1-Score"],
        "Value": ["33%", "33%", "32.5%"]
    }))

st.write("---")
st.button("📊 Lihat Evaluasi Model", on_click=show_evaluation)

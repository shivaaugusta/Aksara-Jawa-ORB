# app.py (Final Deployment Version with Dynamic Sliders)

import streamlit as st
import cv2
import numpy as np
import pickle
import json
import os
import gzip 
import pandas as pd
from PIL import Image

# --- 1. KONFIGURASI DAN LOAD MODEL ---
INDEX_FILE = "orb_index.pkl.gz" 
LABEL_FILE = "label_map.json"
# Nilai default (akan ditimpa oleh slider)
ORB_N_FEATURES = 250
ACCURACY_REPORTED = 32.89 

# Load model dan label saat aplikasi dimulai
@st.cache_resource
def load_resources():
    try:
        with gzip.open(INDEX_FILE, "rb") as f: 
            orb_index = pickle.load(f)
        with open(LABEL_FILE, "r") as f:
            label_map = json.load(f)
        orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES)
        bf_knn = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        id_to_label = {v: k for k, v in label_map.items()}
        return orb_index, label_map, id_to_label, orb, bf_knn
    except FileNotFoundError:
        return None, None, None, None, None
    except Exception as e:
        return None, None, None, None, None

ORB_INDEX, LABEL_MAP, ID_TO_LABEL, ORB, BF_KNN = load_resources()

# --- 2. UTILITY FUNCTIONS (Tetap Sama) ---

def pil_to_cv2_gray(pil_img):
    rgb_img = np.array(pil_img.convert('RGB'))[:, :, ::-1]
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.uint8)

def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    if len(coords) < 10: return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle = -(90 + angle)
    else: angle = -angle
    (h, w) = image.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_image(pil_img):
    img = pil_to_cv2_gray(pil_img)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    deskewed = deskew(th)
    final = cv2.resize(deskewed, (256, 256))
    return final

def extract_orb(image):
    edges = cv2.Canny(image, 50, 150)
    kp, des = ORB.detectAndCompute(edges, None)
    if des is None: return None
    return des.astype(np.uint8)

# PREDICT FUNCTION SEKARANG MEMBUTUHKAN RATIO DAN K SEBAGAI ARGUMEN
def predict_ratio(des_query, index, ratio_thresh, top_k_count):
    all_scores = []
    
    for des_train, label_id in index:
        try:
            matches = BF_KNN.knnMatch(des_query, des_train, k=2)
            good_matches = 0
            for pair in matches:
                if len(pair) < 2: continue
                m, n = pair[0], pair[1]
                # Menggunakan ratio_thresh dari slider
                if m.distance < ratio_thresh * n.distance: 
                    good_matches += 1
            all_scores.append({"score": good_matches, "label_id": label_id})
        except:
            continue

    if not all_scores: return None, []

    # 1. Ambil Top Match Rank 1 (Skor Tertinggi)
    top_results = sorted(all_scores, key=lambda x: x["score"], reverse=True)
    
    predicted_label_id = top_results[0]["label_id"]
    final_prediction = ID_TO_LABEL[predicted_label_id]
    
    # Ambil Top-K dari slider
    top_k_results = top_results[:top_k_count] 
    
    return final_prediction, top_k_results 

# --- 3. APLIKASI STREAMLIT UTAMA ---

st.set_page_config(page_title="Identifikasi Aksara Jawa (ORB-Canny)", layout="wide")

st.title("🔠 Identifikasi Aksara Jawa (Metode ORB)")
st.caption(f"Akurasi Test: {ACCURACY_REPORTED:.2f}%. Model menggunakan {ORB_N_FEATURES} fitur ORB dengan Rasio Lowe.")

# --- Bagian Kiri: Pengaturan & Upload ---
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Upload Query Image")
    uploaded_file = st.file_uploader("Unggah gambar Aksara Jawa (.png, .jpg)", type=["png", "jpg", "jpeg"])

    st.markdown("---")
    st.subheader("Pengaturan Pencocokan")

    # SLIDER LOWE RATIO
    lowe_ratio = st.slider(
        "Lowe ratio", 
        min_value=0.1, max_value=1.0, value=0.75, step=0.01,
        help="Semakin kecil nilai, semakin ketat kriteria pencocokan."
    )
    
    # SLIDER TOP-K
    top_k = st.slider(
        "Top-K", 
        min_value=1, max_value=20, value=5, step=1,
        help="Jumlah hasil teratas (Top Matches) yang akan ditampilkan."
    )

    # UNKNOWN THRESHOLD (Placeholder)
    unknown_threshold = st.slider(
        "Unknown threshold", 
        min_value=0.01, max_value=1.0, value=0.05, step=0.01,
        help="Ambang batas untuk klasifikasi 'Unknown' (Tidak digunakan di prediksi ini)."
    )
    
    st.button("Submit") # Submit button

# --- Bagian Kanan: Results dan Visualisasi ---
with col_right:
    st.subheader("Results")
    st.markdown("---")
    
    if uploaded_file is not None:
        if ORB_INDEX is None:
            st.error("🚨 Model tidak berhasil dimuat! Pastikan orb_index.pkl.gz terunggah.")
            st.stop()

        try:
            pil_img = Image.open(uploaded_file)
            preprocessed_cv = preprocess_image(pil_img)
            des_query = extract_orb(preprocessed_cv)

            # Tampilan Hasil (2 Kolom: Query Preview & Visualisasi)
            col_preview, col_proc = st.columns([1, 1])
            
            with col_preview:
                st.markdown("**Query Preview**")
                st.image(pil_img, use_column_width=True)

            with col_proc:
                st.markdown("**Visualisasi Preprocessing**")
                st.image(preprocessed_cv, caption="Threshold + Deskew + Resize", use_column_width=True)

            if des_query is not None and len(des_query) > 0:
                # Panggil prediksi dengan parameter dari slider
                final_prediction, top_matches = predict_ratio(des_query, ORB_INDEX, lowe_ratio, top_k) 
                
                st.markdown("---")
                st.success(f"**Predicted label:** {final_prediction.upper()}")
                st.info(f"Ditemukan {len(des_query)} deskriptor ORB.")

                # --- TAMPILAN TOP MATCHES DETAIL/THUMBNAILS ---
                st.subheader("Top Matches Detail")
                
                # Mengubah Top Matches menjadi DataFrame dan menampilkannya
                df = pd.DataFrame(top_matches)
                df['label'] = df['label_id'].apply(lambda x: ID_TO_LABEL[x])
                df = df.drop(columns=['label_id']).rename(columns={'score': 'good_matches', 'label': 'label'})
                df.index += 1
                df.index.name = 'Rank'
                
                # Menampilkan tabel detail
                st.dataframe(df)

            else:
                 st.warning("⚠️ Gagal mengekstrak fitur ORB.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

st.markdown("---")
st.caption("Proyek ini didasarkan pada metode ORB (32.89% akurasi).")

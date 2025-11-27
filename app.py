# app.py (Final Deployment Version with Metrik Cleaned Up)

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
ORB_N_FEATURES = 250
RATIO_THRESH = 0.75
ACCURACY_REPORTED = 32.89 # AKURASI TEST FINAL ANDA

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

# --- 2. UTILITY FUNCTIONS (Disesuaikan dari Notebook) ---

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

def predict_ratio(des_query, index):
    scores = {}
    all_scores = []
    for des_train, label_id in index:
        try:
            matches = BF_KNN.knnMatch(des_query, des_train, k=2)
            good_matches = 0
            for pair in matches:
                if len(pair) < 2: continue
                m, n = pair[0], pair[1]
                if m.distance < RATIO_THRESH * n.distance:
                    good_matches += 1
            scores[label_id] = scores.get(label_id, 0) + good_matches
            all_scores.append({"score": good_matches, "label_id": label_id})
        except:
            continue
    if not scores: return None, []
    predicted_label_id = max(scores.items(), key=lambda x: x[1])[0]
    final_prediction = ID_TO_LABEL[predicted_label_id]
    top_results = sorted(all_scores, key=lambda x: x["score"], reverse=True)[:5]
    return final_prediction, top_results 

# --- 3. APLIKASI STREAMLIT UTAMA ---

st.set_page_config(page_title="Identifikasi Aksara Jawa (ORB-Canny)", layout="wide")

st.title("🔠 Identifikasi Aksara Jawa (Metode ORB)")
st.caption(f"Akurasi Test: **{ACCURACY_REPORTED:.2f}%**. Model menggunakan {ORB_N_FEATURES} fitur ORB dengan Rasio Lowe.")

# --- Bagian Tombol Evaluasi (Meniru Tampilan Dosen) ---
col_buttons = st.columns(3)
col_buttons[0].button("Build Index", disabled=True, help="Index sudah dimuat di memori.")
col_buttons[1].button("Evaluate Dataset (full)", disabled=True, help="Evaluasi penuh dilakukan offline. Akurasi: 32.89%")
col_buttons[2].button("Load Sample", disabled=True, help="Fungsi sampel tidak diimplementasikan")
st.markdown("---")

# --- Bagian Utama Prediksi Gambar ---
uploaded_file = st.file_uploader("Unggah gambar Aksara Jawa (.png, .jpg)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    if ORB_INDEX is None:
        st.error("🚨 Model tidak berhasil dimuat! Pastikan orb_index.pkl.gz dan label_map.json terunggah.")
        st.stop()
        
    try:
        pil_img = Image.open(uploaded_file)
        
        # Tampilan 2 kolom untuk Query dan Hasil
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown("### Gambar Asli")
            st.image(pil_img, caption="Query Preview", use_column_width=True)
            
            preprocessed_cv = preprocess_image(pil_img)
            des_query = extract_orb(preprocessed_cv)

            if des_query is not None and len(des_query) > 0:
                final_prediction, top_matches = predict_ratio(des_query, ORB_INDEX) 
                
                # OUTPUT UTAMA (MIRIP TAMPILAN DOSEN)
                st.success(f"**Predicted label:** {final_prediction.upper()}")
                st.info(f"Ditemukan {len(des_query)} deskriptor ORB.")

                # --- TAMPILAN TOP MATCHES DETAIL ---
                st.subheader("Top Matches Detail")
                
                df = pd.DataFrame(top_matches)
                df['label'] = df['label_id'].apply(lambda x: ID_TO_LABEL[x])
                df = df.drop(columns=['label_id']).rename(columns={'score': 'good_matches', 'label': 'label'})
                df.index += 1
                df.index.name = 'Rank'
                
                st.dataframe(df)
                # --- AKHIR TABEL ---
                
            else:
                 st.warning("⚠️ Gagal mengekstrak fitur ORB. Gambar mungkin terlalu polos atau gelap.")

        with col2:
            st.subheader("Visualisasi Preprocessing")
            st.image(preprocessed_cv, caption="Gambar Setelah Preprocessing (Threshold + Deskew + Resize)", use_column_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
        st.markdown(f"Detail Error: {e}")

st.markdown("---")
st.markdown("Proyek ini menggunakan fitur ORB untuk mencocokkan aksara. Jika akurasi rendah, ini adalah batasan metode fitur lokal.")

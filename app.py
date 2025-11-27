# app.py (Final Deployment Version with Gzip Support and Pandas Table)

import streamlit as st
import cv2
import numpy as np
import pickle
import json
import os
import gzip # Diperlukan untuk membaca file terkompresi
import pandas as pd # Diperlukan untuk membuat tabel Top Matches
from PIL import Image

# --- 1. KONFIGURASI DAN LOAD MODEL ---
INDEX_FILE = "orb_index.pkl.gz" 
LABEL_FILE = "label_map.json"
ORB_N_FEATURES = 250
RATIO_THRESH = 0.75

# Load model dan label saat aplikasi dimulai
@st.cache_resource
def load_resources():
    try:
        # MEMUAT FILE TERKOMPRESI MENGGUNAKAN GZIP
        with gzip.open(INDEX_FILE, "rb") as f: 
            orb_index = pickle.load(f)
            
        with open(LABEL_FILE, "r") as f:
            label_map = json.load(f)

        # Inisialisasi ORB dan Matcher
        orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES)
        bf_knn = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Inversi label map (dari ID ke Nama Aksara)
        id_to_label = {v: k for k, v in label_map.items()}

        return orb_index, label_map, id_to_label, orb, bf_knn
    except FileNotFoundError:
        return None, None, None, None, None
    except Exception as e:
        return None, None, None, None, None

ORB_INDEX, LABEL_MAP, ID_TO_LABEL, ORB, BF_KNN = load_resources()

# --- 2. UTILITY FUNCTIONS (Disesuaikan dari Notebook) ---

def pil_to_cv2_gray(pil_img):
    # Mengubah gambar PIL menjadi Grayscale OpenCV
    rgb_img = np.array(pil_img.convert('RGB'))[:, :, ::-1]
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.uint8)

def deskew(image):
    # Fungsi Deskew 
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
    # Melakukan Preprocessing lengkap
    img = pil_to_cv2_gray(pil_img)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    deskewed = deskew(th)
    final = cv2.resize(deskewed, (256, 256))
    return final

def extract_orb(image):
    # Ekstraksi ORB dengan Canny Boosted
    edges = cv2.Canny(image, 50, 150)
    kp, des = ORB.detectAndCompute(edges, None)
    if des is None: return None
    return des.astype(np.uint8)

def predict_ratio(des_query, index):
    # Fungsi Prediksi menggunakan Rasio Lowe (Mengembalikan Prediksi Final dan Top Matches)
    all_scores = []
    scores_by_label = {} 

    for des_train, label_id in index:
        try:
            matches = BF_KNN.knnMatch(des_query, des_train, k=2)
            good_matches = 0
            
            for pair in matches:
                if len(pair) < 2: continue
                m, n = pair[0], pair[1]
                if m.distance < RATIO_THRESH * n.distance:
                    good_matches += 1
            
            score = good_matches
            
            # Simpan skor dan label untuk setiap gambar training (untuk Top Matches Detail)
            all_scores.append({"score": score, "label_id": label_id})
            
            # Tambahkan untuk voting final
            scores_by_label[label_id] = scores_by_label.get(label_id, 0) + score
        except:
            continue

    if not scores_by_label: 
        return None, []

    # 1. Prediksi Final
    predicted_label_id = max(scores_by_label.items(), key=lambda x: x[1])[0]
    final_prediction = ID_TO_LABEL[predicted_label_id]
    
    # 2. Ambil Top 5 Scores (Mirip tampilan dosen)
    top_results = sorted(all_scores, key=lambda x: x["score"], reverse=True)[:5]
    
    return final_prediction, top_results # Mengembalikan dua nilai

# --- 3. APLIKASI STREAMLIT UTAMA ---
st.set_page_config(page_title="Identifikasi Aksara Jawa (ORB-Canny)", layout="wide")

st.title("🔠 Identifikasi Aksara Jawa (Metode ORB)")
st.caption(f"Akurasi Test: {round(32.89, 2)}%. Model menggunakan {ORB_N_FEATURES} fitur ORB dengan Rasio Lowe.")

uploaded_file = st.file_uploader("Unggah gambar Aksara Jawa (.png, .jpg)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Cek apakah model berhasil dimuat (Jika ORB_INDEX adalah None, tampilkan error)
    if ORB_INDEX is None:
        st.error("🚨 Model tidak berhasil dimuat! Pastikan orb_index.pkl.gz dan label_map.json terunggah.")
        st.stop()
        
    try:
        pil_img = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.image(pil_img, caption="Gambar Asli", use_column_width=True)
            
            # Melakukan Preprocessing dan Prediksi
            preprocessed_cv = preprocess_image(pil_img)
            des_query = extract_orb(preprocessed_cv)

            if des_query is not None and len(des_query) > 0:
                # Memanggil fungsi yang mengembalikan Prediksi dan Top Matches
                prediction_label, top_matches = predict_ratio(des_query, ORB_INDEX) 
                
                st.success(f"### ➡️ Hasil Prediksi: **{prediction_label.upper()}**")
                st.info(f"Ditemukan {len(des_query)} deskriptor ORB.")

                # --- MENAMPILKAN TABEL TOP MATCHES ---
                st.subheader("Top Matches Detail (Similarity)")
                
                # Konversi hasil ke DataFrame untuk tampilan Streamlit yang rapi
                df = pd.DataFrame(top_matches)
                df['label'] = df['label_id'].apply(lambda x: ID_TO_LABEL[x])
                
                # Memformat untuk tampilan yang mirip interface dosen
                df = df[['label', 'score']].rename(columns={'score': 'Good Matches', 'label': 'Predicted Label'})
                df['Good Matches'] = df['Good Matches'].astype(int)
                df.index += 1 # Mulai dari Rank 1
                df.index.name = 'Rank'
                
                st.dataframe(df) # Tampilkan tabel
                # --- AKHIR TABEL ---
                
            else:
                 st.warning("⚠️ Gagal mengekstrak fitur ORB. Gambar mungkin terlalu polos atau gelap.")

        with col2:
            st.subheader("Visualisasi Preprocessing (256x256)")
            # Tampilkan gambar yang sudah di-preprocess
            st.image(preprocessed_cv, caption="Gambar Setelah Preprocessing (Threshold + Deskew + Resize)", use_column_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

st.markdown("---")
st.markdown("Proyek ini menggunakan fitur ORB untuk mencocokkan aksara. Jika akurasi rendah, ini adalah batasan metode fitur lokal.")

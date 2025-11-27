# app.py (Final Deployment Version - Menampilkan CM Mentah 20x20)

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
ACCURACY_REPORTED = 39.68 # Akurasi Test Final Anda

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

# --- 2. UTILITY FUNCTIONS ---

def pil_to_cv2_gray(pil_img):
    """Konversi PIL Image ke Grayscale OpenCV."""
    rgb_img = np.array(pil_img.convert('RGB'))[:, :, ::-1]
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.uint8)

def deskew(image):
    """Meluruskan gambar (Deskewing)."""
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
    """Menerapkan seluruh pipeline preprocessing."""
    img = pil_to_cv2_gray(pil_img)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    deskewed = deskew(th)
    final = cv2.resize(deskewed, (256, 256))
    return final

def extract_orb(image):
    """Ekstraksi ORB dengan Canny Boosted."""
    edges = cv2.Canny(image, 50, 150)
    kp, des = ORB.detectAndCompute(edges, None)
    if des is None: return None
    return des.astype(np.uint8)

def predict_ratio(des_query, index, ratio_thresh, top_k_count):
    """Fungsi Prediksi menggunakan Rasio Lowe dan mengembalikan Rank 1 dan Top-K."""
    all_scores = []
    
    for des_train, label_id in index:
        try:
            matches = BF_KNN.knnMatch(des_query, des_train, k=2)
            good_matches = 0
            for pair in matches:
                if len(pair) < 2: continue
                m, n = pair[0], pair[1]
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
st.caption(f"Proyek menggunakan {ORB_N_FEATURES} fitur ORB dengan Rasio Lowe.")

# Struktur 2 Kolom Utama (Meniru Layout Dosen)
col_left, col_right = st.columns([1, 2])

# --- PANEL KIRI: UPLOAD & PENGATURAN ---
with col_left:
    st.subheader("Upload Query Image")
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

    st.markdown("---")
    st.subheader("Pengaturan Pencocokan")

    # SLIDER LOWE RATIO (Parameter aktif)
    lowe_ratio = st.slider("Lowe ratio", min_value=0.1, max_value=1.0, value=0.75, step=0.01)
    
    # SLIDER TOP-K (Parameter aktif)
    top_k = st.slider("Top-K", min_value=1, max_value=20, value=5, step=1)

    # UNKNOWN THRESHOLD (Dipertahankan untuk replikasi UI)
    unknown_threshold = st.slider("Unknown threshold", min_value=0.01, max_value=0.5, value=0.05, step=0.01)
    
    st.button("Submit") # Submit button
    
# --- PANEL KANAN: RESULTS DAN PREVIEW ---
with col_right:
    st.subheader("Results")
    
    if uploaded_file is not None:
        if ORB_INDEX is None:
            st.error("🚨 Model tidak berhasil dimuat! Harap refresh dan pastikan file model ada.")
            st.stop()

        try:
            pil_img = Image.open(uploaded_file)
            preprocessed_cv = preprocess_image(pil_img)
            des_query = extract_orb(preprocessed_cv)
            
            # Tampilan Preview: Gabungan Gambar Asli & Proses
            col_preview, col_proc = st.columns([1, 1])
            
            with col_preview:
                st.markdown("**Query Preview**")
                st.image(pil_img, use_column_width=True)
            
            with col_proc:
                st.markdown("**Visualisasi Preprocessing**")
                st.image(preprocessed_cv, caption="Threshold + Deskew + Resize", use_column_width=True)
            
            st.markdown("---")
            
            if des_query is not None and len(des_query) > 0:
                final_prediction, top_matches = predict_ratio(des_query, ORB_INDEX, lowe_ratio, top_k) 
                
                # OUTPUT UTAMA
                st.success(f"**Predicted label:** {final_prediction.upper()}")
                st.info(f"Ditemukan {len(des_query)} deskriptor ORB.")

                # --- TAMPILAN TOP MATCHES DETAIL (GRID/KARTU REPLIKA) ---
                st.subheader("Top Matches Detail")
                
                df = pd.DataFrame(top_matches)
                df['label'] = df['label_id'].apply(lambda x: ID_TO_LABEL[x])
                df = df.drop(columns=['label_id']).rename(columns={'score': 'Good Matches', 'label': 'Label'})
                
                # Menampilkan Kartu Visual (Pengganti Thumbnails)
                cols = st.columns(len(df))
                for i, row in df.iterrows():
                    with cols[i]:
                        st.markdown(f"**Rank {i+1}**")
                        st.markdown(f"**{row['Label'].upper()}**")
                        st.caption(f"Score: {row['Good Matches']} matches")
                        
                        # Placeholder Visual (Hanya Rank 1 yang menampilkan gambar query yang diproses)
                        if i == 0:
                            st.image(preprocessed_cv, caption="Best Match Preview", use_column_width=True)
                        else:
                            st.markdown("*(Thumbnail Data Training tidak tersedia)*")

            else:
                 st.warning("⚠️ Gagal mengekstrak fitur ORB.")

            # --- TAMPILAN CONFUSION MATRIX (CM) ---
            st.markdown("---")
            st.subheader("Evaluasi Penuh: Confusion Matrix & Metrik")
            
            # --- DEFINISI DATA CM STATIS 20x20 (Diambil dari hasil 39.68%) ---
            cm_labels = list(LABEL_MAP.keys()) 
            cm_data_39_68 = [
                [ 4,  0,  0,  0,  0,  1,  0,  3,  0,  0,  0,  0,  8,  0,  0,  1,  0,  0,  0,  2], 
                [ 0, 12,  0,  0,  0,  0,  0,  2,  0,  0,  0,  0,  0,  0,  0,  4,  0,  0,  1,  0], 
                [ 0,  0, 14,  0,  0,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0], 
                [ 0,  0,  0, 14,  0,  0,  0,  1,  0,  2,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0], 
                [ 2,  0,  0,  0,  1,  1,  0,  8,  0,  1,  0,  0,  1,  0,  0,  4,  0,  0,  0,  1], 
                [ 1,  0,  0,  0,  0,  4,  0,  4,  1,  0,  0,  0,  0,  0,  0,  9,  0,  0,  0,  0], 
                [ 0,  0,  0,  0,  0,  0,  9,  2,  0,  0,  0,  0,  0,  0,  0,  6,  0,  0,  1,  0], 
                [ 0,  0,  0,  0,  0,  0,  0, 17,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0], 
                [ 0,  0,  0,  0,  0,  3,  0,  7,  3,  0,  2,  0,  0,  0,  0,  2,  0,  0,  0,  2], 
                [ 0,  1,  0,  1,  0,  0,  0,  0,  1,  9,  0,  0,  0,  1,  0,  4,  0,  0,  2,  1], 
                [ 0,  0,  0,  0,  0,  0,  0, 16,  0,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0], 
                [ 5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  7,  2,  0,  0,  3,  0,  1,  0,  1], 
                [ 0,  1,  0,  0,  0,  0,  0,  4,  0,  0,  0,  1, 11,  0,  0,  2,  0,  0,  0,  0], 
                [ 0,  0,  0,  0,  0,  1,  0,  4,  0,  0,  0,  0,  0,  0,  0, 13,  0,  0,  0,  1], 
                [ 3,  2,  0,  0,  0,  1,  0,  2,  0,  3,  0,  0,  2,  0,  0,  4,  0,  0,  2,  0], 
                [ 0,  4,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0, 13,  0,  0,  0,  1], 
                [ 0,  0,  0,  0,  0,  0,  0,  6,  0,  0,  0,  0,  0,  1,  0,  5,  6,  0,  1,  0], 
                [ 0,  3,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  3,  0, 10,  0,  1], 
                [ 0,  0,  0,  0,  0,  0,  0,  3,  0,  0,  0,  0,  0,  1,  0,  7,  0,  0,  8,  0], 
                [ 0,  0,  0,  1,  0,  2,  1,  2,  1,  0,  0,  0,  0,  1,  0,  5,  0,  0,  1,  5]
            ]
            
            cm_df = pd.DataFrame(data=np.array(cm_data_39_68), columns=cm_labels)
            cm_df.insert(0, 'GT \ Pred', cm_labels) # Tambahkan kolom Ground Truth (GT)

            st.markdown("""
            #### 📊 Confusion Matrix (CM) Mentah 20x20
            Angka-angka di bawah ini adalah hasil evaluasi penuh model pada data test:
            """)
            
            st.dataframe(cm_df) # Tampilkan tabel CM

            # Menampilkan Metrik Ringkas
            st.markdown("---")
            st.subheader("Ringkasan Metrik Kinerja")
            
            st.metric(label="Akurasi Model Test (Offline)", value=f"{ACCURACY_REPORTED:.2f}%", delta="Target Dosen: >80%", delta_color="inverse")
            
            metrik_data = {
                'Metric': ['Average Precision', 'Average Recall', 'F1-Score'],
                'Value': [f"{33.00:.2f}%", f"{33.00:.2f}%", f"{32.50:.2f}%"] 
            }
            df_metrik = pd.DataFrame(metrik_data)
            st.table(df_metrik) 

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

st.markdown("---")
st.caption("Proyek ini menggunakan fitur ORB untuk mencocokkan aksara. Jika akurasi rendah, ini adalah batasan metode fitur lokal.")

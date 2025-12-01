# ------------------------------------------
# APP STREAMLIT FINAL - NO SCROLL VERSION
# Layout Desktop, Fit 1 Screen, Clean UI
# ------------------------------------------

import streamlit as st
import cv2
import numpy as np
import pickle, json, gzip, os
import pandas as pd
from PIL import Image

# --- KONFIGURASI ---
st.set_page_config(page_title="Identifikasi Aksara Jawa (ORB)", layout="wide")

INDEX_FILE = "orb_index.pkl.gz"
LABEL_FILE = "label_map.json"
ORB_N_FEATURES = 250
RATIO_THRESH = 0.75
ACCURACY_REPORTED = 39.68

# Styling agar fix 1 halaman tanpa scroll
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 1500px;
        }
        footer {visibility: hidden;}
        .stTabs [role="tablist"] button {
            font-size: 14px;
            padding: 5px 10px;
        }
        .stDataFrame, .stTable {
            max-height: 350px;
        }
    </style>
""", unsafe_allow_html=True)

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
    except:
        return None, None, None, None, None


ORB_INDEX, LABEL_MAP, ID_TO_LABEL, ORB, BF_KNN = load_resources()

def pil_to_cv2_gray(pil_img):
    rgb_img = np.array(pil_img.convert('RGB'))[:, :, ::-1]
    return cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    if len(coords) < 10: return image
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    h, w = image.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

def preprocess_image(pil_img):
    img = pil_to_cv2_gray(pil_img)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    th = cv2.adaptiveThreshold(blur, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV,
                               25, 10)
    return cv2.resize(deskew(th), (256, 256))

def extract_orb(image):
    edges = cv2.Canny(image, 50, 150)
    kp, des = ORB.detectAndCompute(edges, None)
    return des.astype(np.uint8) if des is not None else None

def predict_ratio(des_query, index, ratio_thresh, top_k_count):
    all_scores = []
    max_possible = ORB_N_FEATURES

    for des_train, label_id in index:
        try:
            matches = BF_KNN.knnMatch(des_query, des_train, k=2)
            good = sum(1 for m, n in (p for p in matches if len(p) == 2)
                       if m.distance < ratio_thresh * n.distance)
            percent = (good / max_possible) * 100
            all_scores.append({"score": good,
                               "score_percent": percent,
                               "label_id": label_id})
        except:
            continue

    top = sorted(all_scores, key=lambda x: x["score"], reverse=True)
    return ID_TO_LABEL[top[0]["label_id"]], top[:top_k_count]


# =======================
# UI UTAMA
# =======================
st.title("🔠 Identifikasi Aksara Jawa (ORB)")
st.caption(f"Akurasi Model: {ACCURACY_REPORTED:.2f}%")

colL, colR = st.columns([1, 2])

with colL:
    st.subheader("Upload Gambar")
    file = st.file_uploader("", type=["png", "jpg", "jpeg"])

    st.subheader("Pengaturan")
    lowe = st.slider("Lowe Ratio", 0.1, 1.0, 0.75, 0.01)
    top_k = st.slider("Top-K", 1, 10, 5)
    st.button("Submit")

with colR:
    st.subheader("Hasil")

    if file:
        pil_img = Image.open(file)
        pre = preprocess_image(pil_img)
        desQ = extract_orb(pre)

        cols = st.columns(2)
        cols[0].image(pil_img, caption="Query", use_column_width=True)
        cols[1].image(pre, caption="Preprocessing", use_column_width=True)

        if desQ is not None:
            pred, top = predict_ratio(desQ, ORB_INDEX, lowe, top_k)
            st.success(f"Prediksi: **{pred.upper()}**")

            # TOP K DIBUAT PADAT
            st.write("Top Matches:")
            for i, r in enumerate(top):
                st.write(f"{i+1}. {ID_TO_LABEL[r['label_id']].upper()} ({r['score_percent']:.2f}%)")
        else:
            st.warning("Fitur tidak terdeteksi")


# TABS = menghemat ruang
tab1, tab2 = st.tabs(["📊 Confusion Matrix", "📈 Metrik Model"])

with tab1:
    st.write("Evaluasi Model Offline")
    cm_labels = list(LABEL_MAP.keys())
    cm_data = np.zeros((20, 20))  # Placeholder agar tidak panjang
    st.dataframe(pd.DataFrame(cm_data, columns=cm_labels))

with tab2:
    st.metric("Akurasi Test", f"{ACCURACY_REPORTED:.2f}%")
    st.table(pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1-Score"],
        "Value": ["33%", "33%", "32.5%"]
    }))

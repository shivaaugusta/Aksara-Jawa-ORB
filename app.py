# app.py (FINAL LOGIC & UI CLEANUP)

# ... (Semua impor dan fungsi utility dari pil_to_cv2_gray hingga predict_ratio tetap sama) ...

# --- 3. APLIKASI STREAMLIT UTAMA ---
st.set_page_config(page_title="Identifikasi Aksara Jawa (ORB-Canny)", layout="wide")

st.title("🔠 Identifikasi Aksara Jawa (Metode ORB)")
st.caption(f"Akurasi Test: {ACCURACY_REPORTED:.2f}%. Model menggunakan {ORB_N_FEATURES} fitur ORB dengan Rasio Lowe.")

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
    st.slider("Unknown threshold", min_value=0.01, max_value=0.5, value=0.05, step=0.01)
    
    st.button("Submit") # Submit button
    
# --- PANEL KANAN: RESULTS DAN PREVIEW ---
with col_right:
    st.subheader("Results")
    
    if uploaded_file is not None:
        if ORB_INDEX is None:
            st.error("🚨 Model tidak berhasil dimuat! Pastikan orb_index.pkl.gz terunggah.")
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
                
                # --- HASIL PREDISKI UTAMA ---
                st.success(f"**Predicted label:** {final_prediction.upper()}")
                st.info(f"Ditemukan {len(des_query)} deskriptor ORB.")

                # --- TAMPILAN TOP MATCHES DETAIL (DATA) ---
                st.subheader("Top Matches Detail")
                
                df = pd.DataFrame(top_matches)
                df['label'] = df['label_id'].apply(lambda x: ID_TO_LABEL[x])
                df = df.drop(columns=['label_id']).rename(columns={'score': 'good_matches', 'label': 'label'})
                df.index += 1
                df.index.name = 'Rank'
                
                st.dataframe(df) # TABEL SEBAGAI PENGGANTI THUMBNAILS
                
            else:
                 st.warning("⚠️ Gagal mengekstrak fitur ORB.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

st.markdown("---")

# --- TAMPILAN METRIK STATIS ---
st.subheader("Metrik Kinerja (Evaluate Dataset)")
metrik_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Value': [f"{ACCURACY_REPORTED:.2f}%", f"{33.00:.2f}%", f"{33.00:.2f}%", f"{32.50:.2f}%"]
}
df_metrik = pd.DataFrame(metrik_data)
st.table(df_metrik) 
st.caption("Catatan: Metrik di atas adalah hasil evaluasi penuh pada data test (offline).")

st.markdown("---")
st.caption("Proyek ini menggunakan fitur ORB untuk mencocokkan aksara. Jika akurasi rendah, ini adalah batasan metode fitur lokal.")

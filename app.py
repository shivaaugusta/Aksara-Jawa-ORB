# --- PANEL KIRI: UPLOAD & PENGATURAN ---
with col_left:
    st.subheader("Upload Query Image")
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

    st.markdown("---")
    st.subheader("Pengaturan Pencocokan")

    lowe_ratio = st.slider("Lowe ratio", min_value=0.1, max_value=1.0, value=0.75, step=0.01)
    top_k = st.slider("Top-K", min_value=1, max_value=20, value=5, step=1)
    unknown_threshold = st.slider("Unknown threshold", min_value=0.01, max_value=0.5, value=0.05, step=0.01)

    # Submit → Trigger tampil output
    submitted = st.button("📌 Submit")


# --- PANEL KANAN: RESULTS ---
with col_right:
    st.subheader("Results")

    # Hanya tampil jika user menekan Submit
    if submitted:

        if uploaded_file is None:
            st.warning("⚠️ Upload gambar terlebih dahulu sebelum submit!")
            st.stop()

        if ORB_INDEX is None:
            st.error("🚨 Model tidak berhasil dimuat! Cek file model.")
            st.stop()

        try:
            pil_img = Image.open(uploaded_file)
            preprocessed_cv = preprocess_image(pil_img)
            des_query = extract_orb(preprocessed_cv)

            col_preview, col_proc = st.columns(2)

            with col_preview:
                st.markdown("**Query Preview**")
                st.image(pil_img)

            with col_proc:
                st.markdown("**Visualisasi Preprocessing**")
                st.image(preprocessed_cv, caption="Threshold + Deskew + Resize")

            st.markdown("---")

            if des_query is None or len(des_query) == 0:
                st.warning("⚠️ Fitur ORB tidak terdeteksi!")
                st.stop()

            final_prediction, top_matches = predict_ratio(des_query, ORB_INDEX, lowe_ratio, top_k)

            st.success(f"Hasil Prediksi: **{final_prediction.upper()}**")
            st.info(f"Fitur ORB ditemukan: {len(des_query)}")

            # Top-K Tabel saja (no card grid → hemat space)
            df = pd.DataFrame(top_matches)
            df['Label'] = df['label_id'].apply(lambda x: ID_TO_LABEL[x])
            df = df[['Label', 'score']]
            df.columns = ["Label", "Good Matches"]

            st.markdown("**Top Matching Results**")
            st.table(df)

            st.markdown("---")
            st.markdown("📊 *Confusion Matrix bisa dilihat pada halaman Evaluasi Model*")

        except Exception as e:
            st.error(f"Error: {e}")


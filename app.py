import streamlit as st
import pandas as pd
import numpy as np

# PANEL UTAMA TETAP 1 LAYAR
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    max-height: 95vh;
    overflow: hidden;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Tombol buka pop up Evaluasi Model ---
if "show_eval" not in st.session_state:
    st.session_state.show_eval = False

def open_eval():
    st.session_state.show_eval = True

st.button("📊 Lihat Evaluasi Model", on_click=open_eval)

# Modal pop-up overlay
if st.session_state.show_eval:
    with st.container():
        st.markdown("""
        <style>
        .popup {
            position: fixed;
            top: 5%;
            left: 50%;
            transform: translateX(-50%);
            background: white;
            width: 90%;
            height: 90%;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 0 20px rgba(0,0,0,0.35);
            z-index: 9999;
            overflow-y: auto;
        }
        .close-btn {
            position: absolute;
            top:10px;
            right:20px;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="popup">', unsafe_allow_html=True)

        if st.button("❌", key="close_btn"):
            st.session_state.show_eval = False

        st.header("Evaluasi Model Offline")

        # Confusion Matrix Placeholder
        cm_labels = list(LABEL_MAP.keys())
        cm_data = np.zeros((20,20))
        st.dataframe(pd.DataFrame(cm_data, columns=cm_labels))

        st.subheader("📈 Metrik Model")
        st.table(pd.DataFrame({
            "Metric": ["Precision", "Recall", "F1-Score"],
            "Value": ["33%", "33%", "32.5%"]
        }))

        st.markdown("</div>", unsafe_allow_html=True)

import streamlit as st
from tab6_fungsi import (
    DIGIT_LABELS,
    parse_reference_input,
    prepare_training_from_reference,
    train_and_predict_ref_model,
    save_prediction_log
)

def tab6(df, lokasi):
    st.header("📊 Tab 6 - Prediksi 4D Mode B (Referensi → Target df)")
    st.markdown("Referensi: 8 digit per baris (total minimal 10 baris), 1 posisi per text area.")

    refs = {}
    for label in DIGIT_LABELS:
        ref_input = st.text_area(f"📌 Referensi 8 Digit Posisi {label.upper()} (≥10 baris)",
                                 height=300, key=f"ref_input_{label}")
        parsed = parse_reference_input(ref_input) if ref_input else None
        refs[label] = parsed

    if st.button("🔮 Jalankan Prediksi", key="predict_button"):
        all_valid = all(refs[label] for label in DIGIT_LABELS)
        if not all_valid:
            st.error("Pastikan semua text area diisi minimal 10 baris valid (8 digit per baris).")
            return

        hasil_prediksi = {}
        full_probs = {}

        for i, label in enumerate(DIGIT_LABELS):
            ref_data = refs[label]
            X, y = prepare_training_from_reference(ref_data, df, i)
            top6, probs = train_and_predict_ref_model(X, y, ref_data[-1])
            hasil_prediksi[label] = top6
            full_probs[label] = probs

        st.subheader("✅ Hasil Prediksi Top-6 per Posisi")
        for label in DIGIT_LABELS:
            st.markdown(f"**{label.upper()}**: {', '.join(str(d) for d in hasil_prediksi[label])}")

        st.subheader("📥 Simpan Log Prediksi")
        file_log = save_prediction_log(hasil_prediksi, lokasi)
        st.success(f"Log disimpan: `{file_log}`")

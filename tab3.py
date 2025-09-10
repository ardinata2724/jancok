# tab3.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from collections import Counter
from ensemble_probabilistic import ensemble_probabilistic
from ws_scan_catboost import (
    scan_ws_catboost,
    train_temp_lstm_model,
    get_top6_lstm_temp,
    DIGIT_LABELS,
)
from markov_model import top6_markov_hybrid
from tab3_utils import (
    detect_anomaly_latest,
    ensemble_confidence_voting,
    hybrid_voting,
    stacked_hybrid_auto,
    final_ensemble_with_markov,
    dynamic_alpha,
    log_prediction,
)

def show_live_accuracy(df, pred_dict):
    st.markdown("### 📊 Simulasi Live Accuracy")
    col1, col2 = st.columns(2)
    jumlah = col1.number_input("Jumlah Data Terakhir", 10, 200, 50, key="live_acc_n")
    kunci = col2.selectbox("Jenis Prediksi", list(pred_dict.keys()), key="live_acc_kunci")

    if kunci not in pred_dict: return st.warning("Belum ada hasil.")
    real = df["angka"].astype(str).apply(lambda x: int(x[DIGIT_LABELS.index(kunci)]))[-jumlah:]
    pred = pred_dict[kunci]
    benar = sum([r in pred for r in real])
    st.success(f"Akurasi Real ({kunci.upper()}): `{benar / len(real):.2%}` dari {jumlah} data.")

def show_auto_ensemble_adaptive(pred_dict):
    st.markdown("### 🧠 Auto Ensemble Adaptive")
    for label, pred in pred_dict.items():
        if pred:
            st.write(f"{label.upper()}: `{pred}`")

def tab3(df, lokasi):
    st.markdown("## 🎯 Prediksi Per Digit")
    min_ws = st.number_input("Min WS", 3, 20, 5)
    max_ws = st.number_input("Max WS", min_ws+1, 30, min_ws+6)
    folds = st.slider("Jumlah Fold", 2, 10, 3)
    seed = st.number_input("Seed", 0, 9999, 42)

    st.markdown("### ⚖️ Bobot Voting")
    lstm_w = st.slider("LSTM Weight", 0.5, 2.0, 1.2, 0.1)
    cb_w = st.slider("CatBoost Weight", 0.5, 2.0, 1.0, 0.1)
    hm_w = st.slider("Heatmap Weight", 0.0, 1.0, 0.6, 0.1)
    min_conf = st.slider("Min Confidence LSTM", 0.0, 1.0, 0.3, 0.05)

    hybrid_mode = st.selectbox("Mode Hybrid Voting", ["Dynamic Alpha", "Manual Alpha"])
    if hybrid_mode == "Manual Alpha":
        alpha_manual = st.slider("Alpha Manual", 0.0, 1.0, 0.5, 0.05)

    digit = st.selectbox("Digit yang ingin discan", ["(Semua)"] + DIGIT_LABELS)

    for key in ["tab3_stacked", "tab3_final", "simulasi_prediksi", "simulasi_target_real"]:
        if key not in st.session_state:
            st.session_state[key] = {}

    if st.button("🔎 Scan Per Digit", use_container_width=True):
        st.session_state.tab3_final = {}
        target_digits = DIGIT_LABELS if digit == "(Semua)" else [digit]
        simulasi_pred = {}
        simulasi_real = {}

        anomaly_mode = detect_anomaly_latest(df)
        if anomaly_mode:
            st.warning("⚠️ Anomali terdeteksi! Bobot Markov dikurangi dan alpha disesuaikan.")

        for label in target_digits:
            st.markdown(f"### 🔍 {label.upper()}")
            try:
                result_df = scan_ws_catboost(df, label, min_ws, max_ws, folds, seed)
                result_df["Stabilitas"] = result_df["Accuracy Mean"] - result_df["Accuracy Std"]
                best_row = result_df.loc[result_df["Stabilitas"].idxmax()]
                best_ws = int(best_row["WS"])
                acc_conf = best_row["Accuracy Mean"]

                lstm_dict = {}
                for _, row in result_df.sort_values("Accuracy Mean", ascending=False).head(3).iterrows():
                    ws = int(row["WS"])
                    try:
                        model = train_temp_lstm_model(df, label, ws, seed)
                        top6, probs = get_top6_lstm_temp(model, df, ws)
                        lstm_dict[ws] = (top6, probs)
                    except:
                        lstm_dict[ws] = ([], [])

                catboost_top6_all = [d for ws in lstm_dict.values() for d in ws[0]]
                heatmap_counts = Counter()
                for t in result_df["Top6"].apply(lambda x: [int(i) for i in str(x).split(",") if i.strip().isdigit()]): heatmap_counts.update(t)

                conf = ensemble_confidence_voting(lstm_dict, catboost_top6_all, heatmap_counts,
                                                  weights=[lstm_w, cb_w, hm_w], min_lstm_conf=min_conf)

                all_probs = [probs for _, probs in lstm_dict.values() if probs is not None]
                acc_prob = np.mean(result_df.sort_values("Accuracy Mean", ascending=False)["Accuracy Mean"].head(3))
                prob = ensemble_probabilistic(all_probs, [acc_conf]*len(all_probs)) if all_probs else []

                alpha_used = alpha_manual if hybrid_mode == "Manual Alpha" else dynamic_alpha(acc_conf, acc_prob)
                if anomaly_mode: alpha_used *= 0.6
                hybrid = hybrid_voting(conf, prob, alpha_used)

                try:
                    model = train_temp_lstm_model(df, label, best_ws, seed)
                    top6_direct, probs = get_top6_lstm_temp(model, df, best_ws)
                    if probs is not None and np.max(probs) < 0.3: top6_direct = []
                except:
                    top6_direct = []

                stacked = stacked_hybrid_auto(hybrid, top6_direct, acc_conf, acc_conf)
                markov_top6 = top6_markov_hybrid(df)[DIGIT_LABELS.index(label)]
                final = final_ensemble_with_markov(stacked, markov_top6, weight_markov=0.3 if not anomaly_mode else 0.15)

                st.session_state.tab3_stacked[label] = stacked
                st.session_state.tab3_final[label] = final

                log_prediction(label, conf, prob, hybrid, alpha_used, stacked, final, lokasi, anomaly=anomaly_mode)

                st.write(f"Confidence: `{conf}`")
                st.write(f"Probabilistic: `{prob}`")
                st.write(f"Hybrid α={alpha_used:.2f}: `{hybrid}`")
                st.write(f"Stacked Hybrid: `{stacked}`")
                st.success(f"📌 Final + Markov: `{final}`")

                real_digit = int(str(df.iloc[-1]["angka"])[DIGIT_LABELS.index(label)])
                simulasi_pred[label] = final
                simulasi_real[label] = real_digit

            except Exception as e:
                st.error(f"Gagal {label.upper()}: {e}")
                st.session_state.tab3_stacked[label] = []
                st.session_state.tab3_final[label] = []

        st.session_state.simulasi_prediksi = simulasi_pred
        st.session_state.simulasi_target_real = simulasi_real

    show_live_accuracy(df, st.session_state.tab3_final)
    show_auto_ensemble_adaptive(st.session_state.tab3_final)

    st.markdown("### 🎯 Hasil Prediksi Simulasi (Top-6 per Posisi)")
    simulasi_tabel = []
    for pos in DIGIT_LABELS:
        top6 = st.session_state.simulasi_prediksi.get(pos, [])
        real = st.session_state.simulasi_target_real.get(pos, None)
        if top6:
            simulasi_tabel.append({
                "Posisi": pos,
                "Top-6": ", ".join(str(d) for d in top6),
                "Target Real": real,
                "Match": "✅" if real in top6 else "❌"
            })
    if simulasi_tabel:
        st.table(pd.DataFrame(simulasi_tabel))

    st.markdown("---")
    col1, col2 = st.columns(2)
    if col1.button("📄 Lihat Log Prediksi", use_container_width=True):
        if os.path.exists("log_tab3.txt"):
            with open("log_tab3.txt", "r") as f: st.code(f.read())
        else: st.info("Belum ada log.")
    if col2.button("🧹 Hapus Log", use_container_width=True):
        if os.path.exists("log_tab3.txt"):
            os.remove("log_tab3.txt")
            st.success("Log dihapus.")
        else: st.info("Tidak ada log.")

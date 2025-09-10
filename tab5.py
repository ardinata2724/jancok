# tab5.py
import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from ensemble_probabilistic import ensemble_probabilistic
from ws_scan_catboost import scan_ws_catboost, train_temp_lstm_model, get_top6_lstm_temp, DIGIT_LABELS
from markov_model import top6_markov_hybrid

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def detect_anomaly_latest(df, window=10, std_threshold=2.0):
    """Deteksi anomali dari digit 4D terakhir."""
    if len(df) < window + 1:
        return False
    try:
        recent_digits = df["angka"].astype(str).apply(lambda x: [int(d) for d in x.zfill(4)])[-(window+1):]
        arr = np.array(recent_digits.tolist())
        if arr.shape[1] != 4:
            return False
        stds = np.std(arr, axis=0)
        return any(s > std_threshold for s in stds)
    except Exception as e:
        print(f"[Anomaly Detection Error] {e}")
        return False
def ensemble_confidence_voting(lstm_dict, catboost_top6, heatmap_counts,
                                weights=[1.2, 1.0, 0.6], min_lstm_conf=0.3):
    score = defaultdict(float)
    for ws, (digits, confs) in lstm_dict.items():
        if not digits or confs is None or len(confs) == 0:
            continue
        if max(confs) < min_lstm_conf:
            continue
        norm_confs = softmax(confs)
        for d, c in zip(digits, norm_confs):
            score[d] += weights[0] * c
    for d in catboost_top6:
        score[d] += weights[1]
    for d, count in heatmap_counts.items():
        score[d] += weights[2] * count
    if not score:
        return []
    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:6]]

def hybrid_voting(conf, prob, alpha=0.5):
    counter = defaultdict(float)
    for i, d in enumerate(conf or []):
        counter[d] += alpha * (6 - i)
    for i, d in enumerate(prob or []):
        counter[d] += (1 - alpha) * (6 - i)
    ranked = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:6]]

def stacked_hybrid_auto(hybrid, pred_direct, acc_hybrid=0.6, acc_direct=0.4):
    total = acc_hybrid + acc_direct
    w_hybrid = acc_hybrid / total if total else 0.5
    w_direct = acc_direct / total if total else 0.5
    counter = defaultdict(float)
    for i, d in enumerate(hybrid or []):
        counter[d] += w_hybrid * np.exp(-(i / 2))
    for i, d in enumerate(pred_direct or []):
        counter[d] += w_direct * np.exp(-(i / 2))
    ranked = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:6]]

def final_ensemble_with_markov(stacked, markov, weight_markov=0.3):
    counter = defaultdict(float)
    for i, d in enumerate(stacked or []):
        counter[d] += (6 - i)
    for i, d in enumerate(markov or []):
        counter[d] += weight_markov * (6 - i)
    ranked = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:6]]

def dynamic_alpha(acc_conf, acc_prob):
    return acc_conf / (acc_conf + acc_prob) if (acc_conf + acc_prob) else 0.5

def tab5(df, lokasi):
    st.header("🧠 Tab5 - Smart Adaptive Prediction")

    min_ws = st.number_input("Min Window Size", 3, 20, 5, key="tab5_min_ws")
    max_ws = st.number_input("Max Window Size", min_ws + 1, 30, min_ws + 6, key="tab5_max_ws")
    folds = st.slider("CV Fold", 2, 10, 3, key="tab5_folds")
    seed = st.number_input("Seed", 0, 9999, 42, key="tab5_seed")
    digit = st.selectbox("Digit Target", ["(Semua)"] + DIGIT_LABELS, key="tab5_digit")

    lstm_w = st.slider("LSTM Weight", 0.5, 2.0, 1.2, 0.1, key="tab5_lstm_w")
    cb_w = st.slider("CatBoost Weight", 0.5, 2.0, 1.0, 0.1, key="tab5_cb_w")
    hm_w = st.slider("Heatmap Weight", 0.0, 1.0, 0.6, 0.1, key="tab5_hm_w")
    min_conf = st.slider("Min Confidence LSTM", 0.0, 1.0, 0.3, 0.05, key="tab5_min_conf")

    hybrid_mode = st.selectbox("Hybrid Mode", ["Dynamic Alpha", "Manual Alpha"], key="tab5_hybrid_mode")
    if hybrid_mode == "Manual Alpha":
        alpha_manual = st.slider("Alpha Manual", 0.0, 1.0, 0.5, 0.05, key="tab5_alpha_manual")

    if "tab5_results" not in st.session_state:
        st.session_state.tab5_results = {}

    if st.button("🚀 Prediksi Adaptif", use_container_width=True, key="tab5_predict_button"):
        anomaly = detect_anomaly_latest(df)
        st.session_state.tab5_results = {}
        st.subheader("📊 Hasil Prediksi")

        if anomaly:
            st.warning("⚠️ Anomali terdeteksi: Model akan menyesuaikan bobot dan alpha.")

        target_digits = DIGIT_LABELS if digit == "(Semua)" else [digit]

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

                catboost_top6_all = [d for val in lstm_dict.values() for d in val[0]]
                heatmap_counts = Counter()
                for t in result_df["Top6"].apply(lambda x: [int(i) for i in str(x).split(",") if i.strip().isdigit()]): heatmap_counts.update(t)

                conf = ensemble_confidence_voting(
                    lstm_dict, catboost_top6_all, heatmap_counts,
                    weights=[lstm_w, cb_w, hm_w], min_lstm_conf=min_conf
                )

                all_probs = [probs for _, probs in lstm_dict.items() if probs is not None]
                acc_prob = np.mean(result_df.sort_values("Accuracy Mean", ascending=False)["Accuracy Mean"].head(3))
                prob = ensemble_probabilistic(all_probs, [acc_conf] * len(all_probs)) if all_probs else []

                alpha = alpha_manual if hybrid_mode == "Manual Alpha" else dynamic_alpha(acc_conf, acc_prob)
                if anomaly: alpha *= 0.6
                hybrid = hybrid_voting(conf, prob, alpha)

                try:
                    model = train_temp_lstm_model(df, label, best_ws, seed)
                    top6_direct, probs = get_top6_lstm_temp(model, df, best_ws)
                    if probs is not None and np.max(probs) < 0.3: top6_direct = []
                except:
                    top6_direct = []

                stacked = stacked_hybrid_auto(hybrid, top6_direct, acc_conf, acc_conf)
                markov_top6 = top6_markov_hybrid(df)[DIGIT_LABELS.index(label)]
                final = final_ensemble_with_markov(stacked, markov_top6, weight_markov=0.15 if anomaly else 0.3)

                st.session_state.tab5_results[label] = {
                    "conf": conf,
                    "prob": prob,
                    "hybrid": hybrid,
                    "stacked": stacked,
                    "final": final,
                    "target": int(str(df.iloc[-1]["angka"])[DIGIT_LABELS.index(label)])
                }

                st.write(f"Confidence: `{conf}`")
                st.write(f"Probabilistic: `{prob}`")
                st.write(f"Hybrid α={alpha:.2f}: `{hybrid}`")
                st.write(f"Stacked Hybrid: `{stacked}`")
                st.success(f"📌 Final: `{final}`")

            except Exception as e:
                st.error(f"Gagal prediksi {label.upper()}: {e}")
                st.session_state.tab5_results[label] = {}

    if st.session_state.tab5_results:
        st.subheader("🎯 Hasil Prediksi Simulasi")
        tabel = []
        for label, hasil in st.session_state.tab5_results.items():
            if not hasil: continue
            tabel.append({
                "Posisi": label,
                "Top-6 Final": ", ".join(str(d) for d in hasil["final"]),
                "Target Real": hasil["target"],
                "Match": "✅" if hasil["target"] in hasil["final"] else "❌"
            })
        st.table(pd.DataFrame(tabel))

# tab3_utils.py
import numpy as np
from collections import defaultdict

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def detect_anomaly_latest(df, window=10):
    if len(df) < window + 1:
        return False
    recent_digits = df["angka"].astype(str).apply(lambda x: [int(d) for d in x])[-(window+1):]
    recent_array = np.stack(recent_digits.to_numpy())  # ubah menjadi array shape (n, 4)
    stds = np.std(recent_array, axis=0)
    return any(s > 2 for s in stds)

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

def get_stacked_weights(acc_hybrid, acc_direct):
    total = acc_hybrid + acc_direct
    return (acc_hybrid / total, acc_direct / total) if total else (0.5, 0.5)

def stacked_hybrid_auto(hybrid, pred_direct, acc_hybrid=0.6, acc_direct=0.4):
    w_hybrid, w_direct = get_stacked_weights(acc_hybrid, acc_direct)
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

def log_prediction(label, conf, prob, hybrid, alpha, stacked=None, final=None, lokasi=None, anomaly=False):
    with open("log_tab3.txt", "a") as f:
        f.write(f"[{label.upper()}] | Lokasi: {lokasi}\n" if lokasi else f"[{label.upper()}]\n")
        if anomaly: f.write("Anomali terdeteksi, strategi disesuaikan.\n")
        f.write(f"Confidence Voting: {conf}\n")
        f.write(f"Probabilistic Voting: {prob}\n")
        f.write(f"Hybrid Voting (α={alpha:.2f}): {hybrid}\n")
        if stacked: f.write(f"Stacked Hybrid: {stacked}\n")
        if final: f.write(f"Final Hybrid: {final}\n")
        f.write("-" * 40 + "\n")

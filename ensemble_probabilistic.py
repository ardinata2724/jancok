import numpy as np
from collections import defaultdict

def ensemble_probabilistic(probs_list, catboost_accuracies=None):
    """
    Menggabungkan prediksi probabilistik dari berbagai model LSTM
    (opsional menggunakan bobot akurasi dari CatBoost).

    Args:
        probs_list (List[np.array]): List of confidence arrays (1D array length 10)
        catboost_accuracies (List[float], optional): Bobot jika tersedia

    Returns:
        List[int]: Top-6 digit dengan skor tertinggi
    """
    if not probs_list:
        print("[Ensemble Prob] ⚠️ probs_list kosong.")
        return []

    score = defaultdict(float)
    for i, probs in enumerate(probs_list):
        if probs is None or len(probs) != 10:
            continue
        weight = catboost_accuracies[i] if catboost_accuracies and i < len(catboost_accuracies) else 1.0
        for d, p in enumerate(probs):
            score[d] += weight * p

    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return [digit for digit, _ in ranked[:6]]

import streamlit as st
import pandas as pd
import os
import time
import random
import numpy as np
from collections import defaultdict, Counter
from itertools import product
from datetime import datetime
import re
import tensorflow as tf

# ==============================================================================
# BAGIAN 1: FUNGSI-FUNGSI & DEFINISI KELAS INTI
# ==============================================================================

# --- PERBAIKAN: Mendefinisikan Class di level atas agar stabil saat disimpan/dimuat ---
class PositionalEncoding(tf.keras.layers.Layer):
    def call(self, x):
        seq_len, d_model = tf.shape(x)[1], tf.shape(x)[2]
        pos = tf.cast(tf.range(seq_len)[:, tf.newaxis], dtype=tf.float32)
        i = tf.cast(tf.range(d_model)[tf.newaxis, :], dtype=tf.float32)
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = pos * angle_rates
        sines, cosines = tf.math.sin(angle_rads[:, 0::2]), tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        return x + tf.cast(tf.expand_dims(pos_encoding, 0), tf.float32)
    
    def get_config(self):
        config = super().get_config()
        return config

DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]
BBFS_LABELS = ["bbfs_ribuan-ratusan", "bbfs_ratusan-puluhan", "bbfs_puluhan-satuan"]
JUMLAH_LABELS = ["jumlah_depan", "jumlah_tengah", "jumlah_belakang"]
SHIO_LABELS = ["shio_depan", "shio_tengah", "shio_belakang"]
JALUR_LABELS = ["jalur_ribuan-ratusan", "jalur_ratusan-puluhan", "jalur_puluhan-satuan"]

JALUR_ANGKA_MAP = {
    1: "01*13*25*37*49*61*73*85*97*04*16*28*40*52*64*76*88*00*07*19*31*43*55*67*79*91*10*22*34*46*58*70*82*94",
    2: "02*14*26*38*50*62*74*86*98*05*17*29*41*53*65*77*89*08*20*32*44*56*68*80*92*11*23*35*47*59*71*83*95",
    3: "03*15*27*39*51*63*75*87*99*06*18*30*42*54*66*78*90*09*21*33*45*57*69*81*93*12*24*36*48*60*72*84*96"
}

SHIO_MAP = {
    1: {1, 13, 25, 37, 49, 61, 73, 85, 97}, 2: {2, 14, 26, 38, 50, 62, 74, 86, 98},
    3: {3, 15, 27, 39, 51, 63, 75, 87, 99}, 4: {4, 16, 28, 40, 52, 64, 76, 88, 0},
    5: {5, 17, 29, 41, 53, 65, 77, 89},    6: {6, 18, 30, 42, 54, 66, 78, 90},
    7: {7, 19, 31, 43, 55, 67, 79, 91},    8: {8, 20, 32, 44, 56, 68, 80, 92},
    9: {9, 21, 33, 45, 57, 69, 81, 93},    10: {10, 22, 34, 46, 58, 70, 82, 94},
    11: {11, 23, 35, 47, 59, 71, 83, 95},  12: {12, 24, 36, 48, 60, 72, 84, 96}
}

def parse_input_numbers(input_str):
    if not isinstance(input_str, str) or not input_str: return set()
    cleaned_str = re.sub(r'[^0-9,-]', ',', input_str)
    numbers = set()
    for part in cleaned_str.split(','):
        part = part.strip()
        if not part: continue
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                numbers.update(range(start, end + 1))
            except ValueError: continue
        else:
            try: numbers.add(int(part))
            except ValueError: continue
    return numbers

def generate_rekap_grid(dead_numbers):
    cols = st.columns(10)
    for i in range(100):
        num_str = f"{i:02d}"
        background_color = "#E57373" if i in dead_numbers else "#4CAF50"
        cols[i%10].markdown(f"<div style='background-color:{background_color}; color:white; text-align:center; padding:5px; border-radius:5px; margin:2px;'>{num_str}</div>", unsafe_allow_html=True)

def run_rekap_filter(state):
    kepala_off = parse_input_numbers(state.get('rekap_kepala_off', ''))
    ekor_off = parse_input_numbers(state.get('rekap_ekor_off', ''))
    jumlah_off = parse_input_numbers(state.get('rekap_jumlah_off', ''))
    shio_off = parse_input_numbers(state.get('rekap_shio_off', ''))
    ln_off = parse_input_numbers(state.get('rekap_ln_off', ''))
    for shio_num in shio_off:
        if shio_num in SHIO_MAP: ln_off.update(SHIO_MAP[shio_num])
    live_numbers, dead_numbers = [], set()
    for num in range(100):
        kepala, ekor, jumlah = num // 10, num % 10, (num // 10 + num % 10) % 10
        is_dead = any([kepala in kepala_off, ekor in ekor_off, jumlah in jumlah_off, num in ln_off])
        if is_dead: dead_numbers.add(num)
        else: live_numbers.append(f"{num:02d}")
    return live_numbers, dead_numbers

@st.cache_resource
def load_cached_model(model_path):
    from tensorflow.keras.models import load_model
    if os.path.exists(model_path):
        try:
            return load_model(model_path, custom_objects={"PositionalEncoding": PositionalEncoding})
        except Exception as e:
            st.error(f"Gagal memuat model di {model_path}: {e}")
    return None

def tf_preprocess_data(df, window_size=7):
    from tensorflow.keras.utils import to_categorical
    if len(df) < window_size + 1: return np.array([]), {}
    angka = df["angka"].values
    labels_to_process = DIGIT_LABELS + BBFS_LABELS + JUMLAH_LABELS + SHIO_LABELS
    sequences, targets = [], {label: [] for label in labels_to_process}
    for i in range(len(angka) - window_size):
        window = [str(x).zfill(4) for x in angka[i:i+window_size+1]]
        if any(not x.isdigit() for x in window): continue
        sequences.append([int(d) for num in window[:-1] for d in num])
        target_digits = [int(d) for d in window[-1]]
        for j, label in enumerate(DIGIT_LABELS): targets[label].append(to_categorical(target_digits[j], num_classes=10))
        jumlah_map = {"jumlah_depan": (target_digits[0] + target_digits[1]) % 10, "jumlah_tengah": (target_digits[1] + target_digits[2]) % 10, "jumlah_belakang": (target_digits[2] + target_digits[3]) % 10}
        for label, value in jumlah_map.items(): targets[label].append(to_categorical(value, num_classes=10))
        bbfs_map = {"bbfs_ribuan-ratusan": [target_digits[0], target_digits[1]], "bbfs_ratusan-puluhan": [target_digits[1], target_digits[2]], "bbfs_puluhan-satuan": [target_digits[2], target_digits[3]]}
        for label, digit_pair in bbfs_map.items():
            multi_hot_target = np.zeros(10, dtype=np.float32)
            for digit in np.unique(digit_pair): multi_hot_target[digit] = 1.0
            targets[label].append(multi_hot_target)
        shio_num_map = {"shio_depan": target_digits[0] * 10 + target_digits[1], "shio_tengah": target_digits[1] * 10 + target_digits[2], "shio_belakang": target_digits[2] * 10 + target_digits[3]}
        for label, two_digit_num in shio_num_map.items():
            shio_index = (two_digit_num - 1) % 12 if two_digit_num > 0 else 11
            targets[label].append(to_categorical(shio_index, num_classes=12))
    final_targets = {label: np.array(v) for label, v in targets.items() if v}
    return np.array(sequences), final_targets

def tf_preprocess_data_for_jalur(df, window_size, target_position):
    from tensorflow.keras.utils import to_categorical
    if len(df) < window_size + 1: return np.array([]), np.array([])
    jalur_map = {1: [1, 4, 7, 10], 2: [2, 5, 8, 11], 3: [3, 6, 9, 12]}
    shio_to_jalur = {shio: jalur for jalur, shios in jalur_map.items() for shio in shios}
    position_map = {'ribuan-ratusan': (0, 1), 'ratusan-puluhan': (1, 2), 'puluhan-satuan': (2, 3)}
    idx1, idx2 = position_map[target_position]
    angka = df["angka"].values
    sequences, targets = [], []
    for i in range(len(angka) - window_size):
        window = [str(x).zfill(4) for x in angka[i:i+window_size+1]]
        if any(not x.isdigit() for x in window): continue
        sequences.append([int(d) for num in window[:-1] for d in num])
        target_digits = [int(d) for d in window[-1]]
        two_digit_num = target_digits[idx1] * 10 + target_digits[idx2]
        shio_value = (two_digit_num - 1) % 12 + 1 if two_digit_num > 0 else 12
        targets.append(to_categorical(shio_to_jalur[shio_value] - 1, num_classes=3))
    return np.array(sequences), np.array(targets)

def build_tf_model(input_len, model_type, problem_type, num_classes):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout, Dense, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
    inputs = Input(shape=(input_len,)); x = Embedding(10, 64)(inputs); x = PositionalEncoding()(x)
    if model_type == "transformer":
        attn = MultiHeadAttention(num_heads=4, key_dim=64)(x, x); x = LayerNormalization()(x + attn)
    else:
        x = Bidirectional(LSTM(128, return_sequences=True))(x); x = Dropout(0.3)(x)
    x = GlobalAveragePooling1D()(x); x = Dense(128, activation='relu')(x); x = Dropout(0.2)(x)
    outputs, loss = (Dense(num_classes, activation='sigmoid')(x), "binary_crossentropy") if problem_type == "multilabel" else (Dense(num_classes, activation='softmax')(x), "categorical_crossentropy")
    model = Model(inputs, outputs)
    return model, loss

def top_n_model(df, lokasi, window_dict, model_type, top_n):
    results = []
    loc_id = lokasi.lower().strip().replace(" ", "_")
    all_models_exist = True
    for label in DIGIT_LABELS:
        model_path = f"saved_models/{loc_id}_{label}_{model_type}.h5"
        if not os.path.exists(model_path):
            st.error(f"Model untuk {label.upper()} tidak ditemukan. Silakan latih model terlebih dahulu di tab 'Manajemen Model'.")
            all_models_exist = False
    if not all_models_exist:
        return None, "Model tidak lengkap."

    with st.spinner("Memuat model dan menjalankan prediksi..."):
        for label in DIGIT_LABELS:
            ws = window_dict.get(label, 7)
            if len(df) < ws:
                st.error(f"Data tidak cukup untuk prediksi {label.upper()} (diperlukan {ws} baris, tersedia {len(df)}).")
                return None, "Data tidak cukup."
            X, _ = tf_preprocess_data(df, ws)
            if X.shape[0] == 0:
                st.error(f"Gagal memproses data untuk prediksi {label.upper()} dengan WS={ws}.")
                return None, "Gagal memproses data."
            
            model_path = f"saved_models/{loc_id}_{label}_{model_type}.h5"
            model = load_cached_model(model_path)
            
            if model is None: 
                return None, "Model gagal dimuat."

            pred = model.predict(X[-1:], verbose=0)
            top_digits = list(np.argsort(pred[0])[-top_n:][::-1])
            results.append(top_digits)
    return results, None

def find_best_window_size(df, label, model_type, min_ws, max_ws, top_n, top_n_shio):
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.metrics import TopKCategoricalAccuracy
    best_ws, best_score, table_data = None, -1, []
    is_jalur_scan = label in JALUR_LABELS
    if is_jalur_scan: pt, k, nc, cols = "jalur_multiclass", 2, 3, ["Window Size", "Prediksi", "Angka Jalur", "Angka Mati"]
    elif label in BBFS_LABELS: pt, k, nc, cols = "multilabel", top_n, 10, ["Window Size", f"Top-{k}", "Angka Mati"]
    elif label in SHIO_LABELS: pt, k, nc, cols = "shio", top_n_shio, 12, ["Window Size", f"Top-{k}", "Shio Mati"]
    else: pt, k, nc, cols = "multiclass", top_n, 10, ["Window Size", f"Top-{k}", "Angka Mati"]
        
    bar = st.progress(0, text=f"Memulai Scan {label.upper()}... [0%]")
    total_ws = (max_ws - min_ws) + 1
    for i, ws in enumerate(range(min_ws, max_ws + 1)):
        progress_value = (i + 1) / total_ws
        percentage = int(progress_value * 100)
        bar.progress(progress_value, text=f"Mencoba WS={ws}... [{percentage}%]")
        try:
            if is_jalur_scan: X, y = tf_preprocess_data_for_jalur(df, ws, label.split('_')[1])
            else: X, y_dict = tf_preprocess_data(df, ws); y = y_dict.get(label)
            if X.shape[0] < 10: continue
            
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            model, loss = build_tf_model(X.shape[1], model_type, 'multiclass' if is_jalur_scan else pt, nc)
            metrics = ['accuracy']
            if pt != 'multilabel': metrics.append(TopKCategoricalAccuracy(k=k))
            model.compile(optimizer="adam", loss=loss, metrics=metrics)
            model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val), callbacks=[EarlyStopping(monitor='val_loss', patience=3)], verbose=0)
            evals = model.evaluate(X_val, y_val, verbose=0); preds = model.predict(X_val, verbose=0)
            
            if is_jalur_scan:
                top_indices = np.argsort(preds[-1])[::-1][:2]
                pred_str = f"{top_indices[0] + 1}-{top_indices[1] + 1}"
                angka_jalur_str = f"Jalur {top_indices[0] + 1} => {JALUR_ANGKA_MAP[top_indices[0] + 1]}\n\nJalur {top_indices[1] + 1} => {JALUR_ANGKA_MAP[top_indices[1] + 1]}"
                all_jalur = {1, 2, 3}; predicted_jalur = {top_indices[0] + 1, top_indices[1] + 1}
                jalur_mati = list(all_jalur - predicted_jalur)[0]
                angka_mati_str = JALUR_ANGKA_MAP[jalur_mati]
                score = (evals[1] * 0.3) + (evals[2] * 0.7)
                table_data.append((ws, pred_str, angka_jalur_str, angka_mati_str))
            else:
                avg_conf = np.mean(np.sort(preds, axis=1)[:, -k:])*100
                top_indices = np.argsort(preds[-1])[::-1][:k]
                if pt == "shio": all_numbers, top_numbers = set(range(1, 13)), set(top_indices + 1)
                else: all_numbers, top_numbers = set(range(10)), set(top_indices)
                off_numbers = all_numbers - top_numbers
                pred_str = ", ".join(map(str, sorted(list(top_numbers)))); off_str = ", ".join(map(str, sorted(list(off_numbers))))
                score = (evals[1] * 0.7) + (avg_conf/100*0.3) if pt=='multilabel' else (evals[1]*0.2)+(evals[2]*0.5)+(avg_conf/100*0.3)
                table_data.append((ws, pred_str, off_str))
            if score > best_score: best_score, best_ws = score, ws
        except Exception as e: st.warning(f"Gagal di WS={ws}: {e}"); continue
    bar.empty()
    return best_ws, pd.DataFrame(table_data, columns=cols) if table_data else pd.DataFrame()

# --- PERUBAHAN: Fungsi train_and_save_model sekarang menerima daftar label yang akan dilatih ---
def train_and_save_model(df, lokasi, window_dict, model_type, labels_to_train):
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.callbacks import EarlyStopping
    st.info(f"Memulai pelatihan untuk {lokasi}...")
    lokasi_id = lokasi.lower().strip().replace(" ", "_")
    if not os.path.exists("saved_models"): os.makedirs("saved_models")
    for label in labels_to_train:
        ws = window_dict.get(label, 7)
        bar = st.progress(0, text=f"Memproses {label.upper()} (WS={ws})..."); X, y_dict = tf_preprocess_data(df, ws)
        if label not in y_dict or y_dict[label].shape[0] < 10:
            st.warning(f"Data tidak cukup untuk melatih '{label.upper()}'."); bar.empty(); continue
        X_train, X_val, y_train, y_val = train_test_split(X, y_dict[label], test_size=0.2, random_state=42)
        bar.progress(50, text=f"Melatih {label.upper()}...")
        model, loss = build_tf_model(X.shape[1], model_type, 'multiclass', 10)
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), callbacks=[EarlyStopping(monitor='val_loss', patience=5)], verbose=0)
        model_path = f"saved_models/{lokasi_id}_{label}_{model_type}.h5"
        bar.progress(75, text=f"Menyimpan {label.upper()}...")
        model.save(model_path)
        bar.progress(100, text=f"Model {label.upper()} berhasil disimpan!"); time.sleep(1); bar.empty()

# ==============================================================================
# APLIKASI STREAMLIT UTAMA
# ==============================================================================
st.set_page_config(page_title="Prediksi 4D", layout="wide")

# ... (Sisa kode tidak berubah)
# (Kode dari baris ini ke bawah sama persis dengan file sebelumnya)

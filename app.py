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
# BAGIAN 1: DEFINISI FUNGSI & KONSTANTA
# ==============================================================================

# --- Konstanta Global ---
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

# --- Fungsi Helper ---
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

# --- Fungsi Terkait AI & Model ---
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
    inputs = Input(shape=(input_len,))
    x = Embedding(10, 64)(inputs)
    if model_type == "transformer":
        attn = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = LayerNormalization()(x + attn)
    else:
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Dropout(0.3)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs, loss = (Dense(num_classes, activation='sigmoid')(x), "binary_crossentropy") if problem_type == "multilabel" else (Dense(num_classes, activation='softmax')(x), "categorical_crossentropy")
    model = Model(inputs, outputs)
    return model, loss

def find_best_window_size(df, label, model_type, min_ws, max_ws, top_n, top_n_shio):
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.metrics import TopKCategoricalAccuracy
    best_ws, best_score, table_data = None, -1, []
    is_jalur_scan = label in JALUR_LABELS
    
    if is_jalur_scan:
        pt, k, nc = "jalur_multiclass", 2, 3
        cols = ["Window Size", "Prediksi", "Angka Jalur", "Angka Mati"]
    elif label in BBFS_LABELS:
        pt, k, nc = "multilabel", top_n, 10
        cols = ["Window Size", f"Top-{k}", "Angka Mati"]
    elif label in SHIO_LABELS:
        pt, k, nc = "shio", top_n_shio, 12
        cols = ["Window Size", f"Top-{k}", "Shio Mati"]
    else: 
        pt, k, nc = "multiclass", top_n, 10
        cols = ["Window Size", f"Top-{k}", "Angka Mati"]
        
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

# ==============================================================================
# APLIKASI STREAMLIT UTAMA
# ==============================================================================
st.set_page_config(page_title="Prediksi 4D", layout="wide")

if 'angka_list' not in st.session_state: st.session_state.angka_list = []
if 'angka_list_2' not in st.session_state: st.session_state.angka_list_2 = []
if 'active_data' not in st.session_state: st.session_state.active_data = 'A'
if 'scan_outputs' not in st.session_state: st.session_state.scan_outputs = {}
if 'scan_queue' not in st.session_state: st.session_state.scan_queue = []
if 'current_scan_job' not in st.session_state: st.session_state.current_scan_job = None
if 'data_asli_pembalik' not in st.session_state: st.session_state.data_asli_pembalik = ""
if 'hasil_dibalik_pembalik' not in st.session_state: st.session_state.hasil_dibalik_pembalik = ""
if 'rekap_results' not in st.session_state or 'dead' not in st.session_state.rekap_results:
    st.session_state.rekap_results = {"live": [f"{i:02d}" for i in range(100)], "dead": set()}
rekap_inputs = ['rekap_kepala_off', 'rekap_ekor_off', 'rekap_shio_off', 'rekap_jumlah_off', 'rekap_ln_off']
for key in rekap_inputs:
    if key not in st.session_state: st.session_state[key] = ""

st.title("Prediksi 4D")
st.caption("editing by: Andi Prediction")
try: from lokasi_list import lokasi_list
except ImportError: lokasi_list = ["BULLSEYE", "HONGKONGPOOLS", "HONGKONG LOTTO", "SYDNEYPOOLS", "SYDNEY LOTTO", "SINGAPURA"]

with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_lokasi = st.selectbox("🌍 Pilih Pasaran", lokasi_list)
    putaran = st.number_input("🔁 Jumlah Putaran Terakhir", 10, 1000, 100)
    mode_angka = st.radio("Mode", ("2D", "3D", "4D"), horizontal=True, key="mode_angka_selector")
    st.markdown("---")
    st.markdown("### 🎯 Opsi Scan")
    jumlah_digit = st.slider("🔢 Jumlah Digit Prediksi", 1, 9, 9)
    jumlah_digit_shio = st.slider("🐉 Jumlah Digit Prediksi Khusus Shio", 1, 12, 11)
    use_transformer = st.checkbox("🤖 Gunakan Transformer", value=True)
    model_type = "transformer" if use_transformer else "lstm"
    st.markdown("---")
    st.markdown("### 🪟 Pengaturan Window Size (WS)")
    min_ws = st.number_input("Min WS", 1, 99, 5)
    max_ws = st.number_input("Max WS", 1, 100, 31)

def get_file_name_from_lokasi(lokasi):
    cleaned_lokasi = lokasi.lower().replace(" ", "")
    if "hongkonglotto" in cleaned_lokasi: return "keluaran hongkong lotto.txt"
    if "hongkongpools" in cleaned_lokasi: return "keluaran hongkongpools.txt"
    if "sydneylotto" in cleaned_lokasi: return "keluaran sydney lotto.txt"
    if "sydneypools" in cleaned_lokasi: return "keluaran sydneypools.txt"
    return f"keluaran {lokasi.lower()}.txt"

st.subheader("Pengelolaan Data Angka")
if st.button("Ambil Data dari Keluaran Angka", use_container_width=True):
    folder_data = "data_keluaran"
    base_filename = get_file_name_from_lokasi(selected_lokasi)
    file_path = os.path.join(folder_data, base_filename)
    try:
        with open(file_path, 'r') as f: lines = f.readlines()
        angka_from_file = [line.strip()[-4:] for line in lines[-putaran:] if line.strip() and line.strip()[-4:].isdigit()]
        if angka_from_file:
            target_list = 'angka_list' if st.session_state.active_data == 'A' else 'angka_list_2'
            st.session_state[target_list] = angka_from_file
            st.success(f"{len(angka_from_file)} data berhasil dimuat ke 'Data {st.session_state.active_data}' dari '{base_filename}'.")
            st.rerun()
    except FileNotFoundError: st.error(f"File tidak ditemukan: '{file_path}'.")

st.radio("Pilih Set Data Aktif:", ('A', 'B'), key='active_data', horizontal=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("##### ✏️ Edit Data Manual A")
    riwayat_text_1 = st.text_area("1 angka per baris (Data A):", "\n".join(st.session_state.angka_list), height=300, key="manual_data_input_1", label_visibility="collapsed", disabled=(st.session_state.active_data != 'A'))
    if riwayat_text_1 != "\n".join(st.session_state.angka_list):
        st.session_state.angka_list = [line.strip()[:4] for line in riwayat_text_1.splitlines() if line.strip() and line.strip()[:4].isdigit()]
        st.rerun()
with col2:
    st.markdown("##### ✏️ Edit Data Manual B")
    riwayat_text_2 = st.text_area("1 angka per baris (Data B):", "\n".join(st.session_state.angka_list_2), height=300, key="manual_data_input_2", label_visibility="collapsed", disabled=(st.session_state.active_data != 'B'))
    if riwayat_text_2 != "\n".join(st.session_state.angka_list_2):
        st.session_state.angka_list_2 = [line.strip().split()[-1][:4] for line in riwayat_text_2.splitlines() if line.strip() and line.strip().split()[-1][:4].isdigit()]
        st.rerun()

active_list = st.session_state.angka_list if st.session_state.active_data == 'A' else st.session_state.angka_list_2
df = pd.DataFrame({"angka": active_list})

tab_scan, tab_auto_scan, tab_pembalik, tab_rekap_2d = st.tabs(["🪟 Scan Manual", "⚡ Scan Otomatis & Monitoring", "🔄 Pembalik Urutan", "🔢 Rekap Angka 2D"])

def display_scan_progress_and_results(df, model_type, min_ws, max_ws, jumlah_digit, jumlah_digit_shio):
    if st.session_state.current_scan_job and len(df) < max_ws + 10:
        st.error(f"SCAN DIBATALKAN: Data Anda hanya {len(df)} baris. Diperlukan lebih dari {max_ws + 10} baris untuk Max WS={max_ws}.")
        st.info("Solusi: Tambah data melalui 'Ambil Data' atau kurangi nilai 'Max WS' di sidebar.")
        st.session_state.current_scan_job = None
        return 

    if st.session_state.scan_queue:
        queue_display = " ➡️ ".join([f"**{job.replace('_', ' ').upper()}**" for job in st.session_state.scan_queue])
        st.info(f"Antrian Berikutnya: {queue_display}")
    
    if not st.session_state.current_scan_job and st.session_state.scan_queue:
        st.session_state.current_scan_job = st.session_state.scan_queue.pop(0)
        st.rerun()
        
    if st.session_state.current_scan_job:
        label = st.session_state.current_scan_job
        st.warning(f"⏳ Sedang menjalankan scan untuk **{label.replace('_', ' ').upper()}**...")
        best_ws, result_table = find_best_window_size(df, label, model_type, min_ws, max_ws, jumlah_digit, jumlah_digit_shio)
        st.session_state.scan_outputs[label] = {"ws": best_ws, "table": result_table}
        st.session_state.current_scan_job = None
        st.rerun()

    if st.session_state.scan_outputs:
        st.markdown("---")
        # --- PERUBAHAN: Mengembalikan header ke bentuk semula ---
        st.subheader("✅ Hasil Scan Selesai")
                
        display_order = DIGIT_LABELS + JUMLAH_LABELS + BBFS_LABELS + SHIO_LABELS + JALUR_LABELS
        for label in display_order:
            if label in st.session_state.scan_outputs:
                data = st.session_state.scan_outputs[label]
                with st.expander(f"Hasil untuk {label.replace('_', ' ').upper()}", expanded=True):
                    result_df = data.get("table")
                    if result_df is not None and not result_df.empty:
                        st.dataframe(result_df, use_container_width=True)
                        if data["ws"] is not None: st.info(f"💡 **WS terbaik yang ditemukan: {data['ws']}**")
                    else: st.warning("Tidak ada hasil yang valid untuk rentang WS ini.")
        st.markdown("---")

with tab_scan:
    st.subheader("Pencarian Window Size (WS) Optimal per Kategori")
    st.info("Tekan tombol di bawah untuk menambahkan scan ke antrian. Pantau progres dan lihat hasilnya di tab 'Scan Otomatis & Monitoring'.")
    st.divider()
    def create_scan_button(label, container):
        is_pending = label in st.session_state.scan_queue or st.session_state.current_scan_job == label
        if container.button(f"🔎 Scan {label.replace('_', ' ').upper()}", key=f"scan_{label}", use_container_width=True, disabled=is_pending):
            st.session_state.scan_queue.append(label); st.toast(f"✅ Scan untuk '{label.upper()}' ditambahkan ke antrian."); st.rerun()
    
    category_tabs = st.tabs(["Digit", "Jumlah", "BBFS", "Shio", "Jalur Main"])
    with category_tabs[0]:
        cols = st.columns(len(DIGIT_LABELS))
        for label, container in zip(DIGIT_LABELS, cols): create_scan_button(label, container)
    with category_tabs[1]:
        cols = st.columns(len(JUMLAH_LABELS))
        for label, container in zip(JUMLAH_LABELS, cols): create_scan_button(label, container)
    with category_tabs[2]:
        cols = st.columns(len(BBFS_LABELS))
        for label, container in zip(BBFS_LABELS, cols): create_scan_button(label, container)
    with category_tabs[3]:
        cols = st.columns(len(SHIO_LABELS))
        for label, container in zip(SHIO_LABELS, cols): create_scan_button(label, container)
    with category_tabs[4]:
        cols = st.columns(len(JALUR_LABELS))
        for label, container in zip(JALUR_LABELS, cols): create_scan_button(label, container)

with tab_auto_scan:
    st.subheader(f"Otomatisasi & Monitoring Scan untuk Mode {mode_angka}")
    st.info("Jalankan scan otomatis atau pantau progres scan yang sedang berjalan (baik dari manual maupun otomatis) di sini.")
    is_scanning = bool(st.session_state.scan_queue or st.session_state.current_scan_job)
    if st.button(f"🚀 Jalankan Scan Otomatis {mode_angka}", use_container_width=True, type="primary", disabled=is_scanning):
        scan_jobs = []
        if mode_angka == '2D': scan_jobs = ['puluhan', 'satuan', 'jumlah_belakang', 'bbfs_puluhan-satuan', 'shio_belakang', 'jalur_puluhan-satuan']
        elif mode_angka == '3D': scan_jobs = ['ratusan', 'puluhan', 'satuan', 'jumlah_tengah', 'jumlah_belakang', 'bbfs_ratusan-puluhan', 'bbfs_puluhan-satuan', 'shio_tengah', 'shio_belakang', 'jalur_ratusan-puluhan', 'jalur_puluhan-satuan']
        elif mode_angka == '4D': scan_jobs = DIGIT_LABELS + JUMLAH_LABELS + BBFS_LABELS + SHIO_LABELS + JALUR_LABELS
        if scan_jobs:
            st.session_state.scan_queue.extend(scan_jobs)
            st.toast(f"✅ Auto-scan untuk mode {mode_angka} ditambahkan ke antrian!")
            st.rerun()
    st.divider()

    # --- PERUBAHAN DIMULAI ---
    # Tombol hapus dipindahkan ke sini, sesuai permintaan pada gambar.
    # Tombol hanya akan muncul jika ada hasil scan yang bisa dihapus.
    if st.session_state.scan_outputs:
        if st.button("❌ Hapus Hasil Scan", use_container_width=True):
            st.session_state.scan_outputs.clear()
            st.toast("🗑️ Semua hasil scan telah dihapus.")
            st.rerun()
    # --- PERUBAHAN SELESAI ---

    display_scan_progress_and_results(df, model_type, min_ws, max_ws, jumlah_digit, jumlah_digit_shio)

with tab_pembalik:
    st.subheader("Aplikasi Pembalik Urutan")
    st.caption("Tempel daftar angka atau teks Anda di bawah untuk membalik urutannya dari bawah ke atas.")
    col1_pembalik, col2_pembalik = st.columns(2)
    with col1_pembalik:
        st.text_area("Data Asli", height=350, key="data_asli_pembalik")
    with col2_pembalik:
        st.text_area("Hasil Dibalik", value=st.session_state.hasil_dibalik_pembalik, height=350, disabled=True)
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        if st.button("Balikkan Urutan!", use_container_width=True, type="primary"):
            if st.session_state.data_asli_pembalik:
                st.session_state.hasil_dibalik_pembalik = "\n".join(st.session_state.data_asli_pembalik.splitlines()[::-1])
                st.rerun()
    with btn_col2:
        if st.button("Salin Hasil", use_container_width=True):
             st.toast("💡 Silakan salin teks dari kotak 'Hasil Dibalik' secara manual (Ctrl+C).")
    with btn_col3:
        if st.button("Bersihkan", use_container_width=True):
            st.session_state.data_asli_pembalik, st.session_state.hasil_dibalik_pembalik = "", ""
            st.rerun()

with tab_rekap_2d:
    st.subheader("Rekap Angka 2D")
    st.caption("Alat bantu untuk menganalisa dan memfilter angka 2D.")
    col1, col2 = st.columns([1.2, 1.8])
    with col1:
        st.text_input("Kepala Off", placeholder="misal: 1 2 3", key="rekap_kepala_off")
        st.text_input("Ekor Off", placeholder="misal: 4 5 6", key="rekap_ekor_off")
        st.text_input("Jumlah Off", placeholder="misal: 7 8", key="rekap_jumlah_off")
        st.text_input("Shio Off", placeholder="misal: 11 12", key="rekap_shio_off")
        st.text_area("LN OFF / jalur main off", placeholder="masukan LN off pisahkan dengan spasi, koma, atau *", key="rekap_ln_off")
        
        btn_col1, btn_col2 = st.columns(2)
        if btn_col1.button("Generate", use_container_width=True, type="primary"):
            live, dead = run_rekap_filter(st.session_state)
            st.session_state.rekap_results = {"live": live, "dead": dead}
        
        if btn_col2.button("Reset", use_container_width=True):
            for key in rekap_inputs: st.session_state[key] = ""
            st.session_state.rekap_results = {"live": [f"{i:02d}" for i in range(100)], "dead": set()}
    
    with col2:
        st.markdown("<h5 style='text-align: center;'>Papan Angka</h5>", unsafe_allow_html=True)
        generate_rekap_grid(st.session_state.rekap_results.get('dead', set()))

    st.markdown("---")
    live_numbers = st.session_state.rekap_results.get('live', [])
    st.subheader(f"Hasil Rekap ({len(live_numbers)} LN)")
    st.text_area(
        "Hasil Rekap", 
        value="*".join(live_numbers), 
        height=200,
        label_visibility="collapsed"
    )

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def split_digits(num_str):
    return [int(d) for d in str(num_str).zfill(4)]

def analyze_frequency(data, pos):
    digits = [split_digits(num)[pos] for num in data]
    return Counter(digits)

def analyze_delay(data, pos):
    delay = {i: 0 for i in range(10)}
    last_seen = {i: -1 for i in range(10)}
    delays = []
    for idx, num in enumerate(data):
        d = split_digits(num)[pos]
        for i in range(10):
            if i == d:
                if last_seen[i] != -1:
                    delay[i] = idx - last_seen[i]
                last_seen[i] = idx
        delays.append(delay.copy())
    return delays[-1] if delays else {}

def analyze_trend(data, pos):
    trend = {'naik': 0, 'turun': 0, 'tetap': 0}
    prev = None
    for num in data:
        curr = split_digits(num)[pos]
        if prev is not None:
            if curr > prev:
                trend['naik'] += 1
            elif curr < prev:
                trend['turun'] += 1
            else:
                trend['tetap'] += 1
        prev = curr
    return trend

def predict_trend(data, pos):
    trend_list = [split_digits(num)[pos] for num in data]
    trend_array = pd.Series(trend_list)
    if len(trend_array) < 3:
        return "-"
    diff = trend_array.diff().dropna()
    last_trend = diff.iloc[-3:]
    naik = sum(last_trend > 0)
    turun = sum(last_trend < 0)
    if naik > turun:
        return "Naik"
    elif turun > naik:
        return "Turun"
    else:
        return "Tetap"

def even_odd_analysis(data, pos):
    counts = {'genap': 0, 'ganjil': 0}
    for num in data:
        d = split_digits(num)[pos]
        counts['genap' if d % 2 == 0 else 'ganjil'] += 1
    return counts

def predict_even_odd(data, pos):
    last_digit = split_digits(data[-1])[pos]
    return "Genap" if last_digit % 2 != 0 else "Ganjil"

def big_small_analysis(data, pos):
    counts = {'besar': 0, 'kecil': 0}
    for num in data:
        d = split_digits(num)[pos]
        counts['besar' if d >= 5 else 'kecil'] += 1
    return counts

def predict_big_small(data, pos):
    last_digit = split_digits(data[-1])[pos]
    return "Besar" if last_digit < 5 else "Kecil"

def digit_position_heatmap(data):
    pos_counts = np.zeros((4, 10))
    for num in data:
        digits = split_digits(num)
        for i, d in enumerate(digits):
            pos_counts[i][d] += 1
    return pos_counts

def zigzag_pattern(data):
    pattern = 0
    for i in range(2, len(data)):
        a = int(str(data[i - 2])[-1])
        b = int(str(data[i - 1])[-1])
        c = int(str(data[i])[-1])
        if (a < b > c) or (a > b < c):
            pattern += 1
    return pattern

def predict_next_pattern(freq, delay, heatmap):
    predicted_digits = []
    sorted_delay = sorted(delay.items(), key=lambda x: x[1], reverse=True)
    predicted_digits += [d for d, _ in sorted_delay[:2]]
    sorted_freq = sorted(freq.items(), key=lambda x: x[1])
    predicted_digits += [d for d, _ in sorted_freq[:2]]
    rare_pos_digits = [int(np.argmin(row)) for row in heatmap]
    predicted_digits += rare_pos_digits
    return list(dict.fromkeys(predicted_digits))[:6]

def render_digit_badge(digit):
    return f"""<div style='
        background-color: #0e1117;
        color: white;
        border-radius: 50%;
        width: 48px;
        height: 48px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        box-shadow: 0 0 6px rgba(0, 255, 255, 0.4);
        margin-right: 6px;
        margin-top: 6px;
    '>{digit}</div>"""

def render_delay(delay_dict):
    delay_df = pd.DataFrame(delay_dict.items(), columns=["Digit", "Delay"]).sort_values("Digit")
    st.markdown("**📊 Grafik Delay Kemunculan Digit**")
    st.bar_chart(delay_df.set_index("Digit"))

    st.markdown("**🕒 Top-3 Digit dengan Delay Tertinggi**")
    top_delay = sorted(delay_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    for digit, d in top_delay:
        st.info(f"Digit `{digit}` belum muncul selama `{d}` langkah.")

    badge_html = "".join([
        f"<div style='background:#333;color:#fff;border-radius:50%;width:48px;height:48px;"
        f"display:inline-flex;align-items:center;justify-content:center;font-size:20px;"
        f"margin:4px;box-shadow:0 0 6px rgba(255, 255, 0, 0.4);'>{d}</div>"
        for d, _ in top_delay
    ])
    st.markdown("**🔢 Badge Digit dengan Delay Tertinggi:**")
    st.markdown(f"<div style='display:flex;flex-wrap:wrap'>{badge_html}</div>", unsafe_allow_html=True)

def find_historical_pattern(data, pos, pattern_length):
    digits = [split_digits(num)[pos] for num in data]
    if len(digits) <= pattern_length:
        return [], [], digits
    current_pattern = digits[-pattern_length:]
    matches = []
    for i in range(len(digits) - pattern_length):
        window = digits[i:i+pattern_length]
        next_index = i + pattern_length
        if window == current_pattern and next_index < len(digits):
            matches.append(digits[next_index])
    return current_pattern, matches, digits

def tab4(df):
    if "angka" not in df.columns:
        st.error("❌ Kolom 'angka' tidak ditemukan di data.")
        return

    angka_data = df["angka"].dropna().astype(int).tolist()
    if not angka_data:
        st.warning("⚠️ Data 4D kosong.")
        return

    run_all = st.button("🔍 Jalankan Analisis Lengkap")

    if not run_all:
        st.info("Tekan tombol di atas untuk menampilkan hasil analisis.")
        return

    digit_pos_label = ["Ribu", "Ratus", "Puluh", "Satuan"]
    tabs = st.tabs(digit_pos_label)

    for i, tab in enumerate(tabs):
        with tab:
            st.subheader(f"📌 Posisi Digit: {digit_pos_label[i]}")
            recent_data = angka_data

            freq = analyze_frequency(recent_data, i)
            freq_df = pd.DataFrame(freq.items(), columns=["Digit", "Frekuensi"]).sort_values("Digit")
            st.markdown("**📈 Frekuensi Digit (semua data)**")
            st.bar_chart(freq_df.set_index("Digit"))

            st.markdown("**⏱️ Delay Kemunculan Digit**")
            delay = analyze_delay(recent_data, i)
            render_delay(delay)

            st.markdown("**📉 Tren Naik / Turun**")
            trend = analyze_trend(recent_data, i)
            for key, val in trend.items():
                st.success(f"Jumlah tren `{key}`: `{val}`")
            st.info(f"🔮 Prediksi tren berikutnya: **{predict_trend(recent_data, i)}**")

            st.markdown("**🧮 Statistik Ganjil / Genap**")
            eo = even_odd_analysis(recent_data, i)
            for key, val in eo.items():
                st.success(f"Jumlah digit `{key}`: `{val}`")
            st.info(f"🔮 Prediksi berikutnya: **{predict_even_odd(recent_data, i)}**")

            st.markdown("**🔢 Statistik Besar / Kecil**")
            bs = big_small_analysis(recent_data, i)
            for key, val in bs.items():
                st.success(f"Jumlah digit `{key}`: `{val}`")
            st.info(f"🔮 Prediksi berikutnya: **{predict_big_small(recent_data, i)}**")

            # --- Pola Historis ---
            st.markdown("**🔁 Pencarian Pola Historis**")
            slider_key = f"pattern_len_{i}"
            default_len = 3
            if slider_key not in st.session_state:
                st.session_state[slider_key] = default_len

            with st.form(key=f"form_pattern_{i}", clear_on_submit=False):
                new_len = st.slider(f"📏 Pilih panjang pola (Posisi {digit_pos_label[i]})", 2, 6, st.session_state[slider_key], key=f"slider_{i}")
                submitted = st.form_submit_button("🔍 Cari Pola Historis")

                if submitted:
                    st.session_state[slider_key] = new_len
                    pattern, matches, digits = find_historical_pattern(recent_data, i, new_len)
                    if matches:
                        st.success(f"Pola terakhir: {pattern} pernah muncul sebanyak {len(matches)} kali.")
                        st.info("Digit setelah pola tersebut:")
                        badge = "".join([render_digit_badge(d) for d in matches])
                        st.markdown(f"<div style='display:flex;flex-wrap:wrap'>{badge}</div>", unsafe_allow_html=True)
                    else:
                        st.warning("❌ Pola terakhir belum pernah muncul sebelumnya.")

    st.markdown("### 🔥 Heatmap Posisi Digit")
    heatmap = digit_position_heatmap(angka_data)
    fig, ax = plt.subplots(figsize=(8, 2))
    sns.heatmap(heatmap, annot=True, fmt=".0f", cmap="YlGnBu",
                xticklabels=list(range(10)),
                yticklabels=digit_pos_label,
                ax=ax)
    ax.set_title("Heatmap Posisi Digit")
    st.pyplot(fig)

    st.markdown("### 🔀 Pola Zigzag")
    zz = zigzag_pattern(angka_data)
    st.success(f"Zigzag pattern ditemukan: `{zz}` kali")

    st.markdown("### 🧠 Insight Otomatis")
    freq_all = analyze_frequency(angka_data, 0)
    delay_all = analyze_delay(angka_data, 0)
    most_common_digit = max(freq_all.items(), key=lambda x: x[1])[0]
    delay_sorted = sorted(delay_all.items(), key=lambda x: x[1], reverse=True)
    st.info(f"Digit paling sering muncul (ribuan): `{most_common_digit}`")
    if delay_sorted:
        st.info(f"Digit dengan delay tertinggi (ribuan): `{delay_sorted[0][0]}` selama `{delay_sorted[0][1]}` langkah")

    st.markdown("### 🔮 Prediksi Pola Selanjutnya")
    prediksi = predict_next_pattern(freq_all, delay_all, heatmap)
    badge_html = "".join([render_digit_badge(d) for d in prediksi])
    st.markdown(f"<div style='display:flex; flex-wrap:wrap'>{badge_html}</div>", unsafe_allow_html=True)

    st.caption("📌 Prediksi berdasarkan kombinasi statistik delay, frekuensi, dan posisi digit.")

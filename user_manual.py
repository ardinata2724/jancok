import streamlit as st

def tampilkan_user_manual():
    with st.expander("📘 Panduan Lengkap Penggunaan Aplikasi", expanded=False):

        st.markdown("### 🧭 Navigasi Panduan")
        st.markdown("""
        - 📌 [Pengantar Aplikasi](#pengantar-aplikasi)
        - ⚙️ [Pengaturan Awal](#pengaturan-awal)
        - 📥 [Pengambilan & Input Data](#pengambilan--input-data)
        - 🧠 [Manajemen Model AI](#manajemen-model-ai)
        - 🔮 [Prediksi dan Confidence](#prediksi-dan-confidence)
        - 💡 [Kombinasi 4D Terkuat](#kombinasi-4d-terkuat)
        - 📊 [Evaluasi Akurasi](#evaluasi-akurasi)
        - 🪟 [Pencarian Window Size](#pencarian-window-size)
        - ⚠️ [Tips & Catatan Tambahan](#tips--catatan-tambahan)
        """)

        with st.expander("📌 Pengantar Aplikasi"):
            st.markdown("""
            Aplikasi ini dirancang untuk melakukan prediksi angka 4D menggunakan berbagai metode:
            - 📈 **Markov Chain**
            - 🧠 **LSTM AI / Transformer**
            - 🤖 **Ensemble AI + Markov**

            Fokus aplikasi adalah mencari **Top-6 prediksi angka per posisi digit** (ribuan, ratusan, puluhan, satuan) dan mengkombinasikannya menjadi **kombinasi 4D paling menjanjikan** berdasarkan confidence dari model.
            """)

        with st.expander("⚙️ Pengaturan Awal"):
            st.markdown("""
            Di sidebar, Anda akan menemukan pengaturan berikut:

            - **🌍 Lokasi Pasaran**: Pilih lokasi tempat prediksi ingin dilakukan (misalnya: sgp, hk, sidney).
            - **📅 Hari**: Tentukan hari data yang diambil (harian, kemarin, 2hari, 3hari).
            - **🔁 Putaran**: Jumlah baris data historis yang akan digunakan (misalnya 100).
            - **🧠 Metode Prediksi**:
                - `Markov`: Prediksi berbasis transisi angka.
                - `Markov Order-2`: Memperhitungkan 2 angka sebelumnya.
                - `Markov Gabungan`: Kombinasi dua metode Markov.
                - `LSTM AI`: Menggunakan model deep learning (LSTM/Transformer).
                - `Ensemble AI + Markov`: Gabungan AI dan Markov untuk akurasi lebih tinggi.
            - **🪟 Window Size per Digit**: Sesuaikan panjang input historis per digit (ribuan s/d satuan).
            - **🤖 Transformer**: Centang untuk menggunakan arsitektur Transformer (default: LSTM).
            - **🎯 Mode Prediksi**: Pilih antara `confidence`, `ranked`, atau `hybrid`.
            - **📈 Confidence Power / Min Confidence / Voting Mode**: Pengaruh ke pemilihan kombinasi akhir.
            """)

        with st.expander("📥 Pengambilan & Input Data"):
            st.markdown("""
            Anda bisa mengambil data angka dari API atau mengisi manual:

            - Klik tombol **🔄 Ambil Data dari API** untuk mengambil data otomatis.
            - Gunakan token API dan parameter seperti pasaran, hari, dan jumlah putaran.
            - Di bawahnya tersedia kolom untuk **edit manual angka**.
            - Input satu angka 4-digit per baris, contoh:
                ```
                1234
                5642
                0001
                ```

            Data ini akan digunakan sebagai dataset untuk pelatihan dan prediksi model.
            """)

        with st.expander("🧠 Manajemen Model AI"):
            st.markdown("""
            Hanya tersedia jika metode `LSTM AI` atau `Ensemble AI` dipilih:

            - Model akan dibagi ke dalam 4 digit: ribuan, ratusan, puluhan, satuan.
            - Anda bisa:
                - Melihat status model.
                - Hapus model per digit.
                - Hapus file log pelatihan.
                - **Melatih ulang semua model** dengan parameter dan data terbaru.

            Pelatihan model mempertimbangkan:
            - Window size per digit.
            - Jumlah data.
            - Arsitektur model (LSTM atau Transformer).
            """)

        with st.expander("🔮 Prediksi dan Confidence"):
            st.markdown("""
            Setelah menekan tombol **🔮 Prediksi**, Anda akan melihat:

            - **Top-6 prediksi angka** per digit (`ribuan`, `ratusan`, `puluhan`, `satuan`).
            - **Confidence bar chart** untuk tiap prediksi.
            - Jika metode AI digunakan, prediksi disertai confidence per angka (hasil softmax/normalisasi).

            Kombinasi ini menjadi dasar dalam membangun prediksi kombinasi 4D terbaik.
            """)

        with st.expander("💡 Kombinasi 4D Terkuat"):
            st.markdown("""
            Jika metode AI aktif, Anda akan melihat:
            - 10 kombinasi angka 4D terbaik (misal: `1234`, `4021`).
            - Confidence tiap kombinasi ditampilkan.
            - Digabung menggunakan metode `product` atau `average` dari confidence per digit.

            Anda dapat mengatur kekuatan confidence (`power`) dan batas minimal (`min_conf`) agar hasil lebih selektif.
            """)

        with st.expander("📊 Evaluasi Akurasi"):
            st.markdown("""
            Evaluasi akurasi dilakukan untuk metode berbasis AI:
            - Menampilkan **Top-1 accuracy** dan **Top-6 accuracy** untuk tiap digit.
            - Gunakan ini untuk mengetahui kualitas model saat ini.
            - Pastikan jumlah data cukup (> window size).
            """)

        with st.expander("🪟 Scan Angka Otomatis"):
            st.markdown("""
            Masuk ke tab **Scan Angka** untuk mencari window size terbaik:
            - Scan per digit atau semua sekaligus.
            - Bisa menggunakan **Cross Validation (CV)** untuk akurasi lebih robust.
            - Hasil ditampilkan dalam tabel dan grafik.
            - Setelah dapat hasil terbaik, gunakan hasil ini untuk melatih model kembali.

            Tujuan fitur ini:
            - Menemukan panjang historis input optimal untuk setiap digit.
            - Meningkatkan akurasi prediksi signifikan.
            """)

        with st.expander("⚠️ Tips & Catatan Tambahan"):
            st.markdown("""
            - 🧪 Model LSTM/Transformer perlu data cukup panjang.
            - 💾 Model dan log disimpan di folder `saved_models` dan `training_logs`.
            - 🔄 Setiap perubahan window size atau data wajib dilatih ulang.
            - 🔧 Model tidak otomatis menyimpan perubahan, klik tombol latih manual.
            - 📉 Markov cocok untuk data pendek, AI cocok untuk pola kompleks.

            Gunakan metode **Ensemble AI + Markov** untuk hasil paling stabil.
            """)

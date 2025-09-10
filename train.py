# train.py
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout, Dense, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# --- DEFINISI ULANG ---
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

def build_model(input_len, model_type="transformer"):
    inputs = Input(shape=(input_len,))
    x = Embedding(input_dim=10, output_dim=64)(inputs)
    x = PositionalEncoding()(x)
    if model_type == "transformer":
        attn = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = LayerNormalization()(x + attn)
    else:
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Dropout(0.3)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(10, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_for_training(df, window_size=7):
    angka = df["angka"].values
    sequences, targets_ribuan, targets_ratusan, targets_puluhan, targets_satuan = [], [], [], [], []
    for i in range(len(angka) - window_size):
        window = [str(x).zfill(4) for x in angka[i:i+window_size]]
        target_val = str(angka[i+window_size]).zfill(4)
        
        sequences.append([int(d) for num in window for d in num])
        targets_ribuan.append(to_categorical(int(target_val[0]), num_classes=10))
        targets_ratusan.append(to_categorical(int(target_val[1]), num_classes=10))
        targets_puluhan.append(to_categorical(int(target_val[2]), num_classes=10))
        targets_satuan.append(to_categorical(int(target_val[3]), num_classes=10))
        
    return {
        "sequences": np.array(sequences),
        "ribuan": np.array(targets_ribuan),
        "ratusan": np.array(targets_ratusan),
        "puluhan": np.array(targets_puluhan),
        "satuan": np.array(targets_satuan),
    }

def main_training():
    # --- PENGATURAN YANG PERLU ANDA UBAH ---
    NAMA_PASARAN = "HONGKONGPOOLS"  # Ganti dengan nama pasaran yang ingin dilatih
    FILE_DATA_INPUT = "keluaran hongkongpools.txt" # Sesuaikan dengan nama file data
    WINDOW_SIZE = 7  # Sesuaikan dengan window size yang ingin digunakan
    MODEL_TYPE = "transformer" # 'transformer' atau 'lstm'
    # -----------------------------------------

    print(f"Memulai pelatihan untuk pasaran: {NAMA_PASARAN}")
    
    # 1. Baca dan siapkan data
    try:
        df = pd.read_csv(FILE_DATA_INPUT, header=None, names=['angka'])
        print(f"Berhasil memuat {len(df)} baris data dari {FILE_DATA_INPUT}")
    except Exception as e:
        print(f"Gagal memuat file data: {e}")
        return

    data = preprocess_for_training(df, window_size=WINDOW_SIZE)
    if data["sequences"].shape[0] == 0:
        print("Data tidak cukup untuk diproses setelah windowing.")
        return

    # 2. Latih model untuk setiap posisi digit
    lokasi_id = NAMA_PASARAN.lower().strip().replace(" ", "_")
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")

    for digit_label in ["ribuan", "ratusan", "puluhan", "satuan"]:
        print(f"\n--- Melatih model untuk: {digit_label.upper()} ---")
        X = data["sequences"]
        y = data[digit_label]
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = build_model(X.shape[1], model_type=MODEL_TYPE)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=2)
        
        model_path = f"saved_models/{lokasi_id}_{digit_label}_{MODEL_TYPE}.h5"
        model.save(model_path)
        print(f"✅ Model untuk {digit_label.upper()} berhasil disimpan di: {model_path}")

if __name__ == "__main__":
    main_training()

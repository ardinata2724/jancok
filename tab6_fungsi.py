import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.utils import to_categorical
from collections import Counter
from datetime import datetime

DIGIT_LABELS = ["ribuan", "ratusan", "puluhan", "satuan"]

def parse_reference_input(textarea_input):
    lines = textarea_input.strip().splitlines()
    data = []
    for line in lines:
        line = line.strip()
        if len(line) == 8 and line.isdigit():
            data.append([int(d) for d in line])
    return data if len(data) >= 10 else None  # minimal 10 baris

def prepare_training_from_reference(ref_data, df, digit_index):
    X = np.array(ref_data[:-1])  # semua kecuali baris terakhir
    y_digit = df["angka"].astype(str).apply(lambda x: int(x[digit_index])).iloc[-(len(X)):]
    y = to_categorical(y_digit, num_classes=10)
    return X, y

def build_ref_model(input_dim=8):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_predict_ref_model(X, y, ref_last_row):
    model = build_ref_model(input_dim=X.shape[1])
    model.fit(X, y, epochs=30, batch_size=8, verbose=0)
    preds = model.predict(np.array([ref_last_row]), verbose=0)[0]
    top6 = np.argsort(preds)[::-1][:6].tolist()
    return top6, preds.tolist()

def save_prediction_log(result_dict, lokasi):
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"prediksi_tab6_modeB_{lokasi}_{today}.txt"
    with open(filename, "w") as f:
        f.write(f"Prediksi 4D (Mode B) - Lokasi: {lokasi} - Tanggal: {today}\n\n")
        for label, values in result_dict.items():
            f.write(f"{label.upper()}: {', '.join(str(v) for v in values)}\n")
    return filename

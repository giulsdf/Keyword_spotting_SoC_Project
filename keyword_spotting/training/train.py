#!/usr/bin/env python3
"""
training/train.py
=================
Scarica Google Speech Commands, prepara gli MFCC,
addestra una CNN leggera ed esporta il modello in TFLite.

Eseguire sul PC:
    cd keyword_spotting
    python training/train.py
"""

import os, sys
os.environ["PATH"] += r";C:\Users\giulia.difante\AppData\Local\Temp\WinGet\Gyan.FFmpeg.8.0.1\extracted\ffmpeg-8.0.1-full_build\bin"
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

from config import (SAMPLE_RATE, CHUNK, N_MFCC, HOP_LENGTH, N_FFT,
                    LABELS, NUM_CLASSES, MODEL_KERAS, MODEL_TFLITE)
from utils.mfcc import extract_mfcc, pad_or_trim

# ── parametri training ──────────────────────────────────────────────────────
EPOCHS      = 30
BATCH_SIZE  = 64
VAL_SPLIT   = 0.15
TARGET_FRAMES = 1 + (CHUNK - N_FFT) // HOP_LENGTH   # numero di frame MFCC attesi

# ── mapping label Speech Commands → nostre classi ──────────────────────────
# Le keyword nel dataset che vogliamo riconoscere (devono coincidere con LABELS,
# escluso "silence"). Tutte le altre finiscono in "silence" (unknown).
WANTED_KEYWORDS = [l for l in LABELS if l != "silence"]


def label_to_idx(label_str: str) -> int:
    if label_str in WANTED_KEYWORDS:
        return LABELS.index(label_str)
    return LABELS.index("silence")   # unknown → silence


def process_example(audio_tensor, label_str: str):
    """Converte un esempio TFDS in (mfcc_array, class_idx)."""
    audio = audio_tensor.numpy().astype(np.float32) / 32768.0

    # padding/trim a CHUNK campioni
    if len(audio) < CHUNK:
        audio = np.pad(audio, (0, CHUNK - len(audio)))
    else:
        audio = audio[:CHUNK]

    mfcc = extract_mfcc(audio)
    mfcc = pad_or_trim(mfcc, TARGET_FRAMES)
    mfcc = mfcc[..., np.newaxis]          # (n_mfcc, frames, 1)
    idx  = label_to_idx(label_str)
    return mfcc.astype(np.float32), idx


def load_dataset():
    print("📥  Caricamento Google Speech Commands v2 ...")
    ds_full, info = tfds.load(
        "speech_commands",
        split="train+test+validation",
        with_info=True,
        shuffle_files=True
    )

    X, y = [], []
    print("⚙️   Estrazione MFCC (può richiedere qualche minuto) ...")
    for ex in tqdm(ds_full):
        audio = ex["audio"]
        label = info.features["label"].int2str(ex["label"].numpy())
        mfcc, idx = process_example(audio, label)
        X.append(mfcc)
        y.append(idx)

    X = np.array(X, dtype=np.float32)   # (N, n_mfcc, frames, 1)
    y = np.array(y, dtype=np.int32)
    print(f"   Dataset: {len(X)} esempi, shape={X.shape}")
    return X, y


def build_model(input_shape):
    inp = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)

    x = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)

    x = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = tf.keras.Model(inp, out)
    return model


def export_tflite(model):
    os.makedirs("models", exist_ok=True)
    print("💾  Salvataggio modello Keras ...")
    model.save(MODEL_KERAS)

    print("🔄  Conversione a TFLite (con quantizzazione float16) ...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    with open(MODEL_TFLITE, "wb") as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(MODEL_TFLITE) / 1024
    print(f"✅  TFLite salvato: {MODEL_TFLITE} ({size_kb:.1f} KB)")


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)

    X, y = load_dataset()

    # shuffle e split
    idx   = np.random.permutation(len(X))
    X, y  = X[idx], y[idx]
    split = int(len(X) * (1 - VAL_SPLIT))
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    input_shape = X_tr.shape[1:]   # (n_mfcc, frames, 1)
    print(f"🏗️   Build modello, input_shape={input_shape}")
    model = build_model(input_shape)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint(MODEL_KERAS, save_best_only=True)
    ]

    print("🚀  Avvio training ...")
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    export_tflite(model)
    print("\n🎉  Training completato. Ora copia 'models/keyword_model.tflite' sulla MP1.")


if __name__ == "__main__":
    main()

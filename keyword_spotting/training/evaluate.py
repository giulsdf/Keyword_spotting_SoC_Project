#!/usr/bin/env python3
"""
training/evaluate.py
====================
Valuta il modello TFLite su un campione del dataset
e mostra la confusion matrix.

Eseguire sul PC dopo train.py:
    python training/evaluate.py
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tflite_runtime.interpreter as tflite_rt
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from config import LABELS, MODEL_TFLITE, SAMPLE_RATE, CHUNK, N_FFT, HOP_LENGTH
from utils.mfcc import extract_mfcc, pad_or_trim

TARGET_FRAMES = 1 + (CHUNK - N_FFT) // HOP_LENGTH
WANTED        = [l for l in LABELS if l != "silence"]
MAX_SAMPLES   = 2000   # campioni da valutare (per velocità)


def load_test_set():
    ds, info = tfds.load("speech_commands", split="test",
                         with_info=True, shuffle_files=True)
    X, y = [], []
    for ex in ds.take(MAX_SAMPLES):
        audio = ex["audio"].numpy().astype(np.float32) / 32768.0
        label = info.features["label"].int2str(ex["label"].numpy())
        if len(audio) < CHUNK:
            audio = np.pad(audio, (0, CHUNK - len(audio)))
        else:
            audio = audio[:CHUNK]
        mfcc = pad_or_trim(extract_mfcc(audio), TARGET_FRAMES)
        mfcc = mfcc[np.newaxis, ..., np.newaxis].astype(np.float32)
        idx  = LABELS.index(label) if label in WANTED else 0
        X.append(mfcc)
        y.append(idx)
    return np.concatenate(X, axis=0), np.array(y)


def run_tflite(interpreter, X):
    input_idx  = interpreter.get_input_details()[0]['index']
    output_idx = interpreter.get_output_details()[0]['index']
    preds = []
    for i in range(len(X)):
        interpreter.set_tensor(input_idx, X[i:i+1])
        interpreter.invoke()
        out = interpreter.get_tensor(output_idx)[0]
        preds.append(np.argmax(out))
    return np.array(preds)


def main():
    if not os.path.exists(MODEL_TFLITE):
        print(f"❌  Modello non trovato: {MODEL_TFLITE}. Esegui prima train.py.")
        return

    interpreter = tflite_rt.Interpreter(model_path=MODEL_TFLITE)
    interpreter.allocate_tensors()

    print("📊  Caricamento test set ...")
    X, y_true = load_test_set()

    print("🔍  Inference ...")
    y_pred = run_tflite(interpreter, X)

    print("\n" + classification_report(y_true, y_pred, target_names=LABELS))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, rotation=45)
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=150)
    print("📷  Confusion matrix salvata in models/confusion_matrix.png")


if __name__ == "__main__":
    main()

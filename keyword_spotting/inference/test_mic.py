#!/usr/bin/env python3
"""
inference/test_mic.py
=====================
Test rapido: registra 2 secondi dal microfono,
estrae MFCC e verifica che la pipeline funzioni
(non serve il modello TFLite addestrato).

Eseguire sulla MP1 o sul PC:
    python inference/test_mic.py
"""

import os, sys, time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pyaudio

from config import SAMPLE_RATE, CHUNK, N_FFT, HOP_LENGTH, N_MFCC
from utils.mfcc import extract_mfcc, pad_or_trim

TARGET_FRAMES = 1 + (CHUNK - N_FFT) // HOP_LENGTH

def test_microphone():
    pa = pyaudio.PyAudio()

    print("🎙️  Dispositivi di input disponibili:")
    for i in range(pa.get_device_count()):
        dev = pa.get_device_info_by_index(i)
        if dev['maxInputChannels'] > 0:
            print(f"   [{i}] {dev['name']}")

    stream = pa.open(rate=SAMPLE_RATE, channels=1,
                     format=pyaudio.paInt16, input=True,
                     frames_per_buffer=CHUNK)

    print(f"\n⏺️   Registrazione di {CHUNK/SAMPLE_RATE:.1f}s ...")
    raw = stream.read(CHUNK)
    stream.stop_stream(); stream.close(); pa.terminate()

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    print(f"   Campioni acquisiti: {len(audio)}")
    print(f"   Volume RMS: {np.sqrt(np.mean(audio**2)):.4f}")

    mfcc = extract_mfcc(audio)
    mfcc = pad_or_trim(mfcc, TARGET_FRAMES)
    print(f"   MFCC shape: {mfcc.shape}  (atteso: ({N_MFCC}, {TARGET_FRAMES}))")

    assert mfcc.shape == (N_MFCC, TARGET_FRAMES), "❌ Shape MFCC non corretta!"
    print("✅  Pipeline audio OK.\n")

    # verifica modello (se esiste)
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'keyword_model.tflite')
    if os.path.exists(model_path):
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow as tf; tflite = tf.lite

        interp = tflite.Interpreter(model_path=model_path)
        interp.allocate_tensors()
        inp_shape = interp.get_input_details()[0]['shape']
        print(f"🔍  Modello trovato, input shape: {inp_shape}")

        test_inp = mfcc[np.newaxis, :, :, np.newaxis].astype(np.float32)
        interp.set_tensor(interp.get_input_details()[0]['index'], test_inp)
        interp.invoke()
        out = interp.get_tensor(interp.get_output_details()[0]['index'])[0]
        print(f"   Output softmax: {out}")
        print("✅  Modello TFLite OK.")
    else:
        print("ℹ️   Modello TFLite non trovato — esegui prima training/train.py sul PC.")


if __name__ == "__main__":
    test_microphone()

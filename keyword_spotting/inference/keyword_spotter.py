#!/usr/bin/env python3
"""
inference/keyword_spotter.py
============================
Programma principale da eseguire sulla STM32MP1.
Ascolta il microfono in continuo, classifica ogni finestra audio
con il modello TFLite e logga keyword + timestamp.

Eseguire sulla MP1:
    python inference/keyword_spotter.py

"""

import os, sys, time, datetime, signal, struct
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pyaudio

# Usa tflite_runtime sulla MP1, tensorflow su PC
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

from config import (SAMPLE_RATE, CHUNK, N_MFCC, HOP_LENGTH, N_FFT,
                    LABELS, MODEL_TFLITE, THRESHOLD, COOLDOWN_SEC, LOG_FILE, HOP_SEC)
from utils.mfcc import extract_mfcc, pad_or_trim

# ── calcola TARGET_FRAMES in base al CHUNK e ai parametri MFCC ─────────────
TARGET_FRAMES = 1 + (CHUNK - N_FFT) // HOP_LENGTH


class KeywordSpotter:
    def __init__(self):
        print(f"🔄  Caricamento modello: {MODEL_TFLITE}")
        self.interpreter = tflite.Interpreter(model_path=MODEL_TFLITE)
        self.interpreter.allocate_tensors()
        self.input_idx  = self.interpreter.get_input_details()[0]['index']
        self.output_idx = self.interpreter.get_output_details()[0]['index']

        self.pa     = None
        self.stream = None
        self._running = True

        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        self.log_f = open(LOG_FILE, "a", buffering=1)   # line-buffered

        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        # cooldown: tiene traccia dell'ultimo istante in cui ogni keyword è stata loggata
        self._last_detection = {}   # keyword → timestamp

    # ── inference ──────────────────────────────────────────────────────────
    def preprocess(self, raw_bytes: bytes) -> np.ndarray:
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        mfcc  = extract_mfcc(audio)
        mfcc  = pad_or_trim(mfcc, TARGET_FRAMES)
        return mfcc[np.newaxis, :, :, np.newaxis].astype(np.float32)  # (1, mfcc, frames, 1)

    def predict(self, inp: np.ndarray):
        self.interpreter.set_tensor(self.input_idx, inp)
        self.interpreter.invoke()
        probs = self.interpreter.get_tensor(self.output_idx)[0]
        idx   = int(np.argmax(probs))
        return LABELS[idx], float(probs[idx])

    # ── logging ────────────────────────────────────────────────────────────
    def log(self, keyword: str, confidence: float, unix_ts: float):
        iso_ts = datetime.datetime.fromtimestamp(unix_ts).isoformat(timespec='milliseconds')
        entry  = f"{iso_ts} | unix={unix_ts:.3f} | keyword={keyword} | conf={confidence:.4f}\n"
        print(entry, end="", flush=True)
        self.log_f.write(entry)

    # ── main loop ──────────────────────────────────────────────────────────
    def run(self):
        self.pa = pyaudio.PyAudio()

        # Trova dispositivo di input (microfono di default)
        dev_idx = self.pa.get_default_input_device_info()['index']
        print(f"🎙️   Microfono: {self.pa.get_device_info_by_index(dev_idx)['name']}")

        self.stream = self.pa.open(
            rate=SAMPLE_RATE,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            input_device_index=dev_idx,
            frames_per_buffer=CHUNK
        )

        print(f"✅  In ascolto (finestre da {int(CHUNK/SAMPLE_RATE*1000)} ms) ...")
        print(f"    Keyword attive: {[l for l in LABELS if l != 'silence']}")
        print(f"    Soglia confidenza: {THRESHOLD}")
        print(f"    Log: {LOG_FILE}")
        print("    Premi Ctrl+C per fermare.\n")

        hop_samples = int(SAMPLE_RATE * HOP_SEC)   # campioni di hop tra finestre
        buffer      = np.zeros(CHUNK, dtype=np.float32)

        while self._running:
            try:
                raw = self.stream.read(hop_samples, exception_on_overflow=False)
            except OSError:
                continue

            ts_acquisition = time.time()

            # sliding window: shifta il buffer e inserisce nuovi campioni
            new = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            buffer = np.roll(buffer, -len(new))
            buffer[-len(new):] = new

            # MFCC + inference
            mfcc = extract_mfcc(buffer)
            mfcc = pad_or_trim(mfcc, TARGET_FRAMES)
            inp  = mfcc[np.newaxis, :, :, np.newaxis].astype(np.float32)

            keyword, confidence = self.predict(inp)

            if keyword != "silence" and confidence >= THRESHOLD:
                # cooldown: ignora se la stessa keyword e gia stata loggata di recente
                last = self._last_detection.get(keyword, 0)
                if ts_acquisition - last >= COOLDOWN_SEC:
                    self.log(keyword, confidence, ts_acquisition)
                    self._last_detection[keyword] = ts_acquisition

    # ── shutdown ───────────────────────────────────────────────────────────
    def _shutdown(self, *_):
        print("\n🛑  Arresto in corso ...")
        self._running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.pa:
            self.pa.terminate()
        self.log_f.close()
        sys.exit(0)


if __name__ == "__main__":
    spotter = KeywordSpotter()
    spotter.run()

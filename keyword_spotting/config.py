# config.py — parametri condivisi tra training e inference

# Audio
SAMPLE_RATE     = 16000       # Hz
WINDOW_SEC      = 1.0         # durata finestra (secondi)
HOP_SEC         = 0.5         # hop tra finestre consecutive (secondi)
CHUNK           = int(SAMPLE_RATE * WINDOW_SEC)

# MFCC
N_MFCC          = 40
N_FFT           = 512
HOP_LENGTH      = 160         # 10 ms a 16kHz
N_MELS          = 40

# Keyword da riconoscere + classe "silence/unknown"
LABELS          = ["silence", "go", "stop", "yes", "no"]
NUM_CLASSES     = len(LABELS)

# Modello
MODEL_TFLITE    = "models/keyword_model.tflite"
MODEL_KERAS     = "models/keyword_model.keras"

# Inference
THRESHOLD       = 0.85        # confidenza minima per loggare
COOLDOWN_SEC    = 1.5         # secondi minimi tra due rilevazioni consecutive
LOG_FILE        = "logs/keyword_log.txt"

# utils/mfcc.py — estrazione MFCC

import numpy as np
from scipy.fftpack import dct
from scipy.signal import get_window
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import SAMPLE_RATE, N_MFCC, N_FFT, HOP_LENGTH, N_MELS


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def mel_filterbank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS):
    low_mel  = hz_to_mel(0)
    high_mel = hz_to_mel(sr / 2)
    mel_pts  = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_pts   = mel_to_hz(mel_pts)
    bin_pts  = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_m_minus = bin_pts[m - 1]
        f_m       = bin_pts[m]
        f_m_plus  = bin_pts[m + 1]
        for k in range(f_m_minus, f_m):
            fbank[m-1, k] = (k - f_m_minus) / (f_m - f_m_minus + 1e-8)
        for k in range(f_m, f_m_plus):
            fbank[m-1, k] = (f_m_plus - k) / (f_m_plus - f_m + 1e-8)
    return fbank

# Pre-calcola filterbank una sola volta
_FILTERBANK = mel_filterbank()

def extract_mfcc(audio: np.ndarray,
                 sr: int      = SAMPLE_RATE,
                 n_mfcc: int  = N_MFCC,
                 n_fft: int   = N_FFT,
                 hop: int     = HOP_LENGTH) -> np.ndarray:
    """
    Restituisce array (n_mfcc, frames) normalizzato.
    
    """
    audio = audio.astype(np.float32)

    # framing
    win      = get_window("hann", n_fft)
    n_frames = 1 + (len(audio) - n_fft) // hop
    frames   = np.stack([audio[i*hop : i*hop + n_fft] * win
                         for i in range(n_frames)], axis=0)   # (frames, n_fft)

    # power spectrum
    mag  = np.abs(np.fft.rfft(frames, n=n_fft))               # (frames, n_fft//2+1)
    pwr  = (1.0 / n_fft) * (mag ** 2)

    # mel filterbank
    mel  = pwr @ _FILTERBANK.T                                 # (frames, n_mels)
    mel  = np.where(mel == 0, np.finfo(float).eps, mel)
    log_mel = np.log(mel)

    # DCT → MFCC
    mfcc = dct(log_mel, type=2, axis=1, norm='ortho')[:, :n_mfcc]  # (frames, n_mfcc)
    mfcc = mfcc.T                                                    # (n_mfcc, frames)

    # normalizzazione per-clip
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
    return mfcc


def pad_or_trim(mfcc: np.ndarray, target_frames: int) -> np.ndarray:
    """Porta l'MFCC a lunghezza fissa (target_frames)."""
    if mfcc.shape[1] < target_frames:
        pad = target_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)))
    else:
        mfcc = mfcc[:, :target_frames]
    return mfcc

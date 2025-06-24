# mfcc_utils.py
import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import dct
import librosa
import librosa.filters

def load_audio(file_path):
    """
    Load and normalize audio file.
    """
    sr, signal = wav.read(file_path)
    if signal.ndim > 1:
        signal = signal[:, 0]  # Convert to mono if stereo
    signal = signal / np.max(np.abs(signal))  # Normalize
    return signal, sr

def frame_blocking(signal, frame_size=256, hop_size=100):
    """
    Split signal into overlapping frames.
    """
    num_frames = 1 + int((len(signal) - frame_size) / hop_size)
    frames = np.stack([
        signal[i * hop_size : i * hop_size + frame_size]
        for i in range(num_frames)
    ])
    return frames

def apply_window(frames):
    """
    Apply Hamming window to each frame.
    """
    window = np.hamming(frames.shape[1])
    return frames * window

def compute_fft(frames, NFFT=512):
    """
    Compute power spectrum of each frame.
    """
    return np.abs(np.fft.rfft(frames, NFFT))**2

def mel_filter_bank(sr, NFFT, n_mels=20):
    """
    Generate Mel filter bank matrix.
    """
    return librosa.filters.mel(sr=sr, n_fft=NFFT, n_mels=n_mels)

def apply_mel_filters(power_spectrum, mel_filters):
    """
    Apply Mel filter bank and log-compress the energies.
    """
    mel_energy = np.dot(power_spectrum, mel_filters.T)
    return np.log(mel_energy + 1e-9)

def compute_mfcc(mel_log_energy, num_ceps=13):
    """
    Compute MFCCs from log Mel energy using DCT.
    """
    mfccs = dct(mel_log_energy, type=2, axis=1, norm='ortho')
    return mfccs[:, 1:num_ceps+1]  # Skip the 0th coefficient

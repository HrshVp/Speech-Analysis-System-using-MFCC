
# app.py
import streamlit as st
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import datetime
import uuid
import librosa

from mfcc_utils import (
    load_audio, frame_blocking, apply_window,
    compute_fft, mel_filter_bank, apply_mel_filters,
    compute_mfcc
)

st.set_page_config(page_title="MFCC Speaker Recognition", page_icon="🎙️", layout="wide")

# Theme styling
st.markdown("""
    <style>
    .main { background-color: #f3f3f3; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .css-1rs6os.edgvbvh3 { background-color: #e8f0fe; }
    </style>
""", unsafe_allow_html=True)

# Sidebar - Student Info and Parameters
with st.sidebar:
    st.header("📘 Project Info")
    st.success("🎯 DSP Mini Project - Speaker Recognition")
    st.markdown("**Name:** Harshvardhan Patidar")
    st.markdown("**Scholar No:** 2311401217")
    st.markdown("**Dept:** ECE, MANIT Bhopal")
    st.markdown("**Topic:** MFCC-based Speaker Recognition")

    st.divider()
    st.subheader("⚙️ Parameters")

    frame_duration_ms = st.selectbox("Frame Duration (ms)", [20, 30, 50])
    overlap_percent = st.selectbox("Overlap (%)", [25, 50, 75])
    n_mels = st.selectbox("Number of Mel filters", [20, 30, 40])
    num_ceps = st.slider("MFCC coefficients", 8, 20, 13)
    resample_rate = st.selectbox("Resample Audio Rate (Hz)", [8000, 16000, 22050, 'Original'])

    st.divider()
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:6]
    st.caption(f"📅 Run ID: {run_id}")

st.title("🎤 MFCC Feature Extraction & Analysis")

uploaded_file = st.file_uploader("📁 Upload a .wav file", type=["wav"])

if uploaded_file:
    original_signal, sr = load_audio(uploaded_file)

    if resample_rate != 'Original':
        signal = librosa.resample(original_signal.astype(np.float32), orig_sr=sr, target_sr=int(resample_rate))
        sr = int(resample_rate)
    else:
        signal = original_signal

    st.audio(uploaded_file, format='audio/wav')

    # Compute frame size and hop size from ms
    frame_size = int(sr * (frame_duration_ms / 1000.0))
    hop_size = int(frame_size * (1 - overlap_percent / 100.0))

    frames = frame_blocking(signal, frame_size=frame_size, hop_size=hop_size)
    windowed = apply_window(frames)
    power_spectrum = compute_fft(windowed, NFFT=512)
    mel_filters = mel_filter_bank(sr, NFFT=512, n_mels=n_mels)
    mel_energy = apply_mel_filters(power_spectrum, mel_filters)
    mfccs = compute_mfcc(mel_energy, num_ceps=num_ceps)

    st.markdown("---")
    st.subheader("📊 Visualizations")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("📈 Time-Domain Waveform")
        fig1, ax1 = plt.subplots()
        times = np.linspace(0, len(signal)/sr, len(signal))
        ax1.plot(times, signal)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        st.pyplot(fig1)

    with col2:
        st.caption("🎚️ Power Spectrum (First Frame)")
        fig2, ax2 = plt.subplots()
        freqs = np.linspace(0, sr/2, power_spectrum.shape[1])
        ax2.plot(freqs, power_spectrum[0])
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Power")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    with col3:
        st.caption("🔍 Mel Filter Bank")
        fig3, ax3 = plt.subplots()
        for mel in mel_filters:
            ax3.plot(mel)
        ax3.set_title("Mel Filters")
        st.pyplot(fig3)

    with col4:
        st.caption("🎛️ MFCC Heatmap")
        fig4, ax4 = plt.subplots()
        librosa.display.specshow(mfccs.T, x_axis="time", sr=sr)
        ax4.set_title("MFCCs")
        st.pyplot(fig4)

    st.subheader("📌 Spectrogram vs MFCC Comparison")
    col5, col6 = st.columns(2)
    with col5:
        st.caption("🔊 Spectrogram")
        fig5, ax5 = plt.subplots()
        S = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
        img = librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='linear', ax=ax5)
        ax5.set_title("Spectrogram")
        fig5.colorbar(img, ax=ax5, format="%+2.0f dB")
        st.pyplot(fig5)

    with col6:
        st.caption("🎼 MFCC Heatmap")
        fig6, ax6 = plt.subplots()
        librosa.display.specshow(mfccs.T, x_axis="time", sr=sr)
        ax6.set_title("MFCCs")
        st.pyplot(fig6)

    st.success(f"✅ Processing completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
else:
    st.info("📂 Please upload a .wav file to begin analysis.")

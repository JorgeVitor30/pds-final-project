import time
import math
from collections import deque

import numpy as np
import streamlit as st


FS = 250                 
WINDOW_SECONDS = 5       
CHUNK_SECONDS = 0.10     
NOISE = 0.05             

def ecg_beat_template(fs=FS):
    beat_len = int(fs * 0.8)
    t = np.linspace(0, 1, beat_len, endpoint=False)

    def gauss(t, a, mu, sigma):
        return a * np.exp(-0.5 * ((t - mu) / sigma) ** 2)

    p  = gauss(t, 0.08, 0.18, 0.015)
    q  = gauss(t, -0.15, 0.30, 0.010)
    r  = gauss(t, 1.00, 0.33, 0.012)
    s  = gauss(t, -0.25, 0.36, 0.010)
    t_ = gauss(t, 0.35, 0.60, 0.040)

    beat = p + q + r + s + t_
    beat /= np.max(np.abs(beat) + 1e-8)
    return beat

def bpm_to_samples(bpm, fs=FS):
    rr_seconds = 60.0 / max(bpm, 1e-6)
    return max(int(rr_seconds * fs), 1)

class ECGSimulator:
    def __init__(self, fs=FS):
        self.fs = fs
        self.base_beat = ecg_beat_template(fs)
        self.phase = 0.0

    def generate_chunk(self, bpm, chunk_samples):
        rr = bpm_to_samples(bpm, self.fs)

        out = np.zeros(chunk_samples, dtype=np.float32)
        for i in range(chunk_samples):
            idx = int(self.phase * (len(self.base_beat) - 1))
            out[i] = self.base_beat[idx]

            self.phase += 1.0 / rr
            if self.phase >= 1.0:
                self.phase -= 1.0

        white = np.random.normal(0.0, NOISE, size=chunk_samples)
        t = np.arange(chunk_samples) / self.fs
        baseline = 0.02 * np.sin(2 * math.pi * 0.3 * t)
        muscle = np.random.normal(0.0, NOISE * 0.3, size=chunk_samples)
        muscle = np.convolve(muscle, np.ones(3)/3.0, mode="same")

        return out + white + baseline + muscle

st.set_page_config(page_title="ECG Tempo Real", layout="wide")

st.title("ECG Sintético — Execução Fluida e com Ruído Fixo")
st.caption("Geração automática com BPM aleatório entre 60 e 120.")

col1, col2 = st.columns([1, 3])

with col1:
    start = st.button("🚀 GERAR", use_container_width=True)
    stop = st.button("🛑 PARAR", use_container_width=True)

    if start:
        st.session_state.running = True
        st.session_state.bpm = int(np.random.randint(60, 121))
        st.session_state.buffer = deque(maxlen=FS * WINDOW_SECONDS) 
    
    if stop:
        st.session_state.running = False

with col2:
    chart_placeholder = st.empty()
    info_placeholder = st.empty()
if "running" not in st.session_state:
    st.session_state.running = False
if "buffer" not in st.session_state:
    st.session_state.buffer = deque(maxlen=FS * WINDOW_SECONDS)
if "sim" not in st.session_state:
    st.session_state.sim = ECGSimulator(FS)

if start:
    st.session_state.running = True
    st.session_state.bpm = int(np.random.randint(60, 121)) 
if stop:
    st.session_state.running = False

def run_loop():
    chunk_samples = int(FS * CHUNK_SECONDS)
    next_tick = time.perf_counter()

    while st.session_state.running:
        sig = st.session_state.sim.generate_chunk(
            bpm=st.session_state.bpm,
            chunk_samples=chunk_samples
        )

        st.session_state.buffer.extend(sig.tolist())

        y = np.array(st.session_state.buffer, dtype=np.float32)

        chart_placeholder.line_chart({"ECG (mV)": y}, height=350)

        info_placeholder.markdown(
            f"**BPM:** {st.session_state.bpm} | **Ruído:** {NOISE} | "
            f"**Janela:** {WINDOW_SECONDS}s | **Passo:** {CHUNK_SECONDS}s"
        )

        next_tick += CHUNK_SECONDS
        sleep_time = next_tick - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_tick = time.perf_counter()

if st.session_state.running:
    run_loop()
else:
    y = np.array(st.session_state.buffer if len(st.session_state.buffer) > 0 else [0])
    chart_placeholder.line_chart({"ECG (mV)": y}, height=350)

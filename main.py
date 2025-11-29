import time
import math
from collections import deque

import numpy as np
import streamlit as st
from scipy import signal


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

class PanTompkins:
    def __init__(self, fs=FS):
        self.fs = fs
        self.signal_filtered = None
        self.peaks = []
        self.rr_intervals = []
        self.bpm = 0
        
        # Pan-Tompkins parameters
        self.lowcut = 5.0
        self.highcut = 15.0
        self.moving_window = int(0.15 * fs) 
        self.refractory_period = int(0.2 * fs) 
        self.threshold_factor = 0.5
        
    def bandpass_filter(self, ecg_signal):
        """Apply bandpass filter (5-15 Hz)"""
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = signal.butter(2, [low, high], btype='band')
        return signal.filtfilt(b, a, ecg_signal)
    
    def derivative(self, ecg_signal):
        """Apply derivative filter"""
        return np.diff(ecg_signal, prepend=ecg_signal[0])
    
    def squaring(self, ecg_signal):
        """Square the signal"""
        return ecg_signal ** 2
    
    def moving_window_integration(self, ecg_signal):
        """Apply moving window integration"""
        window = np.ones(self.moving_window) / self.moving_window
        return np.convolve(ecg_signal, window, mode='same')
    
    def detect_peaks(self, integrated_signal):
        """Detect R-peaks using adaptive thresholding"""
        signal_max = np.max(integrated_signal)
        threshold = self.threshold_factor * signal_max
        
        self.peaks = []
        for i in range(len(integrated_signal)):
            if integrated_signal[i] > threshold:
                if i > 0 and i < len(integrated_signal) - 1:
                    if integrated_signal[i] >= integrated_signal[i-1] and integrated_signal[i] >= integrated_signal[i+1]:
                        if not self.peaks or (i - self.peaks[-1]) > self.refractory_period:
                            self.peaks.append(i)
                            recent_peaks = [p for p in self.peaks if (i - p) < int(2 * self.fs)]
                            if len(recent_peaks) > 0:
                                recent_values = [integrated_signal[p] for p in recent_peaks]
                                threshold = self.threshold_factor * np.mean(recent_values)
        
        return self.peaks
    
    def calculate_rr_intervals(self):
        """Calculate RR intervals and BPM"""
        if len(self.peaks) < 2:
            self.rr_intervals = []
            self.bpm = 0
            return
        
        self.rr_intervals = np.diff(self.peaks) / self.fs 
        
        if len(self.rr_intervals) > 0:
            self.bpm = 60.0 / np.mean(self.rr_intervals)
        else:
            self.bpm = 0
    
    def process(self, ecg_signal):
        filtered = self.bandpass_filter(ecg_signal)
        
        derivative = self.derivative(filtered)
        
        squared = self.squaring(derivative)
        
        integrated = self.moving_window_integration(squared)
        
        self.detect_peaks(integrated)
        
        self.calculate_rr_intervals()
        
        self.signal_filtered = filtered
        return integrated

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
if "pan_tompkins" not in st.session_state:
    st.session_state.pan_tompkins = PanTompkins(FS)

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
        
        if len(y) > 0:
            integrated = st.session_state.pan_tompkins.process(y)
            peaks = st.session_state.pan_tompkins.peaks
            detected_bpm = st.session_state.pan_tompkins.bpm
            
            chart_data = {"ECG (mV)": y}
            if peaks:
                peak_values = [y[p] if p < len(y) else 0 for p in peaks]
                chart_data["Picos R"] = [None] * len(y)
                for p, val in zip(peaks, peak_values):
                    if p < len(y):
                        chart_data["Picos R"][p] = val
        else:
            detected_bpm = 0
            chart_data = {"ECG (mV)": y}

        chart_placeholder.line_chart(chart_data, height=350)

        info_placeholder.markdown(
            f"**BPM Gerado:** {st.session_state.bpm} | **BPM Detectado:** {detected_bpm:.1f} | "
            f"**Picos:** {len(peaks) if peaks else 0} | "
            f"**Ruído:** {NOISE} | **Janela:** {WINDOW_SECONDS}s | **Passo:** {CHUNK_SECONDS}s"
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
    if len(y) > 1:  
        integrated = st.session_state.pan_tompkins.process(y)
        peaks = st.session_state.pan_tompkins.peaks
        detected_bpm = st.session_state.pan_tompkins.bpm
        
        chart_data = {"ECG (mV)": y}
        if peaks:
            peak_values = [y[p] if p < len(y) else 0 for p in peaks]
            chart_data["Picos R"] = [None] * len(y)
            for p, val in zip(peaks, peak_values):
                if p < len(y):
                    chart_data["Picos R"][p] = val
    else:
        detected_bpm = 0
        chart_data = {"ECG (mV)": y}
    
    chart_placeholder.line_chart(chart_data, height=350)

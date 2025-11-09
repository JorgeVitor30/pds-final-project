import sys
import time
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
)
from PyQt6.QtCore import QThread, pyqtSignal

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


def ecg_beat_template(fs=250):
    t = np.linspace(0, 1, fs, endpoint=False)
    g = lambda c, w, a: a * np.exp(-0.5 * ((t - c) / w) ** 2)

    beat = (
        g(0.18, 0.05, 0.12) +
        g(0.37, 0.02, -0.15) +
        g(0.40, 0.015, 1.0) +
        g(0.43, 0.015, -0.25) +
        g(0.60, 0.08, 0.35)
    )

    beat /= np.max(np.abs(beat))
    return beat


def generate_streaming_ecg(fs=250, hr=70,
                           noise_std=0.03,
                           emg_std=0.02,
                           powerline_amp=0.05,
                           mains_freq=60):

    beat = ecg_beat_template(fs)
    rr = int((60 / hr) * fs)
    t = 0

    while True:
        t_base = np.linspace(0, 1, len(beat), endpoint=False)
        t_rr = np.linspace(0, 1, rr, endpoint=False)
        stretched = np.interp(t_rr, t_base, beat)

        for sample in stretched:
            sample += np.random.normal(0, noise_std)
            sample += np.random.normal(0, emg_std) * np.random.uniform(-1, 1)
            sample += powerline_amp * np.sin(2 * np.pi * mains_freq * (t / fs))
            t += 1
            yield sample


class ECGThread(QThread):
    new_sample = pyqtSignal(float)

    def __init__(self, fs=50):
        super().__init__()
        self.fs = fs
        self.running = False

    def run(self):
        self.running = True
        generator = generate_streaming_ecg(fs=self.fs)

        while self.running:
            sample = next(generator)
            self.new_sample.emit(sample)
            time.sleep(1 / self.fs)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()


class ECGCanvas(FigureCanvasQTAgg):
    def __init__(self, fs=50, seconds=4):
        self.fig = Figure(figsize=(6, 2.8)) 
        super().__init__(self.fig)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylim(-2, 2)
        self.ax.set_title("ECG em Tempo Real", fontsize=9)
        self.ax.set_xlabel("Tempo", fontsize=8)

        self.buffer_size = fs * seconds
        self.buffer = np.zeros(self.buffer_size)

        self.line, = self.ax.plot(self.buffer, linewidth=1)


    def update_plot(self, new_value):
        self.buffer = np.roll(self.buffer, -1)
        self.buffer[-1] = new_value

        self.line.set_ydata(self.buffer)
        self.fig.canvas.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ECG Real-Time com PyQt")
        self.resize(1200, 700)

        main_layout = QHBoxLayout() 

        left_layout = QVBoxLayout()
        self.canvas = ECGCanvas(fs=50)
        self.thread = ECGThread(fs=50)
        self.thread.new_sample.connect(self.canvas.update_plot)

        self.button = QPushButton("Iniciar ECG")
        self.button.clicked.connect(self.toggle_ecg)

        left_layout.addWidget(self.canvas)
        left_layout.addWidget(self.button)

        # onde ficará o outro sinal (filtrado)
        right_layout = QVBoxLayout()
        # placeholder
        # você vai adicionar depois!
        # ex: self.filtered_canvas = ECGCanvas(...)

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def toggle_ecg(self):
        if not self.thread.isRunning():
            self.button.setText("Parar ECG")
            self.thread.start()
        else:
            self.button.setText("Iniciar ECG")
            self.thread.stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

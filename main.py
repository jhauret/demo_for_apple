# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Native macOS PyQt6 GUI for the Mimi real-time loopback demo.
# Displays a live spectrogram and allows switching between 3 modes:
#   [K] kyutai/moshiko-pytorch-bf16       (default Mimi, PyTorch/MPS)
#   [C] Cnam-LMSSC/mimi_throat_microphone (throat-mic fine-tune, PyTorch/MPS)
#   [P] Passthrough                       (no processing, matched latency)
#
# Requires: PyQt6, scipy, matplotlib, moshi (PyTorch), sounddevice, numpy, huggingface_hub
#
# Usage:
#   uv pip install PyQt6 scipy matplotlib
#   uv run python mimi_rt_mic_ui.py
#   uv run python mimi_rt_mic_ui.py --input-device 4 --output-device 5

import argparse
import collections
import queue
import threading
import time

import numpy as np
import scipy.signal
import sounddevice as sd
import torch
from huggingface_hub import hf_hub_download
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from moshi.models import loaders
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressDialog,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Audio / model constants
# ---------------------------------------------------------------------------
SR = 24_000
CHUNK = 1_920          # 80 ms at 24 kHz — model frame size
SD_BLOCK = 480         # PortAudio blocksize (20 ms); CHUNK must be a multiple
assert CHUNK % SD_BLOCK == 0
FORWARD_PAD_MS = 40.0  # pad every forward pass to this duration (ms)

MODELS = [
    {
        "name": "Cnam throat-mic",
        "repo": "Cnam-LMSSC/mimi_throat_microphone",
        "file": "kyutai_implementation.safetensors",
        "num_codebooks": 32,
    },
    {
        "name": "Bypass",
    },
]

PASSTHROUGH = 1

# Compute once — MPS availability doesn't change at runtime.
_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Spectrogram constants
# ---------------------------------------------------------------------------
HOP = 64
N_FFT = 1_024
PLOT_TIME_S = 4.0
N_FREQ = N_FFT // 2 + 1                       # 513
N_COLS = int(PLOT_TIME_S * SR / HOP)           # 1500  ← correct when carry buffer is used
# With carry buffer, scipy produces exactly CHUNK // HOP columns per chunk:
# extended = (N_FFT-1) + CHUNK = 2943 samples
# n_segs = (2943 - 1024) // 64 + 1 = 30 = CHUNK // HOP  ✓
# Without carry buffer it would be (CHUNK - N_FFT) // HOP + 1 = 15  → 2× too slow
assert CHUNK % HOP == 0, f"CHUNK ({CHUNK}) must be a multiple of HOP ({HOP})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_mimi(cfg: dict, device: str):
    path = hf_hub_download(cfg["repo"], cfg["file"])
    model = loaders.get_mimi(path, device=device)
    model.set_num_codebooks(cfg["num_codebooks"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Spectrogram worker thread
# ---------------------------------------------------------------------------

class SpectrogramThread(QThread):
    """Reads raw PCM chunks from audio_queue, computes spectrograms, and
    puts results into spec_queue for the Qt timer to consume.

    A carry buffer of (N_FFT - 1) samples is maintained so that each chunk
    of CHUNK samples produces exactly CHUNK // HOP = 30 spectrogram columns,
    making N_COLS × HOP / SR == PLOT_TIME_S exactly.
    """

    def __init__(self, audio_queue: queue.Queue, spec_queue: queue.Queue):
        super().__init__()
        self._audio_q = audio_queue
        self._spec_q = spec_queue
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        pre_emp_b = np.array([1.0, -0.97])  # numerator (float64 for lfilter)
        # zi: persistent pre-emphasis filter state across chunks.
        # lfilter with b=[1,-0.97], a=[1] has order 1 → zi shape (1,).
        # Resetting to zero each time run() is called ensures a clean start.
        zi = np.zeros(1)
        # Carry: last (N_FFT - 1) pre-emphasised samples of the previous chunk,
        # prepended to the next chunk so the STFT windows span the boundary cleanly.
        carry = np.zeros(N_FFT - 1, dtype=np.float32)

        while not self._stop:
            try:
                chunk = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # Pre-emphasis with persistent state → no discontinuity at chunk
            # boundaries, which would otherwise create artificial vertical lines
            # in the spectrogram when the signal is loud.
            chunk_emp, zi = scipy.signal.lfilter(pre_emp_b, [1.0], chunk, zi=zi)
            chunk_emp = chunk_emp.astype(np.float32)

            # Prepend carry so the STFT spans the full chunk:
            # extended length = (N_FFT-1) + CHUNK = 1023 + 1920 = 2943
            # → n_segs = (2943 - 1024) // 64 + 1 = 30 = CHUNK // HOP  ✓
            extended = np.concatenate([carry, chunk_emp])
            carry = chunk_emp[-(N_FFT - 1):].copy()

            # Short-time Fourier transform
            _, _, Sxx = scipy.signal.spectrogram(
                extended,
                fs=SR,
                nperseg=N_FFT,
                noverlap=N_FFT - HOP,
                window="hann",
            )  # shape: (N_FREQ, COLS_PER_FRAME) = (513, 30)

            # Convert to dB
            Sxx_db = 10.0 * np.log10(Sxx + 1e-9)

            try:
                self._spec_q.put_nowait(Sxx_db)
            except queue.Full:
                pass


# ---------------------------------------------------------------------------
# Audio thread
# ---------------------------------------------------------------------------

class AudioThread(threading.Thread):
    """Daemon thread that owns the sounddevice stream and model inference.
    Mirrors the run() loop from mimi_rt_mic_switch.py verbatim, with two
    additions:
      - pushes input PCM to audio_queue for the spectrogram
      - writes forward time into perf['avg_ms'] for the status bar
    """

    def __init__(
        self,
        mimi: list,
        audio_queue: queue.Queue,
        active: list,
        model_lock: threading.Lock,
        perf: dict,
        extra_latency_ms: list,
        input_device=None,
        output_device=None,
    ):
        super().__init__(daemon=True)
        self._mimi = mimi
        self._audio_q = audio_queue
        self._active = active
        self._lock = model_lock
        self._perf = perf
        self._extra_latency_ms = extra_latency_ms
        self._input_device = input_device
        self._output_device = output_device
        self.stop_event = threading.Event()

    def run(self):
        CHUNK_MS = CHUNK / SR * 1000  # 80.0 ms
        frame_times: collections.deque[float] = collections.deque(maxlen=25)
        output_deque: collections.deque = collections.deque()
        frame_count = 0

        try:
            with sd.Stream(
                samplerate=SR,
                channels=1,
                dtype="float32",
                blocksize=SD_BLOCK,
                latency="low",
                device=(self._input_device, self._output_device),
            ) as stream:
                while not self.stop_event.is_set():
                    pcm_in = np.empty((CHUNK, 1), dtype=np.float32)
                    for i in range(CHUNK // SD_BLOCK):
                        block, overflowed = stream.read(SD_BLOCK)
                        if overflowed:
                            print("Warning: input overflow", flush=True)
                        pcm_in[i * SD_BLOCK:(i + 1) * SD_BLOCK] = block

                    with self._lock:
                        cur = self._active[0]
                        t0 = time.perf_counter()
                        if cur == PASSTHROUGH:
                            out = np.clip(pcm_in, -1.0, 1.0)
                        else:
                            x = torch.from_numpy(pcm_in.T[np.newaxis]).to(_DEVICE)
                            with torch.no_grad():
                                codes = self._mimi[cur].encode(x)
                                pcm_out = self._mimi[cur].decode(codes)
                            out = np.clip(
                                pcm_out[0, 0].cpu().numpy(), -1.0, 1.0
                            )[:, None]
                        elapsed = (time.perf_counter() - t0) * 1000

                    # Pad to constant FORWARD_PAD_MS so output buffer refill cadence
                    # is deterministic — safe to use with latency='low'.
                    remaining = FORWARD_PAD_MS / 1000 - (time.perf_counter() - t0)
                    if remaining > 0:
                        time.sleep(remaining)

                    # Push model output to spectrogram thread
                    try:
                        self._audio_q.put_nowait(out[:, 0].copy())
                    except queue.Full:
                        pass

                    frame_times.append(elapsed)
                    frame_count += 1

                    # Update perf dict every 25 frames (~2 s)
                    if frame_count % 25 == 0:
                        self._perf["avg_ms"] = sum(frame_times) / len(frame_times)

                    # Extra-latency deque: buffer `n_buffered` chunks before playing.
                    output_deque.append(out)
                    n_buffered = round(self._extra_latency_ms[0] / CHUNK_MS)
                    # Trim excess chunks when slider was reduced (causes brief silence).
                    while len(output_deque) > n_buffered + 1:
                        output_deque.popleft()
                    if len(output_deque) > n_buffered:
                        to_play = output_deque.popleft()
                    else:
                        to_play = np.zeros((CHUNK, 1), dtype=np.float32)
                    stream.write(to_play)

        except Exception as exc:
            print(f"AudioThread error: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, input_device=None, output_device=None):
        super().__init__()
        self._input_device = input_device
        self._output_device = output_device

        self._active = [PASSTHROUGH]
        self._model_lock = threading.Lock()
        self._perf: dict = {"avg_ms": 0.0}
        self._extra_latency_ms: list = [0]
        self._audio_queue: queue.Queue = queue.Queue(maxsize=8)
        self._spec_queue: queue.Queue = queue.Queue(maxsize=32)
        self._audio_thread: AudioThread | None = None
        self._spec_thread: SpectrogramThread | None = None

        # Spectrogram rolling buffer
        self._spec_data = np.full((N_FREQ, N_COLS), -90.0, dtype=np.float32)

        # Will be populated by _load_models()
        self._mimi: list = []

        self.setWindowTitle("Real-Time Throat Mic Enhancement")
        self.resize(780, 600)

        self._build_ui()
        self._load_models()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        # --- Top bar: title + Enhance toggle + Start/Stop ---
        top = QHBoxLayout()
        title = QLabel("<b>Real-Time Throat Mic Enhancement</b>")
        title.setStyleSheet("font-size: 16px;")
        top.addWidget(title)
        top.addStretch()
        self._btn_enhance = QPushButton("Enhance")
        self._btn_enhance.setCheckable(True)
        self._btn_enhance.setChecked(False)  # starts in Bypass
        self._btn_enhance.setFixedWidth(110)
        self._btn_enhance.toggled.connect(
            lambda on: self._select_model(0 if on else PASSTHROUGH)
        )
        top.addWidget(self._btn_enhance)
        self._btn_start = QPushButton("Start")
        self._btn_start.setEnabled(False)  # enabled after load
        self._btn_start.setFixedWidth(90)
        self._btn_start.clicked.connect(self._toggle_stream)
        top.addWidget(self._btn_start)
        root.addLayout(top)

        # --- Matplotlib spectrogram ---
        self._fig = Figure(figsize=(7, 4), tight_layout=True)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setMinimumHeight(320)
        ax = self._fig.add_subplot(111)
        self._im = ax.imshow(
            self._spec_data,
            aspect="auto",
            origin="lower",
            vmin=-90,
            vmax=-58,
            cmap="viridis",
            extent=[-PLOT_TIME_S, 0, 0, SR / 2 / 1000],
        )
        self._fig.colorbar(self._im, ax=ax, label="Power (dB)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (kHz)")
        ax.set_xticks(np.arange(-PLOT_TIME_S, 1, 1))
        ax.set_xlim(-PLOT_TIME_S, 0)
        ax.set_ylim(0, SR / 2 / 1000)

        root.addWidget(self._canvas)

        # --- Additional latency slider ---
        latency_row = QHBoxLayout()
        latency_row.addWidget(QLabel("Additional latency:"))
        self._lbl_extra = QLabel("0 ms")
        self._lbl_extra.setFixedWidth(60)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(5000)
        self._slider.setSingleStep(80)
        self._slider.setPageStep(500)
        self._slider.setValue(0)
        self._slider.valueChanged.connect(self._on_latency_slider)
        latency_row.addWidget(self._slider)
        latency_row.addWidget(self._lbl_extra)
        root.addLayout(latency_row)

        # --- Status bar ---
        status_bar = QHBoxLayout()
        self._lbl_fwd = QLabel("Forward: — ms/frame")
        self._lbl_status = QLabel("Status: Idle")
        status_bar.addWidget(self._lbl_fwd)
        status_bar.addStretch()
        status_bar.addWidget(self._lbl_status)
        root.addLayout(status_bar)

        # --- Latency breakdown bar ---
        latency_bar = QHBoxLayout()
        self._lbl_latency = QLabel("E2E latency: —")
        self._lbl_latency.setStyleSheet("color: #888; font-size: 11px;")
        latency_bar.addWidget(self._lbl_latency)
        root.addLayout(latency_bar)

        # --- Plot refresh timer (40 ms ≈ 25 fps) ---
        self._plot_timer = QTimer(self)
        self._plot_timer.setInterval(40)
        self._plot_timer.timeout.connect(self._update_plot)
        self._plot_timer.start()

        # --- Perf label refresh timer (every 2 s) ---
        self._perf_timer = QTimer(self)
        self._perf_timer.setInterval(2_000)
        self._perf_timer.timeout.connect(self._update_perf_label)
        self._perf_timer.start()

    # ------------------------------------------------------------------
    # Model loading (runs in a QThread so UI stays responsive)
    # ------------------------------------------------------------------

    def _load_models(self):
        self._lbl_status.setText("Status: Loading models…")
        self._progress = QProgressDialog(
            "Downloading / loading models…", None, 0, len(MODELS) - 1, self
        )
        self._progress.setWindowTitle("Please wait")
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.setMinimumDuration(0)
        self._progress.setValue(0)

        self._loader_thread = _ModelLoaderThread()
        self._loader_thread.progress.connect(self._on_load_progress)
        self._loader_thread.done.connect(self._on_load_done)
        self._loader_thread.start()

    def _on_load_progress(self, value: int, name: str):
        self._progress.setValue(value)
        self._progress.setLabelText(f"Loading {name}…")

    def _on_load_done(self, mimi: list):
        self._mimi = mimi
        self._progress.close()
        self._lbl_status.setText("Status: Ready")
        self._btn_start.setEnabled(True)
        self._btn_start.setText("Start")

    # ------------------------------------------------------------------
    # Stream control
    # ------------------------------------------------------------------

    def _toggle_stream(self):
        if self._audio_thread is not None:
            self._stop_stream()
        else:
            self._start_stream()

    def _start_stream(self):
        if self._audio_thread is not None or not self._mimi:
            return

        # Drain stale data from previous run
        for q in (self._audio_queue, self._spec_queue):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        self._audio_thread = AudioThread(
            mimi=self._mimi,
            audio_queue=self._audio_queue,
            active=self._active,
            model_lock=self._model_lock,
            perf=self._perf,
            extra_latency_ms=self._extra_latency_ms,
            input_device=self._input_device,
            output_device=self._output_device,
        )
        self._spec_thread = SpectrogramThread(self._audio_queue, self._spec_queue)

        self._spec_thread.start()
        self._audio_thread.start()

        self._btn_start.setText("Stop")
        self._lbl_status.setText("Status: Running")

    def _stop_stream(self):
        if self._audio_thread is None:
            return

        self._audio_thread.stop_event.set()
        if self._spec_thread:
            self._spec_thread.stop()
            self._spec_thread.wait(3_000)

        self._audio_thread = None
        self._spec_thread = None
        self._btn_start.setText("Start")
        self._lbl_status.setText("Status: Stopped")

    # ------------------------------------------------------------------
    # Model switching
    # ------------------------------------------------------------------

    def _select_model(self, idx: int):
        with self._model_lock:
            if self._active[0] != PASSTHROUGH and self._mimi:
                self._mimi[self._active[0]].reset_streaming()
            self._active[0] = idx
            if idx != PASSTHROUGH and self._mimi:
                self._mimi[idx].reset_streaming()

        self._btn_enhance.setChecked(idx != PASSTHROUGH)

    def _switch_model(self):
        """Toggle Enhance on/off (Space key shortcut)."""
        self._select_model(PASSTHROUGH if self._active[0] != PASSTHROUGH else 0)

    def _on_latency_slider(self, value: int):
        self._extra_latency_ms[0] = value
        self._lbl_extra.setText(f"{value} ms" if value < 1000 else f"{value/1000:.1f} s")

    # ------------------------------------------------------------------
    # Plot update (called by QTimer every 40 ms)
    # ------------------------------------------------------------------

    def _update_plot(self):
        updated = False
        while True:
            try:
                cols = self._spec_queue.get_nowait()  # shape (N_FREQ, k)
            except queue.Empty:
                break
            k = cols.shape[1]
            self._spec_data = np.roll(self._spec_data, -k, axis=1)
            self._spec_data[:, -k:] = cols
            updated = True

        if updated:
            self._im.set_data(self._spec_data)
            self._canvas.draw_idle()

    def _update_perf_label(self):
        avg = self._perf.get("avg_ms", 0.0)
        if avg > 0:
            self._lbl_fwd.setText(f"Forward: {avg:.1f} ms/frame")
            buf_ms = CHUNK / SR * 1000  # 80 ms
            out_wait_ms = SD_BLOCK / SR * 1000  # ~20 ms
            extra_ms = self._extra_latency_ms[0]
            total = buf_ms + FORWARD_PAD_MS + out_wait_ms + extra_ms
            label = (
                f"E2E ≈  in buf {buf_ms:.0f} ms  +  "
                f"forward {avg:.0f} ms (padded to {FORWARD_PAD_MS:.0f})  +  "
                f"out wait ~{out_wait_ms:.0f} ms"
            )
            if extra_ms > 0:
                label += f"  +  extra {extra_ms:.0f} ms"
            label += f"  =  {total:.0f} ms"
            self._lbl_latency.setText(label)

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Space, Qt.Key.Key_S):
            self._switch_model()
        elif event.key() in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            self.close()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._plot_timer.stop()
        self._perf_timer.stop()
        self._stop_stream()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Model loader worker (QThread so the Qt event loop stays responsive)
# ---------------------------------------------------------------------------

class _ModelLoaderThread(QThread):
    progress = pyqtSignal(int, str)
    done = pyqtSignal(list)

    def run(self):
        mimi = []
        for cfg in MODELS:
            if "repo" not in cfg:
                continue
            self.progress.emit(len(mimi), cfg["name"])
            model = load_mimi(cfg, _DEVICE)
            mimi.append(model)

        # Warmup
        self.progress.emit(len(mimi), "warmup")
        dummy = torch.zeros(1, 1, CHUNK, device=_DEVICE)
        for m in mimi:
            with torch.no_grad(), m.streaming(1):
                m.decode(m.encode(dummy))

        # Enter streaming mode permanently
        for m in mimi:
            m.streaming_forever(1)

        self.done.emit(mimi)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mimi real-time demo (PyQt6 GUI)")
    parser.add_argument("--input-device", type=int, default=None)
    parser.add_argument("--output-device", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=480,
                        help="PortAudio blocksize in samples (default 480 = 20ms)")
    args = parser.parse_args()

    global SD_BLOCK
    SD_BLOCK = args.block_size
    assert CHUNK % SD_BLOCK == 0, f"CHUNK ({CHUNK}) must be a multiple of --block-size ({SD_BLOCK})"

    app = QApplication([])
    app.setApplicationName("Real-Time Throat Mic Enhancement")

    win = MainWindow(
        input_device=args.input_device,
        output_device=args.output_device,
    )
    win.show()
    app.exec()


if __name__ == "__main__":
    main()

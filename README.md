# Mimi Real-Time Codec Demo

## 1. Clone the moshi repo (for the `moshi_mlx` package)

```bash
git clone https://github.com/kyutai-labs/moshi.git
```

## 2. Create a virtual environment and install dependencies

```bash
uv venv --python 3.11
source .venv/bin/activate


uv pip install mlx
uv pip install -e /path/to/moshi/moshi_mlx
uv pip install sounddevice numpy huggingface_hub torchaudio datasets moshi
```

> `sounddevice` wraps PortAudio. If it fails to install, first run:
> ```bash
> brew install portaudio
> ```

## 3. Check your audio device

The script runs at **24 000 Hz, mono**. Verify your default input/output device supports it:

```bash
uv run python -c "import sounddevice; print(sounddevice.query_devices())"
```

If your device does not natively support 24 kHz, macOS Core Audio will resample transparently, but you may notice slightly higher latency.

## 4. Run

```bash
uv run python mimi_rt_mic.py
```

---

## Variant: Throat-Microphone Fine-Tuned Weights (PyTorch/MPS)

`mimi_rt_mic_torch.py` uses the same loopback structure but loads the throat-microphone
fine-tuned weights from [`Cnam-LMSSC/mimi_throat_microphone`](https://huggingface.co/Cnam-LMSSC/mimi_throat_microphone)
via the **PyTorch + MPS** backend (Apple Silicon GPU).

| | `mimi_rt_mic.py` | `mimi_rt_mic_torch.py` |
|---|---|---|
| Backend | MLX | PyTorch / MPS |
| Weights | `kyutai/moshiko-mlx-q4` (quantized) | `Cnam-LMSSC/mimi_throat_microphone` (float32) |
| Forward time (M4 Pro) | ~16 ms/frame | varies |
| Use case | General speech | Throat-microphone / body-conducted speech |

### Additional setup

Install the PyTorch `moshi` package (only needed for the torch variant):

```bash
uv pip install -e /path/to/moshi/moshi
```

### Run

```bash
uv run python mimi_rt_mic_torch.py
uv run python mimi_rt_mic_torch.py --input-device 4 --output-device 5
uv run python mimi_rt_mic_torch.py --model-file /path/to/kyutai_implementation.safetensors
```

---

## GUI with Spectrogram (native macOS)

`mimi_rt_mic_ui.py` is a native macOS PyQt6 app that wraps `mimi_rt_mic_switch.py`'s
audio/model logic and adds a live scrolling spectrogram.

### Features
- Live spectrogram with correct time axis (−8 → 0 s) and frequency axis (0 → 12 kHz)
- Three model buttons: **Kyutai default**, **Cnam throat-mic**, **Bypass**
- `Space` key cycles through models (same shortcut as the console version)
- Forward latency label updates every 2 seconds
- Start / Stop button; models load with a progress dialog so the window never freezes

### Additional dependencies

```bash
uv pip install PyQt6 scipy matplotlib
```

### Run

```bash
uv run python mimi_rt_mic_ui.py
uv run python mimi_rt_mic_ui.py --input-device 4 --output-device 5
```

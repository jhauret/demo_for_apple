# Mimi Real-Time Demo (macOS)

Live loopback demo with a throat-microphone enhancement model and a scrolling spectrogram.
Runs on Apple Silicon via PyTorch/MPS.

## Setup

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install torch sounddevice numpy huggingface_hub PyQt6 scipy matplotlib moshi
```

> If `sounddevice` fails to install, first run `brew install portaudio`.

## Run

```bash
uv run python main.py
uv run python main.py --input-device 4 --output-device 5
```

List available audio devices:

```bash
uv run python -c "import sounddevice; print(sounddevice.query_devices())"
```

## Usage

- **Enhance** — toggle the Cnam throat-mic model on/off (Space key)
- **Start / Stop** — start or stop the audio stream
- Models are downloaded automatically from Hugging Face on first launch

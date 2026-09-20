# sleepy-accent 🎙️😴

> *"you can sleep peacefully in your lectures now"*

> [!WARNING]
> **Academic Policy & Recording Consent Notice**:
> This tool is intended strictly for personal study, review, and accessibility note-taking. Before recording audio in lectures, ensure compliance with your university's academic regulations, student code of conduct, and regional privacy laws. Always obtain explicit permission or consent from your course instructor prior to recording in-class sessions.

A lightweight, terminal-native live lecture recorder and real-time speech transcriber. Captures audio directly from your laptop microphone, streams live transcriptions with timestamps to your console, auto-saves every sentence to disk in real-time, and generates structured post-lecture study recaps (definitions, formulas, and exam takeaways).

---

## ⚡ Features

* **Mic Audio Capture & Auto-Archiving:** Streams 16kHz PCM audio while continuously saving a backup `.wav` file (`lectures/audio/`).
* **Accent & Rapid-Pacing Resilient:** Pre-calibrates ambient room noise (HVAC humming, typing, hall chatter) using dynamic energy thresholds.
* **Continuous Real-Time Flush:** Writes incoming sentences directly to markdown (`lectures/notes/`) as they are spoken—no notes are lost if your battery dies.
* **Post-Lecture Synthesis:** Synthesizes clean study recaps with concept extraction, formula detection, and exam takeaways via Gemini, OpenAI, Ollama, or local heuristic fallback.
* **Zero Bloat & Battery Friendly:** Pure Python background threading without heavy GPU dependencies.

---

## 📦 Architecture

```
sleepy-accent/
├── recorder.py       # Background mic capture & parallel audio file writer
├── transcriber.py    # Speech-to-text worker with noise calibration & accent handling
├── session.py        # Live session coordinator & real-time markdown logger
├── summarizer.py     # Post-lecture cleaner & study note synthesizer
├── main.py           # Single interactive CLI entrypoint
├── tests/            # Automated test suite (pytest)
└── lectures/
    ├── audio/        # Raw .wav audio recordings
    └── notes/        # Live transcript logs & generated study recaps
```

---

## 🚀 Quick Start

### 1. Prerequisites
Python 3.10+ with `pyaudio` and `speechrecognition`:

```bash
pip install pyaudio SpeechRecognition
```

*(Optional: for enhanced terminal styling)*
```bash
pip install colorama
```

### 2. Run in Class

```bash
python main.py
```

1. Enter your lecture name (e.g. `CIS2520 - Data Structures`) or press **Enter** for default timestamp.
2. Press **Enter** to calibrate room noise and start listening.
3. Transcriptions will stream live onto your terminal with timestamps:
   ```
   [00:04:12] Professor: Today we are discussing balanced AVL trees and rotation operations.
   [00:04:35] Professor: Notice that the height difference between subtrees can never exceed one.
   ```
4. Press **`q`** or **`Ctrl+C`** when class ends.
5. `sleepy-accent` will instantly finalize your raw transcript and generate a structured study recap.

---

## ⚙️ CLI Options

```bash
python main.py --help
```

| Flag | Description | Default |
|---|---|---|
| `-c, --course` | Course or lecture title | Interactive prompt |
| `-s, --speaker` | Speaker prefix in live log | `Professor` |
| `-o, --output-dir` | Directory for audio and notes | `lectures` |
| `-l, --language` | Speech recognition language code | `en-US` |
| `--calibrate-sec`| Duration (seconds) for ambient noise calibration | `1.5` |
| `-p, --provider` | Recap LLM provider (`auto`, `gemini`, `openai`, `ollama`, `heuristic`) | `auto` |
| `--no-recap` | Skip generating study summary at the end | `False` |
| `-y, --yes` | Skip confirmation and start listening immediately | `False` |

---

## 🧪 Running Tests

```bash
pytest
```

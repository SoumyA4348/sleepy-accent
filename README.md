# sleepy-accent 🎙️⚡

> Real-time speech transcription and automated note-taking for lectures, talks, and presentations.

> [!WARNING]
> **Recording Consent & Session Privacy Notice**:
> This tool is designed for authorized transcription, personal note-taking, and accessibility during public speeches, lectures, and open presentations. Always ensure compliance with local recording consent regulations and event speaker guidelines prior to recording.

A lightweight Python tool that listens through your laptop or external microphone, prints live transcriptions to your terminal with timestamps, automatically saves everything to markdown files, and generates a clean bullet-point summary so you don't miss anything.

Built to solve a real everyday problem: when sitting in a large lecture hall listening to a speaker who talks fast or has an unfamiliar accent, it's easy to miss key points—especially when you're tired. `sleepy-accent` boosts quiet audio, transcribes speech live, and gives you clean notes.

---

## ⚡ Features

* **Live Mic Capture & Backup:** Streams audio from your microphone and saves an uncompressed `.wav` backup in the background (`sessions/audio/`).
* **Built-in Voice Booster:** Automatically boosts quiet or distant voices (up to 8.0x gain) so you can still capture audio from the back of the room.
* **Fast Local Transcription:** Uses `faster-whisper` locally for quick speech-to-text with timestamped console output—without needing a powerful GPU.
* **Auto-Save Markdown Logs:** Flushes text to disk (`sessions/notes/`) phrase by phrase in real time, so nothing is lost if your laptop battery dies.
* **Clean Session Recaps:** Summarizes the full session transcript into key points, definitions, and action items using Gemini API (with a local fallback if offline).
* **Lightweight & Battery-Friendly:** Runs in background Python threads with low CPU usage so your laptop stays cool and quiet during long sessions.

---

## 📦 Project layout

```
sleepy-accent/
├── src/
│   └── sleepy_accent/
│       ├── main.py       # Interactive CLI entrypoint
│       ├── recorder.py   # Background microphone capture
│       ├── session.py    # Session coordinator and markdown logger
│       ├── summarizer.py  # Study-note generator
│       └── transcriber.py # Speech-to-text worker
├── tests/                # Automated test suite
├── pyproject.toml        # Packaging, CLI, and test configuration
├── requirements.txt      # Direct pip-install requirements
├── .github/workflows/    # Continuous integration
└── sessions/
    ├── audio/        # Raw .wav audio recordings (full backup)
    ├── notes/        # Live transcript logs
    └── study_notes/  # Generated study summaries
```

Session output directories are intentionally ignored by Git because they may contain
private recordings, transcripts, or notes.

---

## 🚀 Quick Start

### 1. Install

Python 3.10+ is required. For normal use, install the package in editable mode:

```bash
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
```

The legacy requirements file remains available for environments that use
`pip install -r requirements.txt`.

### 2. Live Capture at a Talk or Lecture

```bash
sleepy-accent
```

You can also run the CLI without installing the console script:

```bash
python -m sleepy_accent
```

1. Enter your session name (e.g. `Algorithms Lecture - Week 4`) or press **Enter** for a default timestamped title.
2. Press **Enter** to calibrate ambient room noise and begin live capture.
3. Transcriptions stream live onto your terminal with session timestamps:
   ```
   [00:04:12] Speaker: Today we are going to cover graph traversal and Dijkstra's algorithm.
   [00:04:35] Speaker: Make sure to notice how priority queues change the runtime complexity.
   ```
4. Press **`q`** or **`Ctrl+C`** when the session concludes.
5. `sleepy-accent` saves your full transcript and generates a clean summary with key points and definitions.

### 3. Transcribe an Existing Audio File

Transcribe pre-recorded lecture or conference audio directly (`.wav`, `.mp3`, etc.) without microphone recording:

```bash
sleepy-accent --transcribe-file path/to/recording.wav
```

You can customize the model size, language, or session title:
```bash
sleepy-accent --transcribe-file keynote.wav --whisper-model small.en --session "AI Keynote"
```

---

## ⚙️ CLI Options

```bash
sleepy-accent --help
```

| Flag | Description | Default |
|---|---|---|
| `-c, --session` | Session or event title (e.g. `'AI Summit Keynote'`) | Interactive prompt |
| `-s, --speaker` | Speaker prefix label in live log | `Speaker` |
| `-o, --output-dir` | Directory for audio recordings and notes | `sessions` |
| `-l, --language` | Speech recognition language code | `en-US` |
| `--engine` | Recognition engine: `whisper` (local), `google`, or `auto` | `auto` |
| `--whisper-model` | Model size for faster-whisper (`tiny.en`, `base.en`, `small.en`, `medium.en`) | `base.en` |
| `--refine` | Perform post-session full audio refinement pass with Whisper on saved WAV | `False` |
| `--transcribe-file` | Transcribe an existing audio file (`.wav`, `.mp3`, etc.) directly without recording | `None` |
| `--calibrate-sec` | Duration (seconds) for ambient hall noise calibration | `1.5` |
| `--max-gain` | Maximum AGC gain multiplier for distant podium audio | `8.0` |
| `--no-boost` | Disable automatic gain boost for close/front-row audio | `False` |
| `-p, --provider` | Recap LLM provider (`auto`, `gemini`, `openai`, `ollama`, `heuristic`) | `auto` |
| `--no-recap` | Skip generating summary at conclusion | `False` |
| `-y, --yes` | Skip confirmation and start listening immediately | `False` |

---

## 🧪 Running Tests

```bash
python -m pytest
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# sleepy-accent 🎙️⚡

> *"Real-time speech capture & intelligence for public speeches, conference keynotes, and high-stakes sessions."*

> [!WARNING]
> **Recording Consent & Session Privacy Notice**:
> This tool is designed for authorized transcription, personal note-taking, and accessibility during public speeches, conferences, and open sessions. For private or Chatham House Rule sessions, always ensure compliance with event policies, local recording consent regulations, and speaker privacy guidelines prior to recording.

A lightweight, terminal-native live speech recorder and real-time audio transcriber. Captures spoken audio directly from your laptop or external microphone, streams live transcriptions with timestamps to your console, auto-saves every phrase to disk in real-time, and synthesizes structured executive session recaps (core arguments, key announcements, metrics, and actionable takeaways).

---

## ⚡ Features

* **Live Stage Audio Capture & Auto-Archiving:** Streams 16kHz PCM audio while continuously saving an uncompressed backup `.wav` file (`sessions/audio/`).
* **Global Accent & Fast-Cadence Resilient:** Purpose-built to decipher rapid delivery and diverse international accents across global conferences and technical keynotes.
* **Auditorium Distance & Voice Booster:** Built-in dynamic Automatic Gain Control (AGC) and DC-offset filtering that amplifies distant stage or podium voices (up to 8.0x / +18 dB) without clipping.
* **Continuous Real-Time Flush:** Writes incoming phrases directly to markdown (`sessions/notes/`) as they are spoken—zero data loss if your laptop sleeps or battery dies.
* **Executive Recap Synthesis:** Automatically parses raw transcripts into polished briefings, extracting key announcements, strategic insights, definitions, and action points via Gemini, OpenAI, Ollama, or local heuristic fallback.
* **Zero Bloat & Battery Friendly:** Pure Python background threading without heavy GPU dependencies—keeps your laptop cold and silent in quiet auditoriums.

---

## 📦 Architecture

```
sleepy-accent/
├── recorder.py       # Background mic capture & parallel audio file writer
├── transcriber.py    # Speech-to-text worker with noise calibration, AGC & accent handling
├── session.py        # Live session coordinator & real-time markdown logger
├── summarizer.py     # Post-speech cleaner & executive briefing synthesizer
├── main.py           # Single interactive CLI entrypoint
├── requirements.txt  # Project dependencies
├── tests/            # Automated test suite (pytest)
└── sessions/
    ├── audio/        # Raw .wav audio recordings (full backup)
    └── notes/        # Live transcript logs & generated executive recaps
```

---

## 🚀 Quick Start

### 1. Prerequisites
Python 3.10+ installed. Install dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

*(Or install core packages manually: `pip install pyaudio SpeechRecognition faster-whisper colorama pytest`)*

### 2. Live Capture at an Event

```bash
python main.py
```

1. Enter your event/session name (e.g. `Tech Summit 2026 - AI Keynote`) or press **Enter** for a default timestamped title.
2. Press **Enter** to calibrate auditorium ambient noise and begin live capture.
3. Transcriptions stream live onto your terminal with session timestamps:
   ```
   [00:04:12] Speaker: Today we are announcing our next-generation distributed inference engine.
   [00:04:35] Speaker: The primary benchmark demonstrates a 3.4x reduction in per-token latency.
   ```
4. Press **`q`** or **`Ctrl+C`** when the session concludes.
5. `sleepy-accent` instantly finalizes your transcript and generates an executive session summary.

### 3. Transcribe an Existing Audio File

Transcribe pre-recorded lecture or conference audio directly (`.wav`, `.mp3`, etc.) without microphone recording:

```bash
python main.py --transcribe-file path/to/recording.wav
```

You can customize the model size, language, or session title:
```bash
python main.py --transcribe-file keynote.wav --whisper-model small.en --session "AI Keynote"
```

---

## ⚙️ CLI Options

```bash
python main.py --help
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
| `--no-recap` | Skip generating executive recap at conclusion | `False` |
| `-y, --yes` | Skip confirmation and start listening immediately | `False` |

---

## 🧪 Running Tests

```bash
pytest
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

"""Live Session & Auto-Saver (session.py)

Orchestrates real-time lecture audio capture and speech-to-text transcription:
  - Prints live timestamps and spoken sentences to console as the professor speaks.
  - Continuously flushes each sentence to lectures/lecture_<timestamp>.md so no
    notes are lost if your battery dies.
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Optional, TextIO

from recorder import AudioRecorder
from transcriber import LiveTranscriber


class LiveSession:
    """Manages an active lecture recording and live transcription session,

    streaming spoken sentences to the console and auto-saving them to markdown.
    """

    def __init__(
        self,
        output_dir: str | Path = "lectures",
        sample_rate: int = 16000,
        language: str = "en-US",
        calibration_duration: float = 1.5,
    ):
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.language = language
        self.calibration_duration = calibration_duration

        self.audio_queue: queue.Queue = queue.Queue()
        self.recorder = AudioRecorder(
            output_dir=self.output_dir,
            sample_rate=self.sample_rate,
            audio_queue=self.audio_queue,
        )
        self.transcriber = LiveTranscriber(
            audio_queue=self.audio_queue,
            sample_rate=self.sample_rate,
            language=self.language,
            calibration_duration=self.calibration_duration,
            on_text=self._handle_sentence,
        )

        self.wav_path: Optional[Path] = None
        self.md_path: Optional[Path] = None
        self._md_file: Optional[TextIO] = None

        self._lock = threading.Lock()
        self._is_running = False
        self._start_time: Optional[float] = None
        self._start_datetime: Optional[datetime] = None
        self.sentence_count: int = 0

    @property
    def is_running(self) -> bool:
        return self._is_running

    def _handle_sentence(self, sentence: str) -> None:
        """Callback invoked immediately when a sentence is transcribed."""
        sentence = sentence.strip()
        if not sentence:
            return

        now = datetime.now()
        timestamp_str = now.strftime("%H:%M:%S")

        with self._lock:
            self.sentence_count += 1

            # 1. Print live timestamp and spoken sentence to console
            print(f"[{timestamp_str}] {sentence}", flush=True)

            # 2. Continuously flush each sentence to disk so notes are preserved
            if self._md_file and not self._md_file.closed:
                self._md_file.write(f"- **[{timestamp_str}]** {sentence}\n")
                self._md_file.flush()
                try:
                    os.fsync(self._md_file.fileno())
                except OSError:
                    pass

    def start(self) -> tuple[Path, Path]:
        """Starts audio recording, initializes auto-saved markdown file,

        calibrates ambient noise, and begins live transcription.
        """
        if self._is_running:
            raise RuntimeError("LiveSession is already running.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._start_datetime = datetime.now()
        self._start_time = time.time()
        self.sentence_count = 0

        # Start audio recording
        self.wav_path = self.recorder.start()

        # Match markdown note file to the recording's timestamped name
        self.md_path = self.wav_path.with_suffix(".md")

        # Initialize markdown file with session metadata header
        self._md_file = open(self.md_path, "w", encoding="utf-8")
        header_text = (
            f"# Lecture Notes - {self._start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"- **Audio Recording:** `{self.wav_path.name}`\n"
            f"- **Date:** {self._start_datetime.strftime('%A, %B %d, %Y')}\n"
            f"- **Session Started:** {self._start_datetime.strftime('%H:%M:%S')}\n"
            f"- **Language:** {self.language}\n\n"
            f"---\n\n"
            f"## Live Transcript\n\n"
        )
        self._md_file.write(header_text)
        self._md_file.flush()
        try:
            os.fsync(self._md_file.fileno())
        except OSError:
            pass

        self._is_running = True

        # Calibrate for ambient lecture hall noise
        self.transcriber.calibrate(duration=self.calibration_duration)

        # Start background transcription threads
        self.transcriber.start()

        return self.wav_path, self.md_path

    def stop(self) -> tuple[Optional[Path], Optional[Path]]:
        """Gracefully stops transcription and recording, appends a session

        summary to the markdown file, flushes to disk, and closes the file.
        """
        if not self._is_running:
            return self.wav_path, self.md_path

        self._is_running = False

        # Stop transcriber and audio recorder
        self.transcriber.stop()
        final_wav = self.recorder.stop()

        end_datetime = datetime.now()
        duration_sec = int(time.time() - self._start_time) if self._start_time else 0
        duration_str = time.strftime("%H:%M:%S", time.gmtime(duration_sec))

        # Write summary footer and flush
        with self._lock:
            if self._md_file and not self._md_file.closed:
                footer_text = (
                    f"\n---\n\n"
                    f"## Session Summary\n\n"
                    f"- **Session Ended:** {end_datetime.strftime('%H:%M:%S')}\n"
                    f"- **Duration:** {duration_str}\n"
                    f"- **Total Sentences Captured:** {self.sentence_count}\n"
                )
                self._md_file.write(footer_text)
                self._md_file.flush()
                try:
                    os.fsync(self._md_file.fileno())
                except OSError:
                    pass
                self._md_file.close()

        return final_wav, self.md_path

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Live Session & Auto-Saver: Real-time lecture transcription and continuous auto-save."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="lectures",
        help="Directory to save WAV audio and Markdown lecture notes (default: lectures)",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default="en-US",
        help="Speech recognition language code (default: en-US)",
    )
    parser.add_argument(
        "--calibrate-sec",
        "-c",
        type=float,
        default=1.5,
        help="Ambient noise calibration duration in seconds (default: 1.5)",
    )

    args = parser.parse_args()

    session = LiveSession(
        output_dir=args.output_dir,
        language=args.language,
        calibration_duration=args.calibrate_sec,
    )

    print("=" * 64)
    print("        Sleepy Accent: Live Lecture Session & Auto-Saver        ")
    print("=" * 64)

    print("\n[1/3] Initializing microphone and auto-saver...")
    wav_path, md_path = session.start()
    print(f"      • Audio Target:    {wav_path}")
    print(f"      • Auto-Save Notes: {md_path}")

    print(
        f"\n[2/3] Noise calibration complete:\n"
        f"      • Ambient Baseline RMS: {session.transcriber.ambient_energy:.1f}\n"
        f"      • Speech Trigger RMS:   {session.transcriber.speech_threshold:.1f}"
    )

    print(
        "\n[3/3] Live session active! Transcribing and continuously saving...\n"
        "      (Each sentence is flushed immediately to disk)\n"
        "      Press Ctrl+C when lecture finishes.\n"
        + "-" * 64
    )

    try:
        while session.is_running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n" + "-" * 64)
        print("Ending lecture session gracefully...")
    finally:
        saved_wav, saved_md = session.stop()
        duration_sec = int(time.time() - session._start_time) if session._start_time else 0
        duration_str = time.strftime("%H:%M:%S", time.gmtime(duration_sec))

        print("\n" + "=" * 64)
        print("                   Session Complete Summary                     ")
        print("=" * 64)
        print(f"• Audio Saved:        {saved_wav}")
        print(f"• Markdown Notes:     {saved_md}")
        print(f"• Sentences Captured: {session.sentence_count}")
        print(f"• Session Duration:   {duration_str}")
        print("=" * 64)


if __name__ == "__main__":
    main()

"""Live Transcriber (transcriber.py)

Reads audio frames from an in-memory queue and converts speech to text continuously.
Automatically calibrates for lecture hall background noise (HVAC, distant chatter,
keyboard typing) using adaptive energy thresholds, dynamic noise-floor tracking,
and transient impulse rejection.
"""

from collections import deque
import inspect
import logging
from pathlib import Path
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np
import speech_recognition as sr

# Optional import of AudioRecorder for standalone / integration mode
try:
    from recorder import AudioRecorder
except ImportError:
    AudioRecorder = None

logger = logging.getLogger(__name__)


class Phrase(tuple):
    """Tuple of (timestamp_offset, text) with attribute access and string representation."""

    def __new__(cls, timestamp_offset: str, text: str):
        return super().__new__(cls, (timestamp_offset, text))

    @property
    def timestamp_offset(self) -> str:
        return self[0]

    @property
    def text(self) -> str:
        return self[1]

    def __str__(self) -> str:
        return self[1]


def compute_rms(audio_chunk: bytes) -> float:
    """Computes Root Mean Square (RMS) energy for 16-bit PCM mono audio."""
    if not audio_chunk:
        return 0.0
    samples = np.frombuffer(audio_chunk, dtype=np.int16)
    if len(samples) == 0:
        return 0.0
    # Use float32 to prevent overflow during squaring
    mean_sq = np.mean(samples.astype(np.float32) ** 2)
    return float(np.sqrt(mean_sq))


class LiveTranscriber:
    """Consumes audio segments from an in-memory queue and transcribes speech

    continuously with lecture hall background noise adaptation.
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        sample_rate: int = 16000,
        sample_width: int = 2,
        chunk_size: int = 1024,
        calibration_duration: float = 1.5,
        energy_ratio: float = 1.7,
        min_speech_duration: float = 0.35,
        pause_threshold: float = 0.85,
        max_phrase_duration: float = 12.0,
        language: str = "en-US",
        on_text: Optional[Callable] = None,
        on_phrase: Optional[Callable[[str, str], None]] = None,
        session_start_time: Optional[float] = None,
    ):
        """Args:

        audio_queue: Queue delivering raw 16-bit PCM mono bytes chunks.
        sample_rate: Sampling frequency in Hz (default: 16000).
        sample_width: Bytes per sample (2 for 16-bit PCM).
        chunk_size: Samples per audio buffer (default: 1024).
        calibration_duration: Seconds of ambient audio used for initial noise calibration.
        energy_ratio: Multiplier above ambient noise floor required to trigger speech.
                      Filters continuous HVAC hum and distant low-level room chatter.
        min_speech_duration: Minimum sustained speech duration in seconds.
                             Spikes shorter than this (e.g. keyboard typing clicks) are rejected.
        pause_threshold: Seconds of silence marking the end of a spoken phrase.
        max_phrase_duration: Maximum utterance length in seconds before segmenting.
        language: BCP-47 language tag for transcription (default: 'en-US').
        on_text: Optional callback invoked with transcribed phrase/text.
        on_phrase: Optional callback invoked with (timestamp_offset, text).
        session_start_time: Optional session start epoch time (time.time()) for computing offsets.
        """
        self.audio_queue = audio_queue
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.chunk_size = chunk_size
        self.chunk_duration = chunk_size / sample_rate

        self.calibration_duration = calibration_duration
        self.energy_ratio = energy_ratio
        self.min_speech_duration = min_speech_duration
        self.pause_threshold = pause_threshold
        self.max_phrase_duration = max_phrase_duration
        self.language = language
        self.on_text = on_text
        self.on_phrase = on_phrase
        self.session_start_time = session_start_time

        # Queue for transcribed strings or Phrase tuples
        self.text_queue: queue.Queue[Phrase | str] = queue.Queue()
        # Queue specifically for (timestamp_offset, text) phrase tuples
        self.phrase_queue: queue.Queue[Phrase] = queue.Queue()

        # Speech recognition engine
        self.recognizer = sr.Recognizer()

        # Noise calibration state
        self.ambient_energy: float = 150.0
        self.speech_threshold: float = 300.0
        self.is_calibrated: bool = False
        self._damping: float = 0.96  # Exponential smoothing for dynamic ambient tracking

        # Ring buffer for pre-speech frames (prevents cutting off initial phonemes)
        pre_roll_chunks = max(1, int(0.3 / self.chunk_duration))
        self._pre_roll = deque(maxlen=pre_roll_chunks)

        # Internal queues and threading: items are (phrase_start_time, audio_bytes)
        self._phrase_queue: queue.Queue[tuple[float, bytes]] = queue.Queue()
        self._stop_event = threading.Event()
        self._process_thread: Optional[threading.Thread] = None
        self._transcribe_thread: Optional[threading.Thread] = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    def calibrate(self, duration: Optional[float] = None) -> float:
        """Calibrates baseline ambient noise by sampling chunks from the queue."""
        dur = duration if duration is not None else self.calibration_duration
        required_chunks = max(1, int(dur / self.chunk_duration))
        collected_energies = []

        logger.info("Calibrating for ambient lecture hall noise (HVAC/room floor)...")
        while len(collected_energies) < required_chunks and not self._stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.2)
                energy = compute_rms(chunk)
                collected_energies.append(energy)
            except queue.Empty:
                continue

        if collected_energies:
            # Use median/mean ambient level to establish baseline
            self.ambient_energy = float(np.median(collected_energies))
        else:
            self.ambient_energy = 150.0

        # Set threshold with energy ratio to ignore background hum and quiet room chatter
        min_threshold = 120.0
        self.speech_threshold = max(min_threshold, self.ambient_energy * self.energy_ratio)
        self.is_calibrated = True
        logger.info(
            f"Noise calibration complete: ambient baseline RMS={self.ambient_energy:.1f}, "
            f"speech trigger threshold={self.speech_threshold:.1f}"
        )
        return self.ambient_energy

    def start(self, session_start_time: Optional[float] = None) -> None:
        """Starts the audio processor and transcriber background threads."""
        if self._is_running:
            return

        if session_start_time is not None:
            self.session_start_time = session_start_time
        elif self.session_start_time is None:
            self.session_start_time = time.time()

        self._stop_event.clear()
        self._is_running = True

        # Launch audio segmentation thread
        self._process_thread = threading.Thread(
            target=self._audio_processing_loop,
            name="Transcriber-AudioProcessor",
            daemon=True,
        )
        self._process_thread.start()

        # Launch transcription worker thread
        self._transcribe_thread = threading.Thread(
            target=self._transcription_worker,
            name="Transcriber-Worker",
            daemon=True,
        )
        self._transcribe_thread.start()

    def _audio_processing_loop(self) -> None:
        """Reads audio chunks, performs noise tracking, transient rejection,

        and segments audio into speech phrases.
        """
        # Ensure initial calibration
        if not self.is_calibrated:
            self.calibrate()

        in_speech = False
        phrase_start_time = time.time()
        speech_chunks: list[bytes] = []
        silence_chunks_count = 0
        voiced_chunks_count = 0

        silence_chunk_limit = max(1, int(self.pause_threshold / self.chunk_duration))
        min_speech_chunk_limit = max(1, int(self.min_speech_duration / self.chunk_duration))
        max_phrase_chunk_limit = max(1, int(self.max_phrase_duration / self.chunk_duration))

        while not self._stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            energy = compute_rms(chunk)

            if not in_speech:
                # Keep audio in pre-speech rolling window
                self._pre_roll.append(chunk)

                if energy >= self.speech_threshold:
                    # Potential speech onset
                    in_speech = True
                    phrase_start_time = time.time()
                    voiced_chunks_count = 1
                    silence_chunks_count = 0
                    speech_chunks = list(self._pre_roll)
                else:
                    # In silence / ambient: dynamically adapt noise floor to HVAC shifts
                    # Slowly track background noise changes using exponential moving average
                    self.ambient_energy = (
                        self.ambient_energy * self._damping + energy * (1.0 - self._damping)
                    )
                    self.speech_threshold = max(
                        120.0, self.ambient_energy * self.energy_ratio
                    )
            else:
                speech_chunks.append(chunk)

                if energy >= self.speech_threshold:
                    voiced_chunks_count += 1
                    silence_chunks_count = 0
                else:
                    silence_chunks_count += 1

                # Check if utterance finished or reached max duration
                is_silence_complete = silence_chunks_count >= silence_chunk_limit
                is_max_length_reached = len(speech_chunks) >= max_phrase_chunk_limit

                if is_silence_complete or is_max_length_reached:
                    # Filter out short isolated transients (e.g. keyboard typing clicks)
                    if voiced_chunks_count >= min_speech_chunk_limit:
                        full_audio_bytes = b"".join(speech_chunks)
                        self._phrase_queue.put((phrase_start_time, full_audio_bytes))

                    # Reset state for next utterance
                    in_speech = False
                    speech_chunks = []
                    voiced_chunks_count = 0
                    silence_chunks_count = 0
                    self._pre_roll.clear()

        # Flush any in-progress speech when shutting down
        if in_speech and voiced_chunks_count >= min_speech_chunk_limit:
            self._phrase_queue.put((phrase_start_time, b"".join(speech_chunks)))

    def _transcription_worker(self) -> None:
        """Consumes buffered speech utterances and converts them to text."""
        while not self._stop_event.is_set() or not self._phrase_queue.empty():
            try:
                item = self._phrase_queue.get(timeout=0.3)
            except queue.Empty:
                continue

            if isinstance(item, tuple):
                phrase_start_time, audio_bytes = item
            else:
                phrase_start_time, audio_bytes = time.time(), item

            try:
                audio_data = sr.AudioData(
                    audio_bytes,
                    sample_rate=self.sample_rate,
                    sample_width=self.sample_width,
                )
                text = self.recognizer.recognize_google(
                    audio_data, language=self.language
                )
                text = text.strip()
                if text:
                    # Calculate session timestamp offset
                    base_time = (
                        self.session_start_time
                        if self.session_start_time is not None
                        else phrase_start_time
                    )
                    offset_sec = max(0.0, phrase_start_time - base_time)
                    hours = int(offset_sec // 3600)
                    minutes = int((offset_sec % 3600) // 60)
                    seconds = int(offset_sec % 60)
                    timestamp_offset = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                    phrase = Phrase(timestamp_offset, text)
                    self.phrase_queue.put(phrase)
                    self.text_queue.put(phrase)

                    if self.on_phrase:
                        try:
                            self.on_phrase(timestamp_offset, text)
                        except TypeError:
                            self.on_phrase(phrase)
                        except Exception as e:
                            logger.error(f"Error in on_phrase callback: {e}")

                    if self.on_text:
                        try:
                            sig = inspect.signature(self.on_text)
                            params = [
                                p
                                for p in sig.parameters.values()
                                if p.default == p.empty
                                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                            ]
                            if len(params) == 2:
                                self.on_text(timestamp_offset, text)
                            else:
                                self.on_text(phrase)
                        except TypeError:
                            try:
                                self.on_text(text)
                            except Exception as e:
                                logger.error(f"Error in on_text callback: {e}")
                        except Exception as e:
                            logger.error(f"Error in on_text callback: {e}")
            except sr.UnknownValueError:
                # Audio had speech-like energy but no recognizable words (e.g. cough, chair slide)
                pass
            except sr.RequestError as e:
                logger.warning(f"Speech recognition service request error: {e}")
            except Exception as e:
                logger.warning(f"Transcription error: {e}")

    def get_phrase(
        self, block: bool = True, timeout: Optional[float] = None
    ) -> Optional[tuple[str, str]]:
        """Retrieves the next recognized (timestamp_offset, text) phrase tuple from the queue."""
        try:
            return self.phrase_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def get_text(self, block: bool = True, timeout: Optional[float] = None) -> Optional[str]:
        """Retrieves the next transcribed text string from the queue."""
        try:
            item = self.text_queue.get(block=block, timeout=timeout)
            if isinstance(item, tuple):
                return item[1]
            return item
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Stops transcriber threads gracefully."""
        if not self._is_running:
            return

        self._stop_event.set()
        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(timeout=2.0)
        if self._transcribe_thread and self._transcribe_thread.is_alive():
            self._transcribe_thread.join(timeout=2.0)

        self._is_running = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def main():
    """CLI runner demonstrating continuous live transcription paired with AudioRecorder."""
    if AudioRecorder is None:
        print("Error: recorder.py not found in the same directory.")
        return

    print("=" * 60)
    print("  Live Lecture Transcriber with Ambient Noise Calibration  ")
    print("=" * 60)

    # Shared audio queue
    audio_queue = queue.Queue()

    recorder = AudioRecorder(audio_queue=audio_queue)

    def print_transcript(timestamp_offset: str, text: str) -> None:
        print(f"[{timestamp_offset}] {text}")

    transcriber = LiveTranscriber(
        audio_queue=audio_queue,
        on_phrase=print_transcript,
    )

    print("\n1. Starting microphone recording...")
    wav_path = recorder.start()
    print(f"   Audio recording to: {wav_path}")

    print("\n2. Calibrating for lecture hall background noise...")
    print("   (Please keep quiet for 1.5 seconds for HVAC/room floor baseline)")
    transcriber.calibrate(duration=1.5)
    print(f"   -> Ambient Noise RMS: {transcriber.ambient_energy:.1f}")
    print(f"   -> Speech Threshold:  {transcriber.speech_threshold:.1f}")

    print("\n3. Listening and transcribing live...")
    print("   HVAC, distant chatter, and keyboard clatter are filtered out.")
    print("   Press Ctrl+C to stop.\n" + "-" * 60)

    transcriber.start()

    try:
        while transcriber.is_running and recorder.is_recording:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nStopping live transcriber and audio recorder...")
    finally:
        transcriber.stop()
        saved_wav = recorder.stop()
        print(f"Recording saved cleanly to: {saved_wav}")
        print("Live transcriber stopped.")


if __name__ == "__main__":
    main()

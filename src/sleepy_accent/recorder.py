""" Audio Capture (recorder.py)

Opens laptop microphone, records lecture audio directly
to lectures/lecture_<timestamp>.wav, while streaming audio segments to an
in-memory queue.
"""

from datetime import datetime
from pathlib import Path
import queue
import threading
import time
from typing import Optional
import wave
import pyaudio


class CallableBool(int):
    """Boolean that can also be called as a zero-argument function."""

    def __new__(cls, val):
        return super().__new__(cls, 1 if val else 0)

    def __call__(self) -> bool:
        return bool(self)

    def __repr__(self) -> str:
        return "True" if self else "False"

    def __str__(self) -> str:
        return "True" if self else "False"


def ensure_microphone_unmuted() -> bool:
    """Verifies that the Windows master microphone is not muted, automatically unmuting if needed."""
    try:
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        import comtypes
        import ctypes
        dev = AudioUtilities.GetMicrophone()
        if dev:
            interface = dev.Activate(IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None)
            volume = ctypes.cast(interface, ctypes.POINTER(IAudioEndpointVolume))
            if volume.GetMute():
                volume.SetMute(0, None)
                print("\n[AudioRecorder] Notice: Microphone was muted in Windows settings. Unmuted automatically.")
            return True
    except Exception:
        pass
    return False


class AudioRecorder:
    """Manages audio recording from the default microphone, saving uninterrupted audio

    to a WAV file (lectures/audio/lecture_<timestamp>.wav) and concurrently streaming
    audio segments into a Queue.
    """

    def __init__(
        self,
        output_dir: str | Path = "lectures/audio",
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        audio_queue: Optional[queue.Queue] = None,
        input_device_index: Optional[int] = None,
    ):
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.audio_format = pyaudio.paInt16
        self.input_device_index = input_device_index

        # In-memory queue for streaming segments to downstream consumers
        self.audio_queue = audio_queue if audio_queue is not None else queue.Queue()

        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._wav_file: Optional[wave.Wave_write] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_recording = False
        self.current_wav_path: Optional[Path] = None
        self.selected_device_name: Optional[str] = None

    @property
    def is_recording(self) -> CallableBool:
        """Returns recording status, usable both as property or method: is_recording or is_recording()."""
        return CallableBool(self._is_recording)

    def start(self, filename_prefix: Optional[str] = None) -> Path:
        """Starts recording audio in a background thread."""
        if self._is_recording:
            raise RuntimeError("AudioRecorder is already recording.")

        # Ensure Windows microphone endpoint is unmuted
        ensure_microphone_unmuted()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filename_prefix:
            import re
            safe_prefix = re.sub(r'[^\w\-_]+', '_', filename_prefix.strip()).strip('_')
            self.current_wav_path = self.output_dir / f"{safe_prefix}_{timestamp}.wav"
        else:
            self.current_wav_path = self.output_dir / f"lecture_{timestamp}.wav"

        # Initialize PyAudio
        self._pyaudio = pyaudio.PyAudio()

        stream_kwargs = {
            "format": self.audio_format,
            "channels": self.channels,
            "rate": self.sample_rate,
            "input": True,
            "frames_per_buffer": self.chunk_size,
        }
        if self.input_device_index is not None:
            stream_kwargs["input_device_index"] = self.input_device_index
            try:
                self.selected_device_name = self._pyaudio.get_device_info_by_index(self.input_device_index).get("name", "")
            except Exception:
                self.selected_device_name = f"Device #{self.input_device_index}"
        else:
            try:
                default_dev = self._pyaudio.get_default_input_device_info()
                self.selected_device_name = default_dev.get("name", "Default Microphone")
            except Exception:
                self.selected_device_name = "Default Microphone"

        # Open microphone input stream
        self._stream = self._pyaudio.open(**stream_kwargs)

        # Open output WAV file
        self._wav_file = wave.open(str(self.current_wav_path), "wb")
        self._wav_file.setnchannels(self.channels)
        self._wav_file.setsampwidth(self._pyaudio.get_sample_size(self.audio_format))
        self._wav_file.setframerate(self.sample_rate)

        self._stop_event.clear()
        self._is_recording = True

        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        return self.current_wav_path

    def _record_loop(self) -> None:
        """Continuously reads frames from microphone, writes to file, and pushes to queue."""
        try:
            while not self._stop_event.is_set():
                if self._stream is None:
                    break
                data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                if data:
                    if self._wav_file:
                        self._wav_file.writeframes(data)
                    self.audio_queue.put(data)
        finally:
            self._cleanup()

    def stop(self) -> Optional[Path]:
        """Stops recording and cleanly closes audio stream and file."""
        if not self._is_recording:
            return self.current_wav_path

        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join()

        self._is_recording = False
        return self.current_wav_path

    def _cleanup(self) -> None:
        """Closes stream, wave file, and terminates PyAudio cleanly."""
        if self._wav_file:
            try:
                self._wav_file.close()
            except Exception:
                pass
            self._wav_file = None

        if self._stream:
            try:
                if self._stream.is_active():
                    self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._pyaudio:
            try:
                self._pyaudio.terminate()
            except Exception:
                pass
            self._pyaudio = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def main():
    """CLI runner to record audio until interrupted with Ctrl+C."""
    recorder = AudioRecorder()
    print("Initializing microphone...")
    wav_path = recorder.start()
    print(f"Recording started.")
    print(f"Direct WAV destination: {wav_path}")
    print("Streaming chunks to in-memory queue.")
    print("Press Ctrl+C to stop recording...\n")

    try:
        segments_received = 0
        while recorder.is_recording:
            try:
                # Poll queue to display active streaming
                _ = recorder.audio_queue.get(timeout=0.5)
                segments_received += 1
                if segments_received % 16 == 0:  # ~ 1 second of chunks at 16000Hz/1024
                    print(f"\rCaptured and queued {segments_received} segments...", end="", flush=True)
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        print("\nStopping recording gracefully...")
    finally:
        saved_file = recorder.stop()
        print(f"\nRecording stopped. Saved to: {saved_file}")


if __name__ == "__main__":
    main()

"""Tests for main.py Single CLI Entrypoint and integration with session & summarizer."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from main import prompt_course_name, wait_for_start, display_recap_summary
from recorder import AudioRecorder
from session import LiveSession
from summarizer import StudyNoteGenerator


def test_prompt_course_name_custom(monkeypatch):
    """Verifies that entering a course title returns the trimmed string."""
    monkeypatch.setattr("builtins.input", lambda _: "CIS2520 - Data Structures")
    result = prompt_course_name()
    assert result == "CIS2520 - Data Structures"


def test_prompt_course_name_default(monkeypatch):
    """Verifies that hitting Enter returns the default title."""
    monkeypatch.setattr("builtins.input", lambda _: "")
    result = prompt_course_name(default_title="Default Lecture")
    assert result == "Default Lecture"


def test_wait_for_start(monkeypatch):
    """Verifies Enter confirmation is prompted unless auto_start=True."""
    called = False

    def mock_input(_):
        nonlocal called
        called = True
        return ""

    monkeypatch.setattr("builtins.input", mock_input)
    wait_for_start(auto_start=False)
    assert called is True

    called = False
    wait_for_start(auto_start=True)
    assert called is False


def test_recorder_filename_prefix(tmp_path):
    """Verifies AudioRecorder applies custom filename prefixes correctly."""
    recorder = AudioRecorder(output_dir=tmp_path)
    with patch.object(recorder, "_pyaudio"), \
         patch.object(recorder, "_stream"), \
         patch("wave.open") as mock_wave:
        mock_wave.return_value = MagicMock()
        path = recorder.start(filename_prefix="lecture_CIS2520 - Data Structures")
        assert "CIS2520" in path.name
        assert path.name.endswith(".wav")
        recorder.stop()


def test_session_with_title_and_recap_generation(tmp_path):
    """Verifies LiveSession creates titled markdown notes and triggers study recap."""
    session = LiveSession(
        output_dir=tmp_path,
        title="CIS2520 - Data Structures",
        speaker="Dr. Smith",
        auto_summarize=True,
    )

    with patch.object(session.recorder, "start") as mock_rec_start, \
         patch.object(session.recorder, "stop") as mock_rec_stop, \
         patch.object(session.transcriber, "calibrate"), \
         patch.object(session.transcriber, "start"), \
         patch.object(session.transcriber, "stop"):

        wav_fake = tmp_path / "audio" / "lecture_CIS2520_Data_Structures_20260920_120000.wav"
        mock_rec_start.return_value = wav_fake
        mock_rec_stop.return_value = wav_fake

        wav_path, md_path = session.start()
        assert wav_path == wav_fake
        assert md_path.name == "lecture_CIS2520_Data_Structures_20260920_120000.md"

        # Simulate transcribing a couple of phrases
        session._handle_sentence("00:00:10", "A binary search tree maintains sorted order.")
        session._handle_sentence("00:00:25", "Important for the midterm: the height is oh of log n.")

        final_wav, final_md = session.stop()
        assert final_wav == wav_fake
        assert final_md == md_path

        # Verify raw markdown contents
        raw_text = md_path.read_text(encoding="utf-8")
        assert "CIS2520 - Data Structures" in raw_text
        assert "- **Course / Topic:** CIS2520 - Data Structures" in raw_text
        assert "A binary search tree maintains sorted order." in raw_text

        # Verify study recap file was generated
        assert session.study_notes_path is not None
        assert session.study_notes_path.exists()
        recap_text = session.study_notes_path.read_text(encoding="utf-8")
        assert "Post-Lecture Study Recap: CIS2520 - Data Structures" in recap_text
        assert "- **Course / Topic:** CIS2520 - Data Structures" in recap_text
        # Phonetic correction check: "oh of log n" -> O(log n)
        assert "O(log n)" in recap_text


def test_display_recap_summary(tmp_path, capsys):
    """Verifies display_recap_summary prints all details cleanly."""
    session = LiveSession(output_dir=tmp_path)
    session.sentence_count = 42
    recap_file = tmp_path / "study_recap_test.md"
    recap_file.write_text("sample content", encoding="utf-8")

    display_recap_summary(
        session=session,
        course_title="CIS2520 - Data Structures",
        duration_str="00:15:30",
        audio_path=tmp_path / "audio.wav",
        transcript_path=tmp_path / "notes.md",
        recap_path=recap_file,
    )

    captured = capsys.readouterr().out
    assert "CIS2520 - Data Structures" in captured
    assert "00:15:30" in captured
    assert "42" in captured
    assert "study_recap_test.md" in captured


def test_run_cli_full_cycle(tmp_path, monkeypatch, capsys):
    """Verifies run_cli executes, captures args, runs session, and displays recap."""
    from main import run_cli
    import sys

    test_args = [
        "main.py",
        "--course", "CIS2520 - Data Structures",
        "--output-dir", str(tmp_path),
        "--yes",
        "--provider", "heuristic",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with patch("session.AudioRecorder.start") as mock_start, \
         patch("session.AudioRecorder.stop") as mock_stop, \
         patch("session.LiveTranscriber.calibrate"), \
         patch("session.LiveTranscriber.start"), \
         patch("session.LiveTranscriber.stop"), \
         patch("main.start_stop_listener") as mock_listener:

        fake_wav = tmp_path / "audio" / "lecture_test.wav"
        fake_wav.parent.mkdir(parents=True, exist_ok=True)
        fake_wav.touch()
        mock_start.return_value = fake_wav
        mock_stop.return_value = fake_wav

        # Stop event trigger after listener starts
        def trigger_stop(stop_event):
            stop_event.set()

        mock_listener.side_effect = trigger_stop

        run_cli()

        captured = capsys.readouterr().out
        assert "CIS2520 - Data Structures" in captured
        assert "LIVE SPEECH & SESSION RECAP COMPLETE" in captured


def test_boost_and_normalize_audio():
    """Verifies that quiet speech chunks are amplified up to max_gain while preserving limits."""
    import numpy as np
    from transcriber import boost_and_normalize_audio

    # Create a low-amplitude sine wave simulating a distant professor (peak = 2000)
    t = np.linspace(0, 1, 16000, endpoint=False)
    quiet_samples = (2000.0 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    raw_bytes = quiet_samples.tobytes()

    boosted_bytes = boost_and_normalize_audio(raw_bytes, target_peak=24000.0, max_gain=8.0)
    boosted_samples = np.frombuffer(boosted_bytes, dtype=np.int16)

    # Peak should now be significantly amplified (close to target peak)
    assert np.max(np.abs(boosted_samples)) > 15000
    # No clipping overflow above int16 bounds
    assert np.max(boosted_samples) <= 32767
    assert np.min(boosted_samples) >= -32767


def test_boost_and_normalize_audio_silence():
    """Verifies that pure room silence is not amplified excessively."""
    import numpy as np
    from transcriber import boost_and_normalize_audio

    # Silence noise floor around 10
    silence = np.random.randint(-10, 10, size=1024, dtype=np.int16)
    silence_bytes = silence.tobytes()

    result_bytes = boost_and_normalize_audio(silence_bytes)
    result_samples = np.frombuffer(result_bytes, dtype=np.int16)

    # Should remain near zero
    assert np.max(np.abs(result_samples)) < 100


def test_get_whisper_model_cache():
    """Verifies get_whisper_model caches loaded instances."""
    from transcriber import get_whisper_model, _whisper_model_cache

    with patch("transcriber.WhisperModel") as mock_wm:
        mock_instance = MagicMock()
        mock_wm.return_value = mock_instance
        _whisper_model_cache.clear()

        # First call loads model
        m1 = get_whisper_model("base.en", device="cpu")
        assert m1 is not None

        # Second call returns cached instance
        m2 = get_whisper_model("base.en", device="cpu")
        assert m1 is m2
        assert mock_wm.call_count == 1


def test_transcribe_file(tmp_path):
    """Verifies transcribe_file returns formatted Phrase objects."""
    import wave
    from transcriber import transcribe_file, Phrase

    # Create a small valid WAV file
    fake_wav = tmp_path / "test_lecture.wav"
    with wave.open(str(fake_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000)

    # Mock segment
    mock_seg1 = MagicMock()
    mock_seg1.start = 5.2
    mock_seg1.end = 8.5
    mock_seg1.text = "Hello and welcome to CIS2520."

    mock_seg2 = MagicMock()
    mock_seg2.start = 65.0
    mock_seg2.end = 70.0
    mock_seg2.text = "Today we will cover binary search trees."

    with patch("transcriber.get_whisper_model") as mock_get_model:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_seg1, mock_seg2], MagicMock())
        mock_get_model.return_value = mock_model

        phrases = transcribe_file(fake_wav, model_size="base.en")
        assert len(phrases) == 2
        assert isinstance(phrases[0], Phrase)
        assert phrases[0].timestamp_offset == "00:00:05"
        assert phrases[0].text == "Hello and welcome to CIS2520."
        assert phrases[1].timestamp_offset == "00:01:05"
        assert phrases[1].text == "Today we will cover binary search trees."


def test_session_refine_from_audio(tmp_path):
    """Verifies LiveSession.stop re-transcribes audio when refine_from_audio=True."""
    from session import LiveSession
    from transcriber import Phrase

    session = LiveSession(
        output_dir=tmp_path,
        title="Refine Test",
        refine_from_audio=True,
        auto_summarize=False,
    )

    fake_wav = tmp_path / "audio" / "lecture_refine.wav"
    fake_wav.parent.mkdir(parents=True, exist_ok=True)
    fake_wav.touch()

    with patch.object(session.recorder, "start", return_value=fake_wav), \
         patch.object(session.recorder, "stop", return_value=fake_wav), \
         patch.object(session.transcriber, "calibrate"), \
         patch.object(session.transcriber, "start"), \
         patch.object(session.transcriber, "stop"), \
         patch("transcriber.transcribe_file") as mock_transcribe_file:

        refined = [
            Phrase("00:00:02", "Refined sentence one."),
            Phrase("00:00:15", "Refined sentence two."),
        ]
        mock_transcribe_file.return_value = refined

        session.start()
        session.stop()

        assert session.sentence_count == 2
        content = session.md_path.read_text(encoding="utf-8")
        assert "Refined sentence one." in content
        assert "Refined sentence two." in content
        assert "- **Total Sentences Captured:** 2" in content



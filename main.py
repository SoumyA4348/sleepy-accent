#!/usr/bin/env python3
"""Sleepy Accent: Single CLI Entrypoint (main.py)

Interactive terminal interface for live lecture transcription & post-lecture recap:
  1. Prompts for lecture name / course (e.g., CIS2520 - Data Structures).
  2. Press Enter to start listening and calibrating ambient noise.
  3. Live streaming: prints timestamps and sentences as the professor speaks.
  4. Press [q] or [Ctrl+C] to stop listening.
  5. Automatically generates the structured study recap document with phonetic repair
     and concept extraction.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import threading
import time
from typing import Optional

# Optional terminal coloring with graceful fallback
try:
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
except ImportError:
    class Fore:  # type: ignore
        CYAN = GREEN = YELLOW = RED = MAGENTA = BLUE = WHITE = RESET = ""
    class Style:  # type: ignore
        BRIGHT = DIM = NORMAL = RESET_ALL = ""

from session import LiveSession


def print_banner() -> None:
    """Renders the top application banner."""
    print(Fore.CYAN + Style.BRIGHT + "=" * 68)
    print(Fore.CYAN + Style.BRIGHT + "            SLEEPY ACCENT - Live Lecture Assistant & Recap          ")
    print(Fore.CYAN + Style.BRIGHT + "=" * 68)
    print(Style.DIM + " Real-time accent-resilient transcription, auto-save & study summaries")
    print(Fore.CYAN + "-" * 68)


def prompt_course_name(default_title: Optional[str] = None) -> str:
    """Prompts the user for lecture/course title with an optional default."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    fallback = default_title or f"Lecture - {now_str}"

    print(Fore.YELLOW + Style.BRIGHT + "\n[Step 1/2] Lecture Identification")
    try:
        user_input = input(
            Fore.WHITE + "Enter lecture name/course (e.g., "
            + Fore.CYAN + "CIS2520 - Data Structures"
            + Fore.WHITE + ") [Press Enter for default]:\n> "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        sys.exit(0)

    if not user_input:
        user_input = fallback
        print(Style.DIM + f"Using title: {user_input}")
    else:
        print(Fore.GREEN + f"Recorded course title: {user_input}")

    return user_input


def wait_for_start(auto_start: bool = False) -> None:
    """Prompts the user to press Enter to begin listening."""
    if auto_start:
        return

    print(Fore.YELLOW + Style.BRIGHT + "\n[Step 2/2] Start Session")
    try:
        input(Fore.WHITE + "Press " + Fore.GREEN + Style.BRIGHT + "[Enter]" + Fore.WHITE + " to start listening... ")
    except (EOFError, KeyboardInterrupt):
        print("\nSession cancelled.")
        sys.exit(0)


def start_stop_listener(stop_event: threading.Event) -> None:
    """Spawns background listener for 'q' or 'Q' keypress to terminate session."""
    def _listener_loop():
        # Windows instant non-blocking keypress capture
        if sys.platform == "win32":
            try:
                import msvcrt
                while not stop_event.is_set():
                    if msvcrt.kbhit():
                        ch = msvcrt.getch()
                        if ch in (b'q', b'Q', b'\x03'):  # 'q', 'Q', or Ctrl+C
                            stop_event.set()
                            return
                    time.sleep(0.04)
                return
            except Exception:
                pass

        # Fallback console listener
        try:
            while not stop_event.is_set():
                line = sys.stdin.readline()
                if not line or line.strip().lower() in ("q", "quit", "exit"):
                    stop_event.set()
                    return
        except Exception:
            pass

    listener_thread = threading.Thread(target=_listener_loop, daemon=True)
    listener_thread.start()


def display_recap_summary(
    session: LiveSession,
    course_title: str,
    duration_str: str,
    audio_path: Optional[Path],
    transcript_path: Optional[Path],
    recap_path: Optional[Path],
) -> None:
    """Displays structured summary card upon session completion."""
    print("\n" + Fore.CYAN + Style.BRIGHT + "=" * 68)
    print(Fore.CYAN + Style.BRIGHT + "                    LECTURE STUDY RECAP COMPLETE                    ")
    print(Fore.CYAN + Style.BRIGHT + "=" * 68)
    print(Fore.WHITE + Style.BRIGHT + f"  Course / Topic:       " + Fore.YELLOW + f"{course_title}")
    print(Fore.WHITE + Style.BRIGHT + f"  Session Duration:     " + Fore.WHITE + f"{duration_str}")
    print(Fore.WHITE + Style.BRIGHT + f"  Sentences Captured:   " + Fore.GREEN + f"{session.sentence_count}")

    print("\n" + Fore.CYAN + Style.BRIGHT + "  Generated Output Files:")
    if audio_path:
        print(Fore.WHITE + f"  • Audio Recording:    " + Style.DIM + f"{audio_path}")
    if transcript_path:
        print(Fore.WHITE + f"  • Raw Transcript:     " + Style.DIM + f"{transcript_path}")
    if recap_path and recap_path.exists():
        print(Fore.GREEN + Style.BRIGHT + f"  • Final Study Recap:  " + Fore.GREEN + f"{recap_path}")
        print(Fore.WHITE + f"    (Contains phonetic corrections, definitions, formulas & exam tips)")
    else:
        print(Fore.YELLOW + f"  • Final Study Recap:  [Not generated or disabled]")

    print(Fore.CYAN + Style.BRIGHT + "=" * 68 + "\n")


def run_cli():
    """Main CLI execution flow."""
    parser = argparse.ArgumentParser(
        description="Sleepy Accent CLI: Live lecture listening and automated study recap generation."
    )
    parser.add_argument(
        "--course",
        "-c",
        type=str,
        default=None,
        help="Lecture name or course title (e.g. 'CIS2520 - Data Structures')",
    )
    parser.add_argument(
        "--speaker",
        "-s",
        type=str,
        default="Professor",
        help="Speaker label prefix for transcription (default: Professor)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="lectures",
        help="Root directory for saving audio, notes, and study recaps (default: lectures)",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default="en-US",
        help="Spoken language code for speech recognition (default: en-US)",
    )
    parser.add_argument(
        "--calibrate-sec",
        type=float,
        default=1.5,
        help="Microphone ambient noise calibration duration (default: 1.5s)",
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=["auto", "gemini", "openai", "ollama", "heuristic"],
        default="auto",
        help="LLM provider for recap synthesis (default: auto)",
    )
    parser.add_argument(
        "--no-recap",
        action="store_true",
        help="Disable automatic generation of post-lecture study recap",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip Enter confirmation and start listening immediately",
    )

    args = parser.parse_args()

    # Clear terminal screen cleanly if in interactive tty (optional)
    print_banner()

    # 1. Prompt for lecture name / course if not specified in args
    course_name = args.course if args.course else prompt_course_name()

    # 2. Wait for user to hit Enter before starting audio listening
    wait_for_start(auto_start=args.yes)

    # 3. Initialize LiveSession
    print(Fore.CYAN + "\n[*] Initializing microphone and continuous auto-saver...")
    session = LiveSession(
        output_dir=args.output_dir,
        title=course_name,
        speaker=args.speaker,
        language=args.language,
        calibration_duration=args.calibrate_sec,
        auto_summarize=not args.no_recap,
        summarizer_provider=args.provider,
    )

    # Start audio capture & continuous transcript file
    wav_path, md_path = session.start()

    print(
        Fore.GREEN
        + f"[+] Microphone calibrated (Ambient RMS: {session.transcriber.ambient_energy:.1f}, "
        + f"Trigger RMS: {session.transcriber.speech_threshold:.1f})"
    )
    print(Style.DIM + f"    • Audio file:      {wav_path}")
    print(Style.DIM + f"    • Live auto-save:  {md_path}")

    # 4. Active listening banner
    print("\n" + Fore.GREEN + Style.BRIGHT + "=" * 68)
    print(Fore.GREEN + Style.BRIGHT + "  ● LIVE RECORDING ACTIVE")
    print(Fore.WHITE + "  • Spoken sentences will stream below in real time.")
    print(Fore.YELLOW + "  • Press [q] or [Ctrl+C] at any time to finish and generate recap.")
    print(Fore.GREEN + Style.BRIGHT + "=" * 68 + "\n")

    stop_event = threading.Event()
    start_stop_listener(stop_event)

    start_time = time.time()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()

    # 5. Session shutdown and recap generation
    print("\n" + Fore.YELLOW + Style.BRIGHT + "\n[■] Lecture session stopped. Processing final audio...")
    if not args.no_recap:
        print(Fore.CYAN + "[*] Generating final study recap file (phonetic repair & concept extraction)...")

    saved_wav, saved_md = session.stop()

    duration_sec = int(time.time() - start_time)
    duration_str = time.strftime("%H:%M:%S", time.gmtime(duration_sec))

    # 6. Display complete summary card
    display_recap_summary(
        session=session,
        course_title=course_name,
        duration_str=duration_str,
        audio_path=saved_wav,
        transcript_path=saved_md,
        recap_path=session.study_notes_path,
    )


if __name__ == "__main__":
    run_cli()

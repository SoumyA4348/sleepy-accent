"""Post-Lecture Study Note Generator (summarizer.py)

Takes raw transcribed lecture sessions and produces structured study notes:
  1. Fixes common phonetic misinterpretations caused by fast speaking or thick accents.
  2. Extracts Key Definitions & Terminology.
  3. Identifies Formulas / Algorithms / Rules.
  4. Generates Actionable Exam Takeaways / Homework Mentions.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# 1. Phonetic Correction Dictionary & Engine
# ============================================================================

# Academic, mathematical, and computer science phonetic confusion pairs
# Maps misheard phonetic string (lowercase) -> corrected canonical term
ACADEMIC_PHONETIC_REPLACEMENTS: dict[str, str] = {
    # Complexity & asymptotic notation
    "and login": "n log n",
    "and log in": "n log n",
    "in login": "n log n",
    "n login": "n log n",
    "and log and": "n log n",
    "all of n": "O(n)",
    "all of one": "O(1)",
    "oh of n": "O(n)",
    "oh of one": "O(1)",
    "oh of log n": "O(log n)",
    "big oh": "Big-O",
    "big o": "Big-O",
    "theta of": "Theta(",
    "feta of": "Theta(",
    "omega of": "Omega(",
    "o mega of": "Omega(",
    "poly nominal": "polynomial",
    "poly normal": "polynomial",
    "a symptom tick": "asymptotic",
    "a symptom take": "asymptotic",
    "awesome totally": "asymptotically",
    "asymptotic notation": "asymptotic notation",

    # Recurrence & math
    "t of n": "T(n)",
    "tea of n": "T(n)",
    "t parentheses n": "T(n)",
    "t of n over 2": "T(n/2)",
    "t of n divided by 2": "T(n/2)",
    "master theorem": "Master Theorem",
    "recurrence relation": "recurrence relation",
    "base case": "base case",
    "base keys": "base case",

    # Linear algebra & calculus
    "great in descent": "gradient descent",
    "radiant descent": "gradient descent",
    "icon vector": "eigenvector",
    "i can vector": "eigenvector",
    "eye can vector": "eigenvector",
    "icon value": "eigenvalue",
    "i can value": "eigenvalue",
    "eye can value": "eigenvalue",
    "the terminant": "determinant",
    "tan sir": "tensor",
    "single value decomposition": "singular value decomposition (SVD)",
    "you clitorian": "Euclidean",
    "u clidian": "Euclidean",

    # Machine Learning & Stats
    "new roll net": "neural network",
    "new real net": "neural network",
    "back property": "backpropagation",
    "back propagation": "backpropagation",
    "hyper parameter": "hyperparameter",
    "high per parameter": "hyperparameter",
    "lost function": "loss function",
    "cross on trophy": "cross-entropy",
    "regular eyes asian": "regularization",
    "over fitting": "overfitting",
    "under fitting": "underfitting",
    "super wise learning": "supervised learning",
    "unsuper wise learning": "unsupervised learning",
    "rain forest meant learning": "reinforcement learning",
    "mar cov chain": "Markov chain",
    "mark of chain": "Markov chain",
    "bay's rule": "Bayes' rule",
    "base rule": "Bayes' rule",
    "fryer probability": "prior probability",
    "post terrier probability": "posterior probability",
    "stock stick": "stochastic",
    "slow classic": "stochastic",

    # Algorithms & Data Structures
    "die extra": "Dijkstra",
    "dyke stra": "Dijkstra",
    "dike stra": "Dijkstra",
    "re curtain": "recursion",
    "the cursor": "recursion",
    "breath first": "breadth-first",
    "deaf first": "depth-first",
    "binary surge": "binary search",
    "boo lean": "boolean",
    "bullion": "boolean",
    "cash memory": "cache memory",
    "catch memory": "cache memory",
    "cash table": "hash table",
    "grid the algorithm": "greedy algorithm",
    "die namic programming": "dynamic programming",
}

# Regex patterns for phonetic & syntactic cleanups
PHONETIC_REGEX_PATTERNS: list[tuple[re.Pattern, str]] = [
    # "oh of <variable>" or "all of <variable>" -> O(<variable>)
    (re.compile(r"\b(?:oh|all)\s+of\s+([a-zA-Z0-9_\^]+)\b", re.IGNORECASE), r"O(\1)"),
    # "T of <n>" -> T(<n>)
    (re.compile(r"\b(?:t|tea)\s+of\s+([a-zA-Z0-9_\^/]+)\b", re.IGNORECASE), r"T(\1)"),
    # "equals 2 T of n over 2 plus" -> "= 2T(n/2) +"
    (re.compile(r"\bequals\s+2\s*T\(n/2\)\s*plus\b", re.IGNORECASE), r"= 2T(n/2) +"),
    # "n log n" cleanups
    (re.compile(r"\b(?:and|in|an)\s+log\s+(?:n|in|and)\b", re.IGNORECASE), "n log n"),
    # Big-O notation format O(n log n)
    (re.compile(r"\bO\s*\(\s*and\s+log\s+n\s*\)", re.IGNORECASE), "O(n log n)"),
]


class PhoneticCorrector:
    """Corrects phonetic misinterpretations and speech-to-text slurs in academic transcripts."""

    def __init__(self, custom_replacements: Optional[dict[str, str]] = None):
        self.replacements = dict(ACADEMIC_PHONETIC_REPLACEMENTS)
        if custom_replacements:
            self.replacements.update(custom_replacements)

        # Precompile case-insensitive phrase replacements sorted longest first
        sorted_phrases = sorted(self.replacements.keys(), key=len, reverse=True)
        escaped = [re.escape(p) for p in sorted_phrases]
        self._pattern = re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)

    def correct(self, text: str) -> tuple[str, list[dict[str, str]]]:
        """Fixes common phonetic misinterpretations.

        Returns (corrected_text, list_of_corrections_made).
        """
        corrections_made: list[dict[str, str]] = []

        def _replace_match(match: re.Match) -> str:
            matched_text = match.group(1)
            lower_text = matched_text.lower()
            replacement = self.replacements.get(lower_text, matched_text)
            corrections_made.append({"original": matched_text, "corrected": replacement})
            return replacement

        # 1. Apply dictionary replacements
        corrected = self._pattern.sub(_replace_match, text)

        # 2. Apply regex pattern corrections
        for pattern, repl in PHONETIC_REGEX_PATTERNS:
            subbed = pattern.sub(repl, corrected)
            if subbed != corrected:
                corrected = subbed

        return corrected, corrections_made


# ============================================================================
# 2. Heuristic Pattern Extractor (100% Offline / Zero API Key Fallback)
# ============================================================================

class HeuristicStudyNoteExtractor:
    """Extracts study notes from raw lecture lines using deterministic NLP heuristics."""

    DEFINITION_TRIGGERS = [
        re.compile(r"\b(?:is\s+defined\s+as|is\s+referred\s+to\s+as)\s+(.+)", re.IGNORECASE),
        re.compile(r"\b([A-Za-z\s\-]+)\s+means\s+(.+)", re.IGNORECASE),
        re.compile(r"\b(?:the\s+definition\s+of|by)\s+([A-Za-z\s\-]+)[,\s]+(?:is|we\s+mean)\s+(.+)", re.IGNORECASE),
        re.compile(r"\bwe\s+call\s+this\s+([A-Za-z\s\-]+)\b", re.IGNORECASE),
        re.compile(r"\b([A-Za-z\s\-]+)\s+is\s+(?:a|an|the)\s+(.+)", re.IGNORECASE),
    ]

    FORMULA_TRIGGERS = [
        re.compile(r"\b(T\(n\)\s*=\s*.+)", re.IGNORECASE),
        re.compile(r"\b(O\([^)]+\)|Theta\([^)]+\)|Omega\([^)]+\))", re.IGNORECASE),
        re.compile(r"\b(?:formula|equation|recurrence|algorithm|theorem|rule|invariant|law)\b", re.IGNORECASE),
        re.compile(r"([A-Za-z0-9_]+\s*=\s*[A-Za-z0-9_\+\-\*\/\^\(\)\s]+)"),
    ]

    EXAM_TRIGGERS = [
        re.compile(r"\b(?:exam|midterm|final|quiz|test)\b", re.IGNORECASE),
        re.compile(r"\b(?:will\s+be\s+on\s+the|guaranteed\s+question|remember\s+for\s+the)\b", re.IGNORECASE),
        re.compile(r"\b(?:common\s+mistake|pay\s+attention\s+to\s+this)\b", re.IGNORECASE),
    ]

    HOMEWORK_TRIGGERS = [
        re.compile(r"\b(?:homework|problem\s+set|ps[0-9]+|assignment|project|lab)\b", re.IGNORECASE),
        re.compile(r"\b(?:due|submission|submit|gradescope|canvas|turn\s+in)\b", re.IGNORECASE),
        re.compile(r"\b(?:due\s+on|due\s+by|deadline)\b", re.IGNORECASE),
    ]

    def extract(self, lines: list[tuple[str, str]]) -> dict[str, list[dict[str, str]]]:
        """Extracts definitions, formulas, exam cues, and homework mentions.

        Args:
            lines: List of (timestamp_str, text_content).
        """
        definitions: list[dict[str, str]] = []
        formulas: list[dict[str, str]] = []
        exam_takeaways: list[dict[str, str]] = []
        homework_mentions: list[dict[str, str]] = []

        for ts, text in lines:
            clean_text = text.strip()
            if not clean_text:
                continue

            # Check Exam Cues
            if any(p.search(clean_text) for p in self.EXAM_TRIGGERS):
                exam_takeaways.append({"timestamp": ts, "content": clean_text})

            # Check Homework Cues
            if any(p.search(clean_text) for p in self.HOMEWORK_TRIGGERS):
                homework_mentions.append({"timestamp": ts, "content": clean_text})

            # Check Formulas / Algorithms / Rules
            if any(p.search(clean_text) for p in self.FORMULA_TRIGGERS):
                formulas.append({"timestamp": ts, "content": clean_text})

            # Check Definitions
            for pat in self.DEFINITION_TRIGGERS:
                match = pat.search(clean_text)
                if match:
                    definitions.append({"timestamp": ts, "content": clean_text})
                    break

        return {
            "definitions": definitions,
            "formulas": formulas,
            "exam_takeaways": exam_takeaways,
            "homework_mentions": homework_mentions,
        }


# ============================================================================
# 3. LLM Study Note Extractor (Gemini / OpenAI / Ollama)
# ============================================================================

SYSTEM_PROMPT = """You are an expert academic tutor and post-lecture note synthesizer.
You are given a raw transcribed session of a university lecture. Live speech transcription often
suffers from phonetic misinterpretations due to rapid speech, ambient hall noise, or thick professor accents.

Your task is to analyze the lecture and produce high-yield, structured study notes adhering to these 4 targets:

1. PHONETIC & ACCENT CORRECTION:
   - Identify and repair phonetic transcription misinterpretations (e.g. "and login" -> "n log n", "poly nominal" -> "polynomial", "radiant descent" -> "gradient descent", "icon value" -> "eigenvalue", "die extra" -> "Dijkstra").
   - Maintain the professor's exact conceptual meaning while rectifying slurred academic jargon.

2. KEY DEFINITIONS & TERMINOLOGY:
   - Extract every concept, term, or theorem formally introduced or defined.
   - Present as bold term followed by an accurate, clear explanation and timestamp where discussed.

3. FORMULAS, ALGORITHMS & RULES:
   - Identify and format all mathematical equations, recurrence relations, Big-O complexities (LaTeX $...$), algorithm steps, and governing rules/invariants.

4. ACTIONABLE EXAM TAKEAWAYS & HOMEWORK MENTIONS:
   - Flag explicit cues regarding midterms, finals, quizzes, and grading traps ("This will be on the exam", "Common mistake on tests").
   - Highlight homework, problem sets, project deadlines, and submission requirements mentioned.

Output formatted in clean GitHub Markdown with dedicated headers for each section.
"""


class LLMStudyNoteExtractor:
    """Invokes LLM API (Gemini, OpenAI, or Ollama) to synthesize post-lecture study notes."""

    def __init__(
        self,
        provider: str = "auto",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model_name = model_name

        if self.provider == "auto":
            self.provider = self._detect_provider()

    def _detect_provider(self) -> str:
        """Determines best available provider from environment variables."""
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            return "gemini"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        return "heuristic"

    def synthesize(self, raw_transcript: str) -> Optional[str]:
        """Synthesizes structured study notes via the active LLM provider."""
        if self.provider == "gemini":
            return self._call_gemini(raw_transcript)
        elif self.provider == "openai":
            return self._call_openai(raw_transcript)
        elif self.provider == "ollama":
            return self._call_ollama(raw_transcript)
        return None

    def _call_gemini(self, transcript: str) -> Optional[str]:
        """Generates study notes via Google Generative AI (Gemini)."""
        try:
            import google.generativeai as genai

            key = (
                self.api_key
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
            )
            if not key:
                logger.warning("No Gemini API key available.")
                return None

            genai.configure(api_key=key)
            model_name = self.model_name or "gemini-1.5-flash"
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=SYSTEM_PROMPT,
            )
            prompt = (
                "Please analyze the following raw lecture transcript and generate complete, "
                f"structured study notes following the 4 required sections:\n\n{transcript}"
            )
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini generation failed: {e}")
        return None

    def _call_openai(self, transcript: str) -> Optional[str]:
        """Generates study notes via OpenAI API."""
        try:
            from openai import OpenAI

            key = self.api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                logger.warning("No OpenAI API key available.")
                return None

            client = OpenAI(api_key=key)
            model_name = self.model_name or "gpt-4o-mini"
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Analyze this lecture transcript and generate study notes:\n\n{transcript}",
                    },
                ],
                temperature=0.3,
            )
            content = response.choices[0].message.content
            if content:
                return content.strip()
        except Exception as e:
            logger.warning(f"OpenAI generation failed: {e}")
        return None

    def _call_ollama(self, transcript: str) -> Optional[str]:
        """Generates study notes via local Ollama instance (localhost:11434)."""
        try:
            import urllib.error
            import urllib.request

            model_name = self.model_name or "llama3.2"
            payload = json.dumps(
                {
                    "model": model_name,
                    "system": SYSTEM_PROMPT,
                    "prompt": f"Analyze this lecture transcript:\n\n{transcript}",
                    "stream": False,
                }
            ).encode("utf-8")

            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "").strip()
        except Exception as e:
            logger.warning(f"Ollama local generation failed: {e}")
        return None


# ============================================================================
# 4. Master Study Note Generator
# ============================================================================

class StudyNoteGenerator:
    """Post-lecture processor that coordinates phonetic repair and structured study note generation."""

    def __init__(
        self,
        output_dir: str | Path = "lectures/study_notes",
        provider: str = "auto",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.corrector = PhoneticCorrector()
        self.heuristic_extractor = HeuristicStudyNoteExtractor()
        self.llm_extractor = LLMStudyNoteExtractor(
            provider=provider,
            api_key=api_key,
            model_name=model_name,
        )

    def parse_transcript_file(self, filepath: str | Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
        """Reads a lecture markdown file and extracts header metadata and phrase lines.

        Returns (metadata_dict, list_of_tuples[(timestamp, text)]).
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Transcript file not found: {path}")

        lines = path.read_text(encoding="utf-8").splitlines()
        metadata: dict[str, str] = {"file": path.name}
        transcript_phrases: list[tuple[str, str]] = []

        line_regex = re.compile(
            r"^-\s+\*\*\[([0-9:]+)\]\*\*\s+(?:[A-Za-z0-9_\s]+:\s+)?(.+)$"
        )
        alt_regex = re.compile(
            r"^\[([0-9:]+)\]\s+(?:[A-Za-z0-9_\s]+:\s+)?(.+)$"
        )

        for line in lines:
            line_str = line.strip()
            # Parse header metadata
            if line_str.startswith("- **Audio Recording:**"):
                metadata["audio"] = line_str.split("`")[1] if "`" in line_str else ""
            elif line_str.startswith("- **Date:**"):
                metadata["date"] = line_str.replace("- **Date:**", "").strip()
            elif line_str.startswith("- **Speaker:**"):
                metadata["speaker"] = line_str.replace("- **Speaker:**", "").strip()

            # Parse transcript phrases
            m = line_regex.match(line_str) or alt_regex.match(line_str)
            if m:
                ts, text = m.group(1), m.group(2)
                transcript_phrases.append((ts, text))

        return metadata, transcript_phrases

    def generate_from_file(
        self,
        transcript_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> Path:
        """Parses raw lecture note, applies phonetic repair, extracts study notes,

        and saves to study note markdown file.
        """
        source_path = Path(transcript_path)
        metadata, phrases = self.parse_transcript_file(source_path)

        # 1. Apply Phonetic Corrections to all phrases
        corrected_phrases: list[tuple[str, str]] = []
        all_corrections: list[dict[str, str]] = []

        for ts, raw_text in phrases:
            clean_text, corr = self.corrector.correct(raw_text)
            corrected_phrases.append((ts, clean_text))
            all_corrections.extend(corr)

        # Format reconstructed transcript text
        full_transcript_str = "\n".join(
            f"[{ts}] {text}" for ts, text in corrected_phrases
        )

        # 2. Attempt LLM Synthesis (if API available)
        llm_study_notes = None
        if self.llm_extractor.provider != "heuristic":
            llm_study_notes = self.llm_extractor.synthesize(full_transcript_str)

        # 3. Fallback to Heuristic Engine if LLM not used or unavailable
        if llm_study_notes:
            body_content = llm_study_notes
        else:
            body_content = self._format_heuristic_notes(
                corrected_phrases, all_corrections
            )

        # Construct final Study Note Markdown Document
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lecture_date = metadata.get("date", now_str)
        speaker = metadata.get("speaker", "Professor")

        study_note_doc = (
            f"# Post-Lecture Study Notes: {source_path.stem}\n\n"
            f"- **Source Transcript:** `{source_path.name}`\n"
            f"- **Lecture Date:** {lecture_date}\n"
            f"- **Speaker:** {speaker}\n"
            f"- **Generated:** {now_str}\n\n"
            f"---\n\n"
            f"{body_content}\n"
        )

        # Determine target file destination
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if output_path is not None:
            target_file = Path(output_path)
        else:
            stem = source_path.stem.replace("lecture_", "study_note_")
            target_file = self.output_dir / f"{stem}.md"

        target_file.write_text(study_note_doc, encoding="utf-8")
        return target_file

    def _format_heuristic_notes(
        self,
        phrases: list[tuple[str, str]],
        corrections: list[dict[str, str]],
    ) -> str:
        """Formats extracted notes using heuristic NLP patterns into structured markdown."""
        extracted = self.heuristic_extractor.extract(phrases)

        sections = []

        # Section 1: Phonetic & Accent Misinterpretation Corrections
        sections.append("## 1. Phonetic & Accent Corrections (Speech-to-Text Refinements)\n")
        if corrections:
            sections.append(
                "The following misinterpretations caused by fast speaking or thick accents were identified and repaired:\n"
            )
            # Deduplicate corrections
            seen = set()
            for c in corrections:
                orig, fixed = c["original"], c["corrected"]
                key = (orig.lower(), fixed.lower())
                if key not in seen:
                    seen.add(key)
                    sections.append(f"- **\"{orig}\"** -> **`{fixed}`**")
            sections.append("")
        else:
            sections.append(
                "_No significant phonetic distortion detected; lecture audio was clear._\n"
            )

        # Section 2: Key Definitions & Terminology
        sections.append("## 2. Key Definitions & Terminology\n")
        if extracted["definitions"]:
            for item in extracted["definitions"]:
                sections.append(f"- **[{item['timestamp']}]** {item['content']}")
            sections.append("")
        else:
            sections.append(
                "_No explicit definition patterns identified in this session._\n"
            )

        # Section 3: Formulas, Algorithms & Rules
        sections.append("## 3. Formulas, Algorithms & Governing Rules\n")
        if extracted["formulas"]:
            for item in extracted["formulas"]:
                sections.append(f"- **[{item['timestamp']}]** {item['content']}")
            sections.append("")
        else:
            sections.append(
                "_No explicit mathematical formulas or algorithm steps detected._\n"
            )

        # Section 4: Actionable Exam Takeaways & Homework Mentions
        sections.append("## 4. Actionable Exam Takeaways & Homework Mentions\n")

        # 4a. Exam Takeaways
        sections.append("### Exam Takeaways & High-Yield Alerts\n")
        if extracted["exam_takeaways"]:
            for item in extracted["exam_takeaways"]:
                sections.append(f"- [ ] **[{item['timestamp']}]** {item['content']}")
            sections.append("")
        else:
            sections.append(
                "_No direct exam hints or testing warnings were flagged in this session._\n"
            )

        # 4b. Homework Mentions
        sections.append("### Homework Deadlines & Problem Set Mentions\n")
        if extracted["homework_mentions"]:
            for item in extracted["homework_mentions"]:
                sections.append(f"- [ ] **[{item['timestamp']}]** {item['content']}")
            sections.append("")
        else:
            sections.append(
                "_No specific homework due dates or assignment submissions mentioned._\n"
            )

        return "\n".join(sections)


# ============================================================================
# 5. CLI Runner
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Post-Lecture Study Note Generator: Repair phonetic misinterpretations and extract study notes."
    )
    parser.add_argument(
        "transcript",
        nargs="?",
        default=None,
        help="Path to raw lecture markdown transcript (default: latest file in lectures/notes/)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="lectures/study_notes",
        help="Directory to save generated study notes (default: lectures/study_notes)",
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=["auto", "gemini", "openai", "ollama", "heuristic"],
        default="auto",
        help="LLM provider for synthesis (default: auto - uses API key if present, otherwise heuristic)",
    )
    parser.add_argument(
        "--api-key",
        "-k",
        type=str,
        default=None,
        help="Optional API key for Gemini or OpenAI",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Model name to use (e.g. gemini-1.5-flash or gpt-4o-mini)",
    )

    args = parser.parse_args()

    # Find transcript file
    target_path = None
    if args.transcript:
        target_path = Path(args.transcript)
    else:
        # Search for most recent lecture markdown in lectures/notes/ or lectures/
        candidates = list(Path("lectures/notes").glob("*.md"))
        if not candidates:
            candidates = list(Path("lectures").glob("*.md"))

        if candidates:
            target_path = max(candidates, key=os.path.getmtime)
            print(f"Discovered latest lecture transcript: {target_path}")
        else:
            print("Error: No transcript file found. Please specify path to lecture notes markdown.")
            sys.exit(1)

    if not target_path.exists():
        print(f"Error: File does not exist: {target_path}")
        sys.exit(1)

    print("=" * 64)
    print("      Post-Lecture Study Note Generator (summarizer.py)         ")
    print("=" * 64)
    print(f"• Input Transcript:  {target_path}")
    print(f"• Output Directory:  {args.output_dir}")
    print(f"• Provider:          {args.provider}")
    print("-" * 64)
    print("Analyzing lecture transcript...")

    generator = StudyNoteGenerator(
        output_dir=args.output_dir,
        provider=args.provider,
        api_key=args.api_key,
        model_name=args.model,
    )

    study_note_file = generator.generate_from_file(target_path)
    print("\n" + "=" * 64)
    print("                Study Notes Generated Successfully               ")
    print("=" * 64)
    print(f"• Study Notes Saved: {study_note_file}")
    print("=" * 64)


if __name__ == "__main__":
    main()

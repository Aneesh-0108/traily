"""Tests for STT handling with mocked subprocess (no models, no mic)."""

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.stt import transcribe as stt  # noqa: E402


class TestSTT(unittest.TestCase):
    def test_empty_transcription_detection(self):
        self.assertTrue(stt.is_empty_transcription(""))
        self.assertTrue(stt.is_empty_transcription("   \n "))
        self.assertFalse(stt.is_empty_transcription("What's nearby?"))

    def test_transcribe_parses_stdout(self):
        with mock.patch("src.stt.transcribe.subprocess.run") as run, mock.patch(
            "src.stt.transcribe.shutil.which", return_value="/usr/bin/whisper-cli"
        ), mock.patch("src.stt.transcribe.Path.is_file", return_value=True):
            run.return_value = mock.Mock(returncode=0, stdout=" What's nearby?\n")
            text, _elapsed = stt.transcribe(Path("/tmp/q.wav"))
        self.assertEqual(text, "What's nearby?")
        sent = run.call_args[0][0]
        self.assertIn("--no-timestamps", sent)
        self.assertIn("--no-prints", sent)

    def test_transcribe_empty_audio_returns_empty(self):
        with mock.patch("src.stt.transcribe.subprocess.run") as run, mock.patch(
            "src.stt.transcribe.shutil.which", return_value="/usr/bin/whisper-cli"
        ), mock.patch("src.stt.transcribe.Path.is_file", return_value=True):
            run.return_value = mock.Mock(returncode=0, stdout="   \n")
            text, _elapsed = stt.transcribe(Path("/tmp/q.wav"))
        self.assertTrue(stt.is_empty_transcription(text))

    def test_missing_binary(self):
        with mock.patch("src.stt.transcribe.shutil.which", return_value=None):
            with self.assertRaises(stt.STTError):
                stt.transcribe(Path("/tmp/q.wav"))

    def test_missing_model(self):
        with mock.patch("src.stt.transcribe.shutil.which", return_value="/usr/bin/whisper-cli"), mock.patch(
            "src.stt.transcribe.Path.is_file",
            side_effect=lambda: False,
        ):
            with self.assertRaises(stt.STTError):
                stt.transcribe(Path("/tmp/q.wav"), model_path="/nope.bin")

    def test_nonzero_exit(self):
        with mock.patch("src.stt.transcribe.subprocess.run") as run, mock.patch(
            "src.stt.transcribe.shutil.which", return_value="/usr/bin/whisper-cli"
        ), mock.patch("src.stt.transcribe.Path.is_file", return_value=True):
            run.return_value = mock.Mock(returncode=1, stdout="")
            with self.assertRaises(stt.STTError):
                stt.transcribe(Path("/tmp/q.wav"))


if __name__ == "__main__":
    unittest.main()

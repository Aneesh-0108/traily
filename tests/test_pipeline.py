"""Tests for TTS argument handling and pipeline orchestration.

All external processes are mocked: no Piper, no speaker, no models.
"""

import io
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.tts import speak as tts  # noqa: E402
from src import main  # noqa: E402


class TestTTS(unittest.TestCase):
    def test_synthesize_passes_voice_and_outfile(self):
        with mock.patch("src.tts.speak.subprocess.run") as run, mock.patch(
            "src.tts.speak.shutil.which", return_value="/app/.venv/bin/piper"
        ), mock.patch("src.tts.speak.Path.is_file", return_value=True):
            run.return_value = mock.Mock(returncode=0)
            tts.synthesize("Hello there.", Path("/tmp/reply.wav"), voice_path="/m/voice.onnx")
            sent = run.call_args[0][0]
        self.assertIn("-m", sent)
        self.assertIn("/m/voice.onnx", sent)
        self.assertNotIn("--cuda", sent, "piper must stay CPU-only (no --cuda flag)")
        self.assertEqual(run.call_args[1]["input"], "Hello there.")

    def test_synthesize_rejects_empty_text(self):
        with self.assertRaises(tts.TTSError):
            tts.synthesize("   ", Path("/tmp/reply.wav"))

    def test_missing_binary(self):
        with mock.patch("src.tts.speak.shutil.which", return_value=None):
            with self.assertRaises(tts.TTSError):
                tts.synthesize("Hello.", Path("/tmp/reply.wav"))


class TestPipeline(unittest.TestCase):
    def _run(self, **patches):
        with tempfile.TemporaryDirectory() as tmp:
            targets = {
                "src.main.load_places": mock.Mock(
                    return_value=[{"name": "P", "type": "park", "distance_m": 100}]
                ),
                "src.main.build_prompt": mock.Mock(return_value="prompt"),
                "src.main.run_llm": mock.Mock(return_value=("The park is near.", 1.0)),
                "src.main.synthesize": mock.Mock(return_value=0.5),
                "src.main.play_wav": mock.Mock(return_value=None),
            }
            targets.update(patches)
            with ExitStack() as stack:
                for target, new in targets.items():
                    stack.enter_context(mock.patch(target, new))
                return main.run_pipeline("What's nearby?", 8.0, True, Path(tmp))

    def test_text_mode_happy_path(self):
        self.assertEqual(self._run(), 0)

    def test_timing_report_labels_post_recording(self):
        # In text mode the recording window is 0, so post-recording
        # latency must equal STT + LLM + TTS, and the report must label
        # it as record-finish -> playback-start (not end-of-speech).
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(self._run(), 0)
        out = buf.getvalue()
        self.assertIn("post-recording latency (record-finish -> playback-start)", out)
        self.assertIn("total interaction (pipeline-start -> playback-start)", out)
        self.assertIn("NOT true end-of-speech", out)
        self.assertIn("LLM elapsed, incl. CLI startup", out)

    def test_empty_transcription_exits_gracefully(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("src.main.transcribe", return_value=("   ", 0.5)), mock.patch(
                "src.main.record_question", return_value=8.0
            ), mock.patch("src.main.input", return_value=""):
                rc = main.run_pipeline(None, 8.0, True, Path(tmp))
        self.assertEqual(rc, 3)

    def test_llm_failure_exits_nonzero(self):
        from src.llm.answer import LLMError

        rc = self._run(**{"src.main.run_llm": mock.Mock(side_effect=LLMError("boom"))})
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()

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
from src.audio.play import PlaybackError, play_wav  # noqa: E402
from src import main  # noqa: E402


class TestPlayback(unittest.TestCase):
    def test_play_wav_reports_failure(self):
        # A cut-off/failed playback must raise, never silently pass: the
        # user would otherwise hear less than the terminal shows.
        with mock.patch("src.audio.play.subprocess.run") as run, mock.patch(
            "src.audio.play.shutil.which", return_value="/usr/bin/paplay"
        ):
            run.return_value = mock.Mock(returncode=0)
            play_wav(Path("/tmp/reply.wav"))  # no raise on success
            run.return_value = mock.Mock(returncode=1)
            with self.assertRaises(PlaybackError):
                play_wav(Path("/tmp/reply.wav"))

    def test_play_wav_missing_binary(self):
        with mock.patch("src.audio.play.shutil.which", return_value=None):
            with self.assertRaises(PlaybackError):
                play_wav(Path("/tmp/reply.wav"))


class TestTTS(unittest.TestCase):
    def test_synthesize_passes_voice_and_outfile(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "src.tts.speak.subprocess.run"
        ) as run, mock.patch(
            "src.tts.speak.shutil.which", return_value="/app/.venv/bin/piper"
        ), mock.patch(
            "src.tts.speak.Path.is_file", return_value=True
        ):
            held: list = []
            self._mock_piper_run(run, held)
            out = Path(tmp) / "reply.wav"
            tts.synthesize("Hello there.", out, voice_path="/m/voice.onnx")
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

    def _mock_piper_run(self, run, out_wav_holder):
        """Mock subprocess.run to write a real WAV like Piper would."""
        import wave as _wave

        def _side_effect(cmd, **kwargs):
            out = Path(cmd[cmd.index("-f") + 1])
            out_wav_holder.append(out)
            with _wave.open(str(out), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(22050)
                wav.writeframes(b"\x01\x02" * 2205)  # 0.1 s of audio
            return mock.Mock(returncode=0)

        run.side_effect = _side_effect

    def test_synthesize_verifies_audio_frames(self):
        import tempfile as _tf

        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "src.tts.speak.subprocess.run"
        ) as run, mock.patch(
            "src.tts.speak.shutil.which", return_value="/app/.venv/bin/piper"
        ), mock.patch(
            "src.tts.speak.Path.is_file", return_value=True
        ):
            held: list = []
            self._mock_piper_run(run, held)
            elapsed = tts.synthesize("Hello there.", Path(tmp) / "reply.wav")
            self.assertGreaterEqual(elapsed, 0.0)
            # Piper received the COMPLETE final answer text on stdin.
            self.assertEqual(run.call_args[1]["input"], "Hello there.")

    def test_synthesize_rejects_empty_wav(self):
        import wave as _wave

        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "src.tts.speak.subprocess.run"
        ) as run, mock.patch(
            "src.tts.speak.shutil.which", return_value="/app/.venv/bin/piper"
        ), mock.patch(
            "src.tts.speak.Path.is_file", return_value=True
        ):
            out = Path(tmp) / "reply.wav"

            def _empty(cmd, **kwargs):
                with _wave.open(str(out), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(22050)
                    wav.writeframes(b"")
                return mock.Mock(returncode=0)

            run.side_effect = _empty
            with self.assertRaises(tts.TTSError):
                tts.synthesize("Hello there.", out)


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

    def _run_pack(self, argv: list[str], **patches):
        args = main.parse_args(argv)
        with tempfile.TemporaryDirectory() as tmp:
            targets = {
                "src.main.load_pack": mock.Mock(
                    return_value={
                        "area_name": "T", "bbox": {},
                        "places": [{"name": "P", "type": "park", "lat": 0.0, "lon": 0.0,
                                    "osm_id": 1, "osm_type": "node"}],
                    }
                ),
                "src.main.nearest_in_pack": mock.Mock(
                    return_value=[{"name": "P", "type": "park", "distance_m": 50}]
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
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = main.run_pipeline("What's nearby?", 0.0, False, Path(tmp), args)
                return rc, buf.getvalue()

    def test_pack_mode_labels_manual_coords(self):
        rc, out = self._run_pack(["--pack", "p.json", "--lat", "12.95", "--lon", "77.55"])
        self.assertEqual(rc, 0)
        self.assertIn("via manual", out)  # manual coords clearly labelled, never live
        self.assertIn("offline pack", out)

    def test_pack_mode_outside_coverage_exits_4(self):
        from src.places.geo import GeoError

        rc, out = self._run_pack(
            ["--pack", "p.json", "--lat", "12.95", "--lon", "77.55"],
            **{"src.main.nearest_in_pack": mock.Mock(side_effect=GeoError("outside"))},
        )
        self.assertEqual(rc, 4)
        self.assertIn("No nearby places", out)

    def test_pack_mode_empty_results_exits_4(self):
        rc, out = self._run_pack(
            ["--pack", "p.json", "--lat", "12.95", "--lon", "77.55"],
            **{"src.main.nearest_in_pack": mock.Mock(return_value=[])},
        )
        self.assertEqual(rc, 4)

    def test_pack_mode_needs_coords(self):
        rc, _ = self._run_pack(["--pack", "p.json"])
        self.assertEqual(rc, 2)

    def test_incomplete_llm_answer_reaches_piper_complete(self):
        # LLM mentions 1 of 2 places -> synthesize must receive the fallback
        # naming every supplied place (Piper never gets a partial answer).
        places = [
            {"name": "Alpha Park", "type": "park", "distance_m": 100},
            {"name": "Beta Cafe", "type": "cafe", "distance_m": 200},
        ]
        got = {}
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("src.main.load_places", return_value=places), mock.patch(
                "src.main.build_prompt", return_value="prompt"
            ), mock.patch(
                "src.main.run_llm", return_value=("Alpha Park is 100 meters away.", 1.0)
            ), mock.patch(
                "src.main.synthesize", side_effect=lambda t, p: got.setdefault("text", t) or 0.5
            ), mock.patch(
                "src.main.play_wav", return_value=None
            ):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = main.run_pipeline("What's nearby?", 0.0, False, Path(tmp))
        self.assertEqual(rc, 0)
        self.assertIn("deterministic complete summary", buf.getvalue())
        self.assertIn("Alpha Park", got["text"])
        self.assertIn("Beta Cafe", got["text"])
        self.assertIn("100", got["text"])
        self.assertIn("200", got["text"])


if __name__ == "__main__":
    unittest.main()

"""Tests for LLM prompt construction and output parsing.

Deterministic: no model execution. The parser fixture mirrors real
`llama-cli -st --simple-io` output captured during verification.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.llm import answer as llm  # noqa: E402

PLACES = [
    {"name": "Example Temple", "type": "place_of_worship", "distance_m": 300},
    {"name": "Green Park", "type": "park", "distance_m": 450},
]

# Captured shape of real llama-cli -st output (banner trimmed for brevity).
CLI_OUTPUT = """\
> Answer only the user's question, using only these nearby places:
- Example Temple (place_of_worship, 300 m)
- Green Park (park, 450 m)
Rules: never invent names, distances, directions, or facts. No navigation directions. Mention EVERY supplied place by name with its distance in meters, each place once. At most two short sentences, natural spoken English, no Markdown, no lists.
User: What's nearby?
You can visit the Example Temple just 300 meters away or head to Green Park in 450 meters.

[ Prompt: 24.3 t/s | Generation: 12.0 t/s ]


Exiting...
"""


class TestPrompt(unittest.TestCase):
    def test_prompt_contains_question_and_places(self):
        prompt = llm.build_prompt("What's nearby?", PLACES)
        self.assertIn("What's nearby?", prompt)
        for place in PLACES:
            self.assertIn(place["name"], prompt)
            self.assertIn(str(place["distance_m"]), prompt)

    def test_prompt_has_grounding_rules(self):
        prompt = llm.build_prompt("What's nearby?", PLACES)
        for keyword in [
            "using only these nearby places",
            "never invent",
            "No navigation directions",
            "EVERY supplied place",
            "each place once",
            "two short sentences",
            "no Markdown",
        ]:
            self.assertIn(keyword, prompt)

    def test_reasoning_off_mandatory_in_argv(self):
        argv = llm.build_llm_argv("/tmp/prompt.txt", "/tmp/model.gguf")
        self.assertIn("--reasoning", argv)
        self.assertEqual(argv[argv.index("--reasoning") + 1], "off")
        # CPU-only and single-turn template flags travel with it.
        for flag in ["-ngl", "0", "--jinja", "-st"]:
            self.assertIn(flag, argv)

    def test_known_good_perf_flags_in_argv(self):
        # Aligned with the standalone laptop measurement (~4.13 s):
        # -c 2048 -b 128 -ub 64 -n 48. All configurable via TRAILY_LLM_*.
        from src import config

        argv = llm.build_llm_argv("/tmp/prompt.txt", "/tmp/model.gguf")
        for flag, value in [
            ("-c", str(config.LLM_CTX_SIZE)),
            ("-b", str(config.LLM_BATCH_SIZE)),
            ("-ub", str(config.LLM_UBATCH_SIZE)),
            ("-n", str(config.LLM_MAX_TOKENS)),
        ]:
            self.assertIn(flag, argv)
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertEqual(config.LLM_CTX_SIZE, 2048)
        self.assertEqual(config.LLM_BATCH_SIZE, 128)
        self.assertEqual(config.LLM_UBATCH_SIZE, 64)
        self.assertEqual(config.LLM_MAX_TOKENS, 96)  # raised: five names+distances need ~60 tokens; 48 cut answers mid-sentence

    def test_run_llm_always_disables_reasoning(self):
        with mock.patch("src.llm.answer.subprocess.run") as run, mock.patch(
            "src.llm.answer.Path.is_file", return_value=True
        ), mock.patch("src.llm.answer.shutil.which", return_value="/usr/bin/llama-cli"):
            run.return_value = mock.Mock(returncode=0, stdout=CLI_OUTPUT)
            llm.run_llm(llm.build_prompt("What's nearby?", PLACES))
            sent_argv = run.call_args[0][0]
        self.assertIn("--reasoning", sent_argv)
        self.assertEqual(sent_argv[sent_argv.index("--reasoning") + 1], "off")

    def test_parse_cli_output(self):
        prompt = llm.build_prompt("What's nearby?", PLACES)
        parsed = llm.parse_llm_output(CLI_OUTPUT, prompt)
        self.assertNotIn("t/s", parsed)
        self.assertNotIn("Exiting", parsed)
        self.assertNotIn("Example Temple (place_of_worship", parsed)  # prompt echo gone
        self.assertIn("Example Temple", parsed)

    def test_parse_strips_think_tags(self):
        prompt = "Rules...\nUser: Q?"
        out = "> Rules...\nUser: Q?\n<think>secret plan</think> The park is near.\n\n[ Prompt: 1 t/s ]\nExiting...\n"
        self.assertEqual(llm.parse_llm_output(out, prompt), "The park is near.")

    def test_parse_empty_output_raises(self):
        with self.assertRaises(llm.LLMError):
            llm.parse_llm_output("\n\nExiting...\n", "User: Q?")

    def test_parse_missing_anchor_raises(self):
        # No prompt echo anywhere: fail closed instead of returning the banner.
        stdout = "Loading model...\nbuild      : x\nhello world\n"
        with self.assertRaises(llm.LLMError):
            llm.parse_llm_output(stdout, "Rules...\nUser: Q?")

    def test_parse_full_contaminated_stdout(self):
        # Real -st layout: banner, logo, build lines, menu, echo, answer,
        # perf line, Exiting. Only the answer may come out.
        prompt = llm.build_prompt("What are the nearby places?", PLACES)
        stdout = (
            "\n\nLoading model...\n\n"
            "▄▄ ▄▄\n██ ██\n\n"
            "build      : b10809-5266f24da7\n"
            "model      : models/llm/Qwen3.5-4B-Q4_K_M.gguf\n"
            "ftype      : Q4_K - Medium\nmodalities : text\n\n"
            "available commands:\n  /exit or Ctrl+C     stop or exit\n\n\n\n"
            + "\n".join(["> " + prompt.splitlines()[0]] + prompt.splitlines()[1:])
            + "\nThe nearest places are Example Temple at 300 meters and Green Park at 450 meters.\n"
            "\n[ Prompt: 38.3 t/s | Generation: 9.4 t/s ]\n\n\nExiting...\n"
        )
        parsed = llm.parse_llm_output(stdout, prompt)
        self.assertEqual(
            parsed,
            "The nearest places are Example Temple at 300 meters and Green Park at 450 meters.",
        )
        for leak in ["Loading model", "available commands", "Exiting", "t/s", "Example Temple (place_of_worship"]:
            self.assertNotIn(leak, parsed)

    def test_parse_phase2_tail_truncated_echo(self):
        # Real Phase 2 regression: 5-place OSM prompt whose echo is cut at
        # "At most t ... (truncated)" — the final `User:` line never appears.
        # The adapter must still return only the assistant answer.
        sjec_places = [
            {"name": "Saint Pio Chapel", "type": "place_of_worship", "distance_m": 64},
            {"name": "The Lassi Corner", "type": "cafe", "distance_m": 120},
            {"name": "Bethany Provincialate, Vamanjoor", "type": "place_of_worship", "distance_m": 184},
            {"name": "Saint Joseph Worker Church", "type": "place_of_worship", "distance_m": 263},
            {"name": "Hayatul Islam Badriya Masjid Kattapuni", "type": "place_of_worship", "distance_m": 298},
        ]
        prompt = llm.build_prompt("What's nearby?", sjec_places)
        echo_lines = prompt.splitlines()
        stdout = (
            "\n\nLoading model...\n\nbuild      : b10809-5266f24da7\n"
            "model      : models/llm/Qwen3.5-4B-Q4_K_M.gguf\nmodalities : text\n\n"
            "available commands:\n  /exit or Ctrl+C     stop or exit\n\n\n\n"
            + "> " + echo_lines[0] + "\n" + "\n".join(echo_lines[1:-2]) + "\n"
            + "Rules: never invent names, distances, directions, or facts. No navigation directions. "
              "Mention EVERY supplied place by name with its distance in meters, each place once. "
              "At most t ... (truncated)\n"
            + "The nearest places nearby are the Saint Pio Chapel at 64 meters "
              "and The Lassi Corner at 120 meters.\n"
            "\n[ Prompt: 35.9 t/s | Generation: 9.5 t/s ]\n\n\nExiting...\n"
        )
        self.assertNotIn("User: What's nearby?", stdout)  # anchor line truly absent
        parsed = llm.parse_llm_output(stdout, prompt)
        self.assertEqual(
            parsed,
            "The nearest places nearby are the Saint Pio Chapel at 64 meters "
            "and The Lassi Corner at 120 meters.",
        )
        for leak in ["Loading model", "available commands", "Exiting", "t/s",
                     "Saint Pio Chapel (place_of_worship", "User:"]:
            self.assertNotIn(leak, parsed)

    def test_parse_truncated_echo(self):
        # simple-io truncates long echoed input lines with "... (truncated)".
        # The anchor must still be found via the surviving line prefix.
        prompt = "Rules...\nUser: What are the nearby places?"
        out = (
            "Loading model...\nbuild      : x\n"
            "> Rules...\nUser: What are the nearby  ... (truncated)\n"
            "The park is 100 meters away.\n\n[ Prompt: 1 t/s ]\nExiting...\n"
        )
        self.assertEqual(llm.parse_llm_output(out, prompt), "The park is 100 meters away.")

    def test_parse_stops_at_runaway_prompts(self):
        prompt = "Rules...\nUser: Q?"
        out = "> Rules...\nUser: Q?\nThe park is near.\n> \n> \n> \nExiting...\n"
        self.assertEqual(llm.parse_llm_output(out, prompt), "The park is near.")

    def test_looks_contaminated(self):
        self.assertTrue(llm.looks_contaminated(""))
        self.assertTrue(llm.looks_contaminated("Loading model... done. The park is near."))
        self.assertTrue(llm.looks_contaminated("See available commands below."))
        self.assertTrue(llm.looks_contaminated("- Green Park (park, 450 m)"))
        self.assertFalse(llm.looks_contaminated("Green Park is 450 meters away."))

    def test_run_llm_rejects_nonzero_exit(self):
        with mock.patch("src.llm.answer.subprocess.run") as run, mock.patch(
            "src.llm.answer.Path.is_file", return_value=True
        ), mock.patch("src.llm.answer.shutil.which", return_value="/usr/bin/llama-cli"):
            import subprocess as sp

            run.return_value = mock.Mock(returncode=2, stdout="", stderr="")
            with self.assertRaises(llm.LLMError):
                llm.run_llm("prompt\nUser: Q?")
            run.return_value = mock.Mock(returncode=0, stdout="garbage without echo", stderr="")
            with self.assertRaises(llm.LLMError):
                llm.run_llm("prompt\nUser: Q?")
            run.side_effect = sp.TimeoutExpired(cmd="llama-cli", timeout=300)
            with self.assertRaises(llm.LLMError):
                llm.run_llm("prompt\nUser: Q?")

    def test_run_llm_captures_streams_separately(self):
        import subprocess as sp

        with mock.patch("src.llm.answer.subprocess.run") as run, mock.patch(
            "src.llm.answer.Path.is_file", return_value=True
        ), mock.patch("src.llm.answer.shutil.which", return_value="/usr/bin/llama-cli"):
            run.return_value = mock.Mock(returncode=0, stdout=CLI_OUTPUT, stderr="warning: x\n")
            answer, _ = llm.run_llm(llm.build_prompt("What's nearby?", PLACES))
            kwargs = run.call_args[1]
        # stdout and stderr are captured to SEPARATE pipes, never concatenated.
        self.assertEqual(kwargs["stdout"], sp.PIPE)
        self.assertEqual(kwargs["stderr"], sp.PIPE)
        self.assertNotIn("warning", answer)


FIVE_PLACES = [
    {"name": "Saint Pio Chapel", "type": "place_of_worship", "distance_m": 130},
    {"name": "Bethany Provincialate", "type": "place_of_worship", "distance_m": 134},
    {"name": "The Lassi Corner", "type": "cafe", "distance_m": 187},
    {"name": "Example Place D", "type": "park", "distance_m": 250},
    {"name": "Example Place E", "type": "library", "distance_m": 310},
]

FULL_ANSWER = (
    "Nearby are Saint Pio Chapel at 130 meters, Bethany Provincialate at 134 meters, "
    "The Lassi Corner at 187 meters, Example Place D at 250 meters, "
    "and Example Place E at 310 meters."
)


class TestCompleteness(unittest.TestCase):
    def test_complete_answer_stands(self):
        self.assertEqual(llm.missing_places(FULL_ANSWER, FIVE_PLACES), [])
        text, fell_back = llm.ensure_complete_answer(FULL_ANSWER, FIVE_PLACES)
        self.assertEqual(text, FULL_ANSWER)
        self.assertFalse(fell_back)

    def test_incomplete_answer_triggers_fallback(self):
        partial = "Saint Pio Chapel is 130 meters away and The Lassi Corner is 187 meters away."
        missing = llm.missing_places(partial, FIVE_PLACES)
        self.assertIn("Bethany Provincialate", missing)
        self.assertIn("Example Place D", missing)
        self.assertIn("Example Place E", missing)
        text, fell_back = llm.ensure_complete_answer(partial, FIVE_PLACES)
        self.assertTrue(fell_back)
        for place in FIVE_PLACES:
            self.assertIn(place["name"], text)
            self.assertIn(str(place["distance_m"]), text)

    def test_wrong_distance_triggers_fallback(self):
        wrong = FULL_ANSWER.replace("130 meters", "999 meters")
        self.assertIn("Saint Pio Chapel", llm.missing_places(wrong, FIVE_PLACES))
        text, fell_back = llm.ensure_complete_answer(wrong, FIVE_PLACES)
        self.assertTrue(fell_back)
        self.assertNotIn("999", text)

    def test_distance_boundary_no_false_match(self):
        # "130" must not match "1300": word-boundary check.
        answer = FULL_ANSWER.replace("130 meters", "1300 meters")
        self.assertIn("Saint Pio Chapel", llm.missing_places(answer, FIVE_PLACES))

    def test_fallback_invents_nothing(self):
        text = llm.fallback_answer(FIVE_PLACES)
        for place in FIVE_PLACES:
            self.assertEqual(text.count(place["name"]), 1)  # each place exactly once
        self.assertNotIn("Example Place F", text)

    def test_fallback_suitable_for_speech(self):
        text = llm.fallback_answer(FIVE_PLACES)
        for ch in "*#{}[]|`_":
            self.assertNotIn(ch, text)
        self.assertNotIn("\n", text)
        self.assertTrue(text.endswith("."))
        self.assertEqual(len(text.split(".")), 2)  # single sentence + trailing empty


if __name__ == "__main__":
    unittest.main()

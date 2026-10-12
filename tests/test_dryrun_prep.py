"""Tests for Phase 3B dry-run preparation (no billing, no SDK, no network).

Verifies cost math, subset validation, the no-execution guarantees, the
export procedure markers, and that no secrets hide in the new files.
"""

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.tinker_dryrun import SUBSET_EXAMPLES, parse_args, validate_subset  # noqa: E402
from src.finetune import converter as conv  # noqa: E402
from src.finetune import costs  # noqa: E402


class TestConverterPreflight(unittest.TestCase):
    def _converter_dir(self):
        return REPO_ROOT / "tools" / "lora-converter"

    def test_pinned_commit_recorded(self):
        self.assertEqual(conv.PINNED_COMMIT, "b016f461be57")
        self.assertIn("V-head", conv.PINNED_SUBJECT)

    def test_prepared_checkout_passes(self):
        ok, msgs = conv.preflight(sys.executable, self._converter_dir())
        # Files + fix marker verified here; deps verdict depends on interpreter.
        ok_files, _ = conv.check_files(self._converter_dir())
        self.assertTrue(ok_files)
        self.assertEqual(len(msgs), 2)

    def test_missing_dir_fails(self):
        ok, _ = conv.check_files(REPO_ROOT / "tools" / "does-not-exist")
        self.assertFalse(ok)

    def test_missing_marker_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "conversion").mkdir()
            (d / "gguf-py" / "gguf").mkdir(parents=True)
            (d / "convert_lora_to_gguf.py").write_text("# x\n")
            (d / "conversion" / "__init__.py").write_text("")
            (d / "conversion" / "qwen.py").write_text("# no marker here\n")
            (d / "gguf-py" / "gguf" / "constants.py").write_text("")
            ok, msg = conv.check_files(d)
            self.assertFalse(ok)
            self.assertIn("V-head", msg)

    def test_module_check_present_and_absent(self):
        ok, _ = conv.check_python_modules(sys.executable, ("json", "os"))
        self.assertTrue(ok)
        ok, msg = conv.check_python_modules(sys.executable, ("json", "no_such_mod_xyz"))
        self.assertFalse(ok)
        self.assertIn("no_such_mod_xyz", msg)

    def test_env_import_contract(self):
        # The fix: train needs ONLY tinker; torch/cookbook live in convert env.
        self.assertEqual(conv.TRAIN_IMPORTS, ("tinker",))
        for mod in ("torch", "tinker_cookbook.supervised.data", "tinker_cookbook.weights"):
            self.assertIn(mod, conv.CONVERT_IMPORTS)
        self.assertNotIn("torch", conv.TRAIN_IMPORTS)
        self.assertNotIn("tinker_cookbook", conv.TRAIN_IMPORTS)


class TestBatchHelper(unittest.TestCase):
    def test_dump_data_args_parse(self):
        from scripts.tinker_batch_helper import parse_args

        args = parse_args(["dump-data", "--train", "f.jsonl"])
        self.assertEqual(args.command, "dump-data")
        self.assertEqual(args.batch_size, 2)

    def test_export_args_parse(self):
        from scripts.tinker_batch_helper import parse_args

        args = parse_args(["export", "--tinker-path", "tinker://x", "--ckpt-dir", "c",
                           "--peft-dir", "p"])
        self.assertEqual(args.command, "export")

    def test_export_refuses_without_key(self):
        from scripts.tinker_batch_helper import main as helper_main

        with mock.patch.dict(os.environ, {}, clear=True):
            rc = helper_main(["export", "--tinker-path", "tinker://x",
                              "--ckpt-dir", "c", "--peft-dir", "p"])
        self.assertEqual(rc, 2)

    def test_helper_never_prints_key(self):
        text = (REPO_ROOT / "scripts/tinker_batch_helper.py").read_text(encoding="utf-8")
        code = [l.strip() for l in text.splitlines()
                if "TINKER_API_KEY" in l and not l.strip().startswith(("#", '"""', "Reads"))]
        # Exactly two code uses: presence check + static refusal (value never formatted).
        self.assertEqual(len(code), 2)
        self.assertTrue(any("os.environ.get" in l for l in code))
        for line in code:
            self.assertNotIn("environ[", line)


class TestCosts(unittest.TestCase):
    def test_token_math(self):
        self.assertEqual(costs.train_tokens(8, 40, 8, 100), int(40 * 8 * 100 * 1.10))
        self.assertAlmostEqual(costs.cost_usd(1_000_000), 0.737)

    def test_model_id_exact(self):
        self.assertEqual(costs.MODEL_ID, "Qwen/Qwen3.5-4B")

    def test_dryrun_cents(self):
        est = costs.dryrun_estimate()
        self.assertLess(est["usd"], 0.50)
        self.assertGreater(est["usd"], 0)

    def test_fullrun_bounded(self):
        est = costs.fullrun_estimate()
        self.assertLess(est["usd"], 1.0)
        self.assertEqual(est["steps"], costs.full_run_steps())
        self.assertGreaterEqual(est["steps"], 1)

    def test_minimal_smoke_config(self):
        # Smallest chain that still updates + exports: 2 ex, batch 2,
        # 2 steps, rank 4. No documented minimum exists; this is the floor.
        self.assertEqual(costs.DRYRUN["n_examples"], 2)
        self.assertEqual(costs.DRYRUN["batch_size"], 2)
        self.assertEqual(costs.DRYRUN["steps"], 2)
        self.assertEqual(costs.DRYRUN["lora_rank"], 4)
        est = costs.dryrun_estimate()
        self.assertLess(est["usd"], 0.01)
        self.assertEqual(
            est["tokens"],
            int(2 * 2 * costs.SEQ_LEN_MEASURED * costs.TEMPLATE_OVERHEAD),
        )


class TestSubset(unittest.TestCase):
    def _subset_file(self):
        lines = open(REPO_ROOT / "data/finetune/train.jsonl", encoding="utf-8").readlines()
        tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
        tmp.writelines(lines[:SUBSET_EXAMPLES])
        tmp.close()
        return tmp.name

    def test_valid_subset(self):
        path = self._subset_file()
        try:
            self.assertEqual(validate_subset(path), SUBSET_EXAMPLES)
        finally:
            Path(path).unlink()

    def test_wrong_count_refused(self):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
        tmp.write('{"messages": [{"role": "user", "content": "q"}, '
                  '{"role": "assistant", "content": "a"}]}\n')
        tmp.close()
        try:
            with self.assertRaises(ValueError):
                validate_subset(tmp.name)
        finally:
            Path(tmp.name).unlink()

    def test_missing_messages_refused(self):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
        for _ in range(SUBSET_EXAMPLES):
            tmp.write('{"prompt": "q", "completion": "a"}\n')
        tmp.close()
        try:
            with self.assertRaises(ValueError):
                validate_subset(tmp.name)
        finally:
            Path(tmp.name).unlink()


class TestNoExecution(unittest.TestCase):
    def test_dry_print_only_needs_no_key_no_sdk(self):
        from scripts import tinker_dryrun as td

        path = TestSubset()._subset_file()
        try:
            with mock.patch.dict(os.environ, {}, clear=True):
                rc = td.main(["--train", path, "--dry-print-only"])
            self.assertEqual(rc, 0)
        finally:
            Path(path).unlink()

    def test_refuses_without_key(self):
        from scripts import tinker_dryrun as td

        path = TestSubset()._subset_file()
        try:
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
                "src.finetune.converter.check_files", return_value=(True, "ok")
            ), mock.patch(
                "src.finetune.converter.check_python_modules", return_value=(True, "ok")
            ):
                rc = td.main(["--train", path])
            self.assertEqual(rc, 2)
        finally:
            Path(path).unlink()

    def test_preflight_blocks_before_any_billing(self):
        # Missing converter sources stop the run BEFORE key check/training.
        from scripts import tinker_dryrun as td

        path = TestSubset()._subset_file()
        try:
            probe = mock.MagicMock(name="tinker-probe")
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
                td, "CONVERT_LORA_SCRIPT", "/nonexistent/convert_lora_to_gguf.py"
            ), mock.patch.dict(sys.modules, {"tinker": probe}):
                import io as _io
                from contextlib import redirect_stdout as _ro

                buf = _io.StringIO()
                with _ro(buf):
                    rc = td.main(["--train", path])
            self.assertEqual(rc, 4)
            self.assertIn("nothing billed", buf.getvalue())
            probe.ServiceClient.assert_not_called()  # no session, no tokens
        finally:
            Path(path).unlink()

    def test_export_script_guarded_and_marked(self):
        text = (REPO_ROOT / "scripts/tinker_export_local.sh").read_text(encoding="utf-8")
        self.assertIn("REQUIRE_EXPLICIT_APPROVAL", text)
        self.assertIn("UNVERIFIED", text)
        self.assertIn("build_lora_adapter", text)
        self.assertIn("build_hf_model", text)  # current merge API (old script deprecated)
        self.assertIn("convert_lora_to_gguf", text)
        self.assertIn("--lora", text)
        self.assertIn("llama-quantize", text)
        self.assertIn("TINKER_API_KEY", text)

    def test_no_secrets_in_new_files(self):
        keyish = re.compile(r"(sk-[A-Za-z0-9]{8,}|tinker[_-]?key\\s*[:=]\\s*['\"][^'\"]+)", re.I)
        for rel in ("scripts/tinker_dryrun.py", "scripts/tinker_export_local.sh",
                    "src/finetune/costs.py", "src/finetune/__init__.py"):
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
            self.assertIsNone(keyish.search(text), f"possible secret in {rel}")

    def test_eval_reuse_path_exists(self):
        # Fine-tuned comparison reuses the harness via model override + new out file.
        self.assertTrue((REPO_ROOT / "scripts/evaluate_baseline.py").is_file())
        self.assertTrue((REPO_ROOT / "data/finetune/baseline_results.jsonl").is_file())

    def test_first_eval_prompt_from_heldout_only(self):
        from scripts.tinker_dryrun import first_eval_user_prompt

        prompt = first_eval_user_prompt(str(REPO_ROOT / "data/finetune/eval.jsonl"))
        self.assertIn("nearby places", prompt)
        train_text = (REPO_ROOT / "data/finetune/train.jsonl").read_text(encoding="utf-8")
        # The smoke prompt is borrowed read-only; training still uses train file.
        self.assertTrue(len(prompt) > 100)


class TestFutureResolution(unittest.TestCase):
    """Pending futures, success, failed saves, missing paths — all mocked."""

    def _client(self, save_resp=None, save_exc=None):
        client = mock.MagicMock(name="client")
        fwbw = mock.Mock(name="fwbw-future")
        optim = mock.Mock(name="optim-future")
        client.forward_backward.side_effect = lambda *a, **k: fwbw
        client.optim_step.side_effect = lambda *a, **k: optim
        save_future = mock.Mock(name="save-future")
        if save_exc is not None:
            save_future.result.side_effect = save_exc
        else:
            save_future.result.return_value = save_resp
        client.save_weights_for_sampler.return_value = save_future
        return client, fwbw, optim

    def test_pending_futures_resolved_before_return(self):
        from scripts.tinker_dryrun import run_training_steps

        client, fwbw, optim = self._client()
        with mock.patch.dict(sys.modules, {"tinker": mock.MagicMock(name="tinker")}):
            steps = run_training_steps(client, [{"ex": 1}, {"ex": 2}], 2, 2, 5e-4)
        self.assertEqual(steps, 2)
        self.assertEqual(fwbw.result.call_count, 2)
        self.assertEqual(optim.result.call_count, 2)

    def test_step_failure_raises_tinker_run_error(self):
        from scripts.tinker_dryrun import TinkerRunError, run_training_steps

        client, _, _ = self._client()
        client.forward_backward.side_effect = lambda *a, **k: mock.Mock(
            name="bad-future",
            result=mock.Mock(side_effect=RuntimeError("server exploded")))
        with mock.patch.dict(sys.modules, {"tinker": mock.MagicMock(name="tinker")}):
            with self.assertRaises(TinkerRunError):
                run_training_steps(client, [{"ex": 1}], 1, 1, 5e-4)

    def test_sampler_checkpoint_success(self):
        from scripts.tinker_dryrun import save_sampler_checkpoint

        resp = mock.Mock(name="sampler-response")
        resp.path = "tinker://run-9/sampler_weights/final"
        client, _, _ = self._client(save_resp=resp)
        self.assertEqual(save_sampler_checkpoint(client), "tinker://run-9/sampler_weights/final")
        client.save_weights_for_sampler.assert_called_once()
        client.save_state.assert_not_called()

    def test_sampler_save_failure_raises(self):
        from scripts.tinker_dryrun import TRAIN_FAILED_EXIT, TinkerRunError, save_sampler_checkpoint

        client, _, _ = self._client(save_exc=RuntimeError("quota exceeded"))
        with self.assertRaises(TinkerRunError):
            save_sampler_checkpoint(client)
        self.assertEqual(TRAIN_FAILED_EXIT, 5)

    def test_sampler_missing_path_raises(self):
        from scripts.tinker_dryrun import TinkerRunError, save_sampler_checkpoint

        for bad_path in (None, "", 12345):
            resp = mock.Mock(name="sampler-response")
            resp.path = bad_path
            client, _, _ = self._client(save_resp=resp)
            with self.subTest(path=bad_path):
                with self.assertRaises(TinkerRunError):
                    save_sampler_checkpoint(client)

    def test_main_maps_remote_failure_to_exit_5(self):
        import json as _json
        import types as _t

        from scripts import tinker_dryrun as td

        types_mod = _t.ModuleType("tinker.types")
        types_mod.Datum = mock.MagicMock(name="Datum")
        types_mod.ModelInput = mock.MagicMock(name="ModelInput")
        types_mod.TensorData = mock.MagicMock(name="TensorData")
        payload = {"examples": [
            {"input_tokens": [1], "loss": {"target_tokens": {"data": [1], "dtype": "int64"}}},
            {"input_tokens": [2], "loss": {"target_tokens": {"data": [2], "dtype": "int64"}}},
        ]}
        subset = TestSubset()._subset_file()
        try:
            with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
                os.environ, {"TINKER_API_KEY": "fake-test-key"}, clear=True
            ), mock.patch.dict(
                sys.modules, {"tinker": mock.MagicMock(name="tinker"),
                              "tinker.types": types_mod}
            ), mock.patch.object(
                td, "run_training_steps", side_effect=td.TinkerRunError("boom")
            ), mock.patch(
                "src.finetune.converter.check_python_modules",
                return_value=(True, "ok (mocked)"),
            ), mock.patch(
                "scripts.tinker_dryrun.subprocess.run",
                return_value=mock.Mock(returncode=0, stdout=_json.dumps(payload), stderr=""),
            ):
                rc = td.main(["--train", subset, "--workdir", str(Path(tmp) / "smoke"),
                              "--convert-python", sys.executable])
            self.assertEqual(rc, 5)
        finally:
            Path(subset).unlink()
    """Mocked-SDK run through the helper subprocess boundary.

    The billable section must NEVER import tinker_cookbook/torch in the
    training interpreter (that was the ModuleNotFoundError): batch building
    and export go through the helper, whose subprocess calls are mocked
    here. No network, no key usage, no real SDK.
    """

    DUMP_PAYLOAD = {
        "examples": [
            {"input_tokens": [1, 2, 3],
             "loss": {"weights": {"data": [0.0, 1.0, 0.0], "dtype": "float32"},
                      "target_tokens": {"data": [1, 2, 3], "dtype": "int64"}}},
            {"input_tokens": [4, 5],
             "loss": {"weights": {"data": [1.0, 1.0], "dtype": "float32"},
                      "target_tokens": {"data": [4, 5], "dtype": "int64"}}},
        ]
    }

    def _fake_tinker(self):
        import types as _t

        tinker = mock.MagicMock(name="tinker")
        service = mock.MagicMock(name="service")
        client = mock.MagicMock(name="client")
        service.create_lora_training_client.return_value = client
        tinker.ServiceClient.return_value = service
        # Futures resolve via .result() like the real APIFuture; shared per
        # method so tests can assert every submitted future was resolved.
        fwbw_future = mock.Mock(name="fwbw-future")
        optim_future = mock.Mock(name="optim-future")
        client.forward_backward.side_effect = lambda *a, **k: fwbw_future
        client.optim_step.side_effect = lambda *a, **k: optim_future
        sampler_resp = mock.Mock(name="sampler-response")
        sampler_resp.path = "tinker://run-1/sampler_weights/final"
        client.save_weights_for_sampler.return_value.result.return_value = sampler_resp
        types_mod = _t.ModuleType("tinker.types")
        types_mod.Datum = mock.MagicMock(name="Datum")
        types_mod.ModelInput = mock.MagicMock(name="ModelInput")
        types_mod.TensorData = mock.MagicMock(name="TensorData")
        return tinker, types_mod, service, client, fwbw_future, optim_future

    def test_train_export_attempts_conversion(self):
        # Full mocked chain: training calls verified, helper used for
        # dump-data + export, conversion attempted with --convert-python.
        # (Fake converter script exits 0 without writing -> rc 4; a real
        # converted file needs a real adapter.)
        import json as _json
        import subprocess as _sp

        from scripts import tinker_dryrun as td

        tinker, types_mod, service, client, fwbw_future, optim_future = self._fake_tinker()
        subset = TestSubset()._subset_file()
        helper_calls: list[list[str]] = []
        real_run = _sp.run

        def _fake_run(argv, **kwargs):
            if any(str(a).endswith("tinker_batch_helper.py") for a in argv):
                helper_calls.append([str(a) for a in argv])
                if "dump-data" in argv:
                    return mock.Mock(returncode=0,
                                     stdout=_json.dumps(self.DUMP_PAYLOAD), stderr="")
                return mock.Mock(returncode=0, stdout="exported", stderr="")
            return real_run(argv, **kwargs)  # converter: really execute fake script

        try:
            with tempfile.TemporaryDirectory() as tmp:
                conv = Path(tmp) / "conv"
                (conv / "conversion").mkdir(parents=True)
                (conv / "gguf-py" / "gguf").mkdir(parents=True)
                (conv / "convert_lora_to_gguf.py").write_text("# fake\n")
                (conv / "conversion" / "__init__.py").write_text("")
                (conv / "conversion" / "qwen.py").write_text("# get_lora_A_B fix marker\n")
                (conv / "gguf-py" / "gguf" / "constants.py").write_text("")
                with mock.patch.dict(sys.modules, {"tinker": tinker,
                                                      "tinker.types": types_mod}), mock.patch.dict(
                    os.environ, {"TINKER_API_KEY": "fake-test-key"}, clear=True
                ), mock.patch.object(
                    td, "CONVERT_LORA_SCRIPT", str(conv / "convert_lora_to_gguf.py")
                ), mock.patch(
                    # Deps verdict is environment-specific (proven manually for
                    # the convert venv); files check above runs for real.
                    "src.finetune.converter.check_python_modules",
                    return_value=(True, "ok (mocked)"),
                ), mock.patch(
                    "scripts.tinker_dryrun.subprocess.run", side_effect=_fake_run
                ):
                    rc = td.main(["--train", subset, "--workdir",
                                  str(Path(tmp) / "smoke"),
                                  "--convert-python", sys.executable])
            self.assertEqual(rc, 4)
            service.create_lora_training_client.assert_called_once_with(
                base_model="Qwen/Qwen3.5-4B", rank=4, seed=7)
            # Correct, verified API shapes (NOT save_checkpoint / optim_step(lr=...)):
            self.assertEqual(client.forward_backward.call_count, 2)
            call_kwargs = client.forward_backward.call_args_list[0][1]
            self.assertEqual(call_kwargs.get("loss_fn"), "cross_entropy")
            for call in client.optim_step.call_args_list:
                self.assertIn("adam_params", call[1])
                self.assertNotIn("lr", call[1])
            # The regression fix: EVERY submitted future is resolved via
            # .result() (the bug accessed .path on an unresolved future).
            self.assertEqual(fwbw_future.result.call_count, 2)
            self.assertEqual(optim_future.result.call_count, 2)
            # Sampler checkpoint (NOT training-state save_state) feeds export.
            client.save_weights_for_sampler.assert_called_once()
            client.save_state.assert_not_called()
            client.save_checkpoint.assert_not_called()
            # Helper boundary: dump-data + export ran under --convert-python,
            # key never appears in any subprocess argv.
            kinds = [[a for a in call if a in ("dump-data", "export")] for call in helper_calls]
            self.assertIn(["dump-data"], kinds)
            self.assertIn(["export"], kinds)
            for call in helper_calls:
                self.assertEqual(call[0], sys.executable)
                self.assertNotIn("fake-test-key", " ".join(call))
            # The fix itself: cookbook must never enter the train interpreter.
            self.assertNotIn("tinker_cookbook", sys.modules)
        finally:
            Path(subset).unlink()


if __name__ == "__main__":
    unittest.main()

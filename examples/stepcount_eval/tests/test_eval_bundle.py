# ruff: noqa: E402 - bundle modules are imported after adding this directory to sys.path.
from __future__ import annotations

import contextlib
import io
import json
import os
import pickle
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


BUNDLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE_DIR))

import analyze_results
import bundle_common
import eval_stepcount
import hf_release
import preflight
import run_v36_eval


class AdaptiveTurnTests(unittest.TestCase):
    def test_gt_plus_three_and_task_caps(self):
        cases = ((2, 13, 5), (10, 13, 13), (11, 53, 14), (50, 53, 53), (51, 54, 54))
        for gt, cap, expected in cases:
            self.assertEqual(eval_stepcount.compute_effective_max_rounds(gt, cap, 3, True), expected)

    def test_invalid_gt_falls_back_to_cap(self):
        self.assertEqual(eval_stepcount.compute_effective_max_rounds("not-int", 13, 3, True), 13)
        self.assertEqual(eval_stepcount.compute_effective_max_rounds(-1, 13, 3, True), 13)

    def test_worker_cost_balancing_is_deterministic(self):
        samples = [{"id": str(index), "answer": answer} for index, answer in enumerate((50, 2, 49, 3, 48, 4))]
        first = eval_stepcount.distribute_samples_by_estimated_turn_cost(samples, 3, 53, 3)
        second = eval_stepcount.distribute_samples_by_estimated_turn_cost(samples, 3, 53, 3)
        self.assertEqual(
            [[row["id"] for row in chunk] for chunk in first[0]], [[row["id"] for row in chunk] for chunk in second[0]]
        )
        self.assertEqual(first[1], second[1])
        self.assertLessEqual(max(first[1]) - min(first[1]), 3)

    def test_spawn_roundtrip_preserves_and_activates_runtime_config(self):
        original = {
            name: getattr(eval_stepcount.Config, name) for name in dir(eval_stepcount.Config) if name.isupper()
        }
        try:
            eval_stepcount.Config.MODEL_PATH = "/models/step77"
            eval_stepcount.Config.OUTPUT_DIR = "/outputs/step77/pixmo-test"
            eval_stepcount.Config.MAX_ROUNDS = 13
            snapshot = eval_stepcount.snapshot_runtime_config()
            restored = pickle.loads(pickle.dumps(snapshot))
            self.assertEqual(restored.MODEL_PATH, "/models/step77")
            self.assertIn("MODEL_PATH", vars(restored))

            eval_stepcount.Config.MODEL_PATH = ""
            eval_stepcount.Config.OUTPUT_DIR = ""
            eval_stepcount.activate_runtime_config(restored)
            self.assertEqual(eval_stepcount.Config.MODEL_PATH, "/models/step77")
            self.assertEqual(
                eval_stepcount.LOG_FILE_PATH,
                "/outputs/step77/pixmo-test/eval_time.log",
            )
        finally:
            for name, value in original.items():
                setattr(eval_stepcount.Config, name, value)
            eval_stepcount.LOG_FILE_PATH = ""


class StrictParsingTests(unittest.TestCase):
    def test_strict_answer_requires_closed_integer_tag(self):
        extract = eval_stepcount.ResponseParser.extract_answer
        self.assertEqual(extract("<answer>17</answer>"), "17")
        self.assertEqual(extract("<answer>-3</answer>"), "-3")
        self.assertEqual(extract("<answer>12 or 13</answer>"), "")
        self.assertEqual(extract("answer 17"), "")
        self.assertEqual(extract("<answer>17"), "")

    def test_first_complete_tag_wins(self):
        point_first = '<point>{"point_2d":[1,2]}</point><answer>1</answer>'
        text, event = eval_stepcount.truncate_at_first_complete_tag(point_first)
        self.assertEqual(event, "point")
        self.assertTrue(text.endswith("</point>"))
        self.assertNotIn("<answer>", text)
        answer_first = "<answer>2</answer><point>{}</point>"
        self.assertEqual(eval_stepcount.truncate_at_first_complete_tag(answer_first), ("<answer>2</answer>", "answer"))

    def test_protocol_cannot_be_downgraded(self):
        eval_stepcount.enforce_strict_protocol(eval_stepcount.Config)
        original = eval_stepcount.Config.ALLOW_POINT_COUNT_FALLBACK
        try:
            eval_stepcount.Config.ALLOW_POINT_COUNT_FALLBACK = True
            with self.assertRaises(ValueError):
                eval_stepcount.enforce_strict_protocol(eval_stepcount.Config)
        finally:
            eval_stepcount.Config.ALLOW_POINT_COUNT_FALLBACK = original

    def test_point_event_forces_next_round_then_answer_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            from PIL import Image

            Image.new("RGB", (8, 8), "white").save(image)
            dataset = root / "dataset.json"
            dataset.write_text("[]", encoding="utf-8")

            class TestConfig(eval_stepcount.Config):
                MODEL_PATH = str(root / "model")
                MODEL_LABEL = "step77"
                EVAL_DATASET_PATH = str(dataset)
                DATASET_ID = "pixmo-test"
                OUTPUT_DIR = str(root / "out")
                OUTPUT_IMAGES_DIR = str(root / "out" / ".working_images")
                OUTPUT_JSON_PATH = str(root / "out" / "results.json")
                MAX_ROUNDS = 13
                DOT_RADIUS = 20
                EVAL_CODE_SHA256 = "test-code"

            class FakeProcessor:
                @staticmethod
                def apply_chat_template(messages, tokenize=False, add_generation_prompt=True):
                    return "prompt"

            class FakeModel:
                def __init__(self):
                    self.processor = FakeProcessor()
                    self.responses = iter(
                        [
                            '<point>{"point_2d":[1,1],"label":"x","count_number":"1"}</point><answer>999</answer>',
                            "<answer>1</answer><point>{}</point>",
                        ]
                    )

                def generate_response(self, text, images, max_new_tokens):
                    return next(self.responses), (8, 8), (8, 8)

            class FakeAnnotator:
                @staticmethod
                def annotate_image(input_path, points, output_path, *args, **kwargs):
                    shutil.copyfile(input_path, output_path)
                    return 1

            evaluator = eval_stepcount.StepCountEvaluator.__new__(eval_stepcount.StepCountEvaluator)
            evaluator.config = TestConfig()
            evaluator.device_id = 0
            evaluator.model = FakeModel()
            evaluator.annotator = FakeAnnotator()
            evaluator.parser = eval_stepcount.ResponseParser()
            result = evaluator.evaluate_sample({"id": "x", "question": "count", "answer": 1, "image_path": str(image)})
            self.assertEqual(result["num_rounds"], 2)
            self.assertEqual(result["predicted_answer"], "1")
            self.assertTrue(result["explicit_answer_detected"])
            self.assertFalse(result["used_point_count_fallback"])
            self.assertEqual(result["last_response_event"], "answer")
            self.assertEqual(result["termination_reason"], "explicit_answer")

    def test_sample_id_cannot_escape_working_image_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            from PIL import Image

            Image.new("RGB", (8, 8), "white").save(image)
            dataset = root / "dataset.json"
            dataset.write_text("[]", encoding="utf-8")
            victim = root / "out" / "victim"
            victim.mkdir(parents=True)
            sentinel = victim / "sentinel.txt"
            sentinel.write_text("keep", encoding="utf-8")

            class TestConfig(eval_stepcount.Config):
                MODEL_PATH = str(root / "model")
                MODEL_LABEL = "step77"
                EVAL_DATASET_PATH = str(dataset)
                DATASET_ID = "pixmo-test"
                OUTPUT_DIR = str(root / "out")
                OUTPUT_IMAGES_DIR = str(root / "out" / ".working_images")
                OUTPUT_JSON_PATH = str(root / "out" / "results.json")
                MAX_ROUNDS = 13
                EVAL_CODE_SHA256 = "test-code"

            class FakeProcessor:
                @staticmethod
                def apply_chat_template(messages, tokenize=False, add_generation_prompt=True):
                    return "prompt"

            class FakeModel:
                processor = FakeProcessor()

                @staticmethod
                def generate_response(text, images, max_new_tokens):
                    return "<answer>1</answer>", (8, 8), (8, 8)

            evaluator = eval_stepcount.StepCountEvaluator.__new__(eval_stepcount.StepCountEvaluator)
            evaluator.config = TestConfig()
            evaluator.device_id = 0
            evaluator.model = FakeModel()
            evaluator.annotator = eval_stepcount.ImageAnnotator()
            evaluator.parser = eval_stepcount.ResponseParser()
            result = evaluator.evaluate_sample(
                {"id": "../victim", "question": "count", "answer": 1, "image_path": str(image)}
            )
            self.assertTrue(result["is_correct"])
            self.assertTrue(sentinel.is_file())

    def test_working_image_root_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "out"
            outside = root / "outside"
            output.mkdir()
            outside.mkdir()
            working = output / ".working_images"
            working.symlink_to(outside, target_is_directory=True)
            sentinel = outside / "sentinel.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                eval_stepcount.prepare_sample_workspace(str(output), str(working), "../chosen")
            self.assertTrue(sentinel.is_file())


class PathRemapTests(unittest.TestCase):
    def test_boundary_aware_longest_prefix(self):
        remaps = bundle_common.parse_image_remaps(["/old=/new", "/old/images=/fast/images"])
        resolved = bundle_common.remap_image_path("/old/images/a/b.png", dataset_path="/data/eval.json", remaps=remaps)
        self.assertEqual(resolved, Path("/fast/images/a/b.png"))
        boundary = bundle_common.remap_image_path("/oldish/a.png", dataset_path="/data/eval.json", remaps=remaps)
        self.assertEqual(boundary, Path("/oldish/a.png"))

    def test_relative_path_uses_explicit_root_without_basename_guess(self):
        resolved = bundle_common.remap_image_path("nested/a.png", dataset_path="/data/eval.json", image_root="/images")
        self.assertEqual(resolved, Path("/images/nested/a.png"))
        untouched = bundle_common.remap_image_path(
            "/unmapped/deep/a.png", dataset_path="/data/eval.json", image_root="/images"
        )
        self.assertEqual(untouched, Path("/unmapped/deep/a.png"))

    def test_rejects_conflicting_or_overbroad_remap(self):
        with self.assertRaises(ValueError):
            bundle_common.parse_image_remaps(["/old=/a", "/old=/b"])
        with self.assertRaises(ValueError):
            bundle_common.parse_image_remaps(["/= /new".replace(" ", "")])
        with self.assertRaises(ValueError):
            bundle_common.parse_image_remaps(["/old=relative-target"])
        with self.assertRaises(ValueError):
            bundle_common.parse_image_remaps(["/tmp/..=/new"])

    def test_relative_image_path_cannot_escape_root(self):
        with self.assertRaises(ValueError):
            bundle_common.remap_image_path("../outside.png", dataset_path="/data/eval.json", image_root="/images")

    def test_relative_image_symlink_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            outside = root / "outside"
            images.mkdir()
            outside.mkdir()
            (outside / "secret.png").write_bytes(b"x")
            (images / "link").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                bundle_common.remap_image_path(
                    "link/secret.png",
                    dataset_path=root / "dataset.json",
                    image_root=images,
                )


class CommandAndDryRunTests(unittest.TestCase):
    def make_config(self):
        return {
            "models": {"77": "/models/77", "60": "/models/60"},
            "datasets": {suite: f"/data/{suite}.json" for suite in bundle_common.SUITE_ORDER},
            "image_roots": dict.fromkeys(bundle_common.SUITE_ORDER, "/images"),
            "image_remaps": ["/source/images=/images"],
            "output_root": "/outputs",
            "num_gpus": 8,
            "workers_per_gpu": 3,
            "max_pixels": 12845056,
            "python": "python3",
            "resume": False,
        }

    def test_generated_command_order_and_fixed_parameters(self):
        commands = run_v36_eval.build_commands(self.make_config(), Path("/bundle"))
        labels = [label for label, _ in commands]
        expected_evals = [
            f"step{step}/{suite}" for step in bundle_common.MODEL_ORDER for suite in bundle_common.SUITE_ORDER
        ]
        self.assertEqual(labels, ["preflight", *expected_evals, "analyze"])
        for label, command in commands[1:-1]:
            self.assertIn("--workers-per-gpu", command)
            self.assertEqual(command[command.index("--workers-per-gpu") + 1], "3")
            self.assertIn("--image-remap", command)
            self.assertNotIn("--sample", command)
            self.assertNotIn("--allow-point-count-fallback", command)
        self.assertEqual(eval_stepcount.Config.MODEL_DTYPE, "bfloat16")
        self.assertFalse(eval_stepcount.Config.DO_SAMPLE)
        self.assertEqual(eval_stepcount.Config.NUM_BEAMS, 1)
        self.assertEqual(eval_stepcount.Config.NUM_RETURN_SEQUENCES, 1)
        self.assertEqual(eval_stepcount.Config.ADAPTIVE_MAX_ROUNDS_EXTRA, 3)

    def test_worker_part_path_does_not_rewrite_parent_suffix(self):
        result = eval_stepcount.worker_part_path("/tmp/root.json/nested/results.json", 2, 1, process_id=99)
        self.assertEqual(
            result,
            "/tmp/root.json/nested/results_part_pid99_worker2_gpu1.json",
        )

    def test_launcher_dry_run_needs_no_files_or_gpu(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = run_v36_eval.main(["--dry-run", "--workers-per-gpu", "2"])
        self.assertEqual(code, 0)
        self.assertIn("step77/pixmo-test", output.getvalue())
        self.assertIn("未加载模型/GPU", output.getvalue())

    def test_preflight_dry_run_needs_no_files_or_gpu(self):
        argv = ["--dry-run"]
        for suite in bundle_common.SUITE_ORDER:
            argv.extend(("--dataset", f"{suite}=/missing/{suite}.json"))
        argv.extend(("--model", "step77=/missing/77", "--model", "step60=/missing/60"))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(preflight.main(argv), 0)
        self.assertIn("未访问文件、模型或 GPU", output.getvalue())

    def test_hf_dry_run_needs_no_token_model_or_network(self):
        self.assertNotIn("--token", hf_release.build_parser()._option_string_actions)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                hf_release.main(
                    [
                        "--step",
                        "60",
                        "--source-dir",
                        "/missing/model",
                        "--staging-dir",
                        "/missing/staging",
                        "--dry-run",
                    ]
                ),
                0,
            )
        plan = json.loads(output.getvalue())
        self.assertTrue(plan["dry_run"])
        self.assertEqual(plan["repo_id"], hf_release.DEFAULT_REPOS["60"])
        self.assertEqual(
            plan["repo_id"],
            "SI-Lab/StepCount-7B-v36-focused10k-step60",
        )
        self.assertTrue(plan["private"])
        self.assertIn("HF_TOKEN environment only", plan["token_source"])
        self.assertEqual(plan["transfer_mode"], "low-memory-lfs")
        self.assertEqual(plan["num_workers"], 1)
        with self.assertRaisesRegex(ValueError, "正整数"):
            hf_release.main(
                [
                    "--step",
                    "60",
                    "--source-dir",
                    "/missing/model",
                    "--staging-dir",
                    "/missing/staging",
                    "--num-workers",
                    "0",
                    "--dry-run",
                ]
            )

    def test_hf_enable_xet_overrides_inherited_disable(self):
        constants = types.SimpleNamespace(HF_HUB_DISABLE_XET=True)
        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.constants = constants
        with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            with mock.patch.dict(os.environ, {"HF_HUB_DISABLE_XET": "1"}, clear=False):
                hf_release.configure_hf_transfer(True)
                self.assertNotIn("HF_HUB_DISABLE_XET", os.environ)
                self.assertFalse(constants.HF_HUB_DISABLE_XET)
                hf_release.configure_hf_transfer(False)
                self.assertEqual(os.environ["HF_HUB_DISABLE_XET"], "1")
                self.assertTrue(constants.HF_HUB_DISABLE_XET)

    def test_hf_staging_does_not_mutate_source_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            staging = root / "staging"
            source.mkdir()
            original = {
                "_name_or_path": "/internal/model/checkpoint",
                "dtype": "float32",
            }
            config_path = source / "config.json"
            config_path.write_text(json.dumps(original), encoding="utf-8")
            large_json = source / "tokenizer.json"
            large_json.write_text(json.dumps({"padding": "x" * 256}), encoding="utf-8")
            source_hashes = {path.name: bundle_common.sha256_file(path) for path in (config_path, large_json)}

            with mock.patch.object(hf_release, "LARGE_FILE_BYTES", 64):
                hf_release.prepare_staging(
                    source,
                    staging,
                    "60",
                    hf_release.DEFAULT_REPOS["60"],
                )

            self.assertEqual(
                json.loads(config_path.read_text(encoding="utf-8")),
                original,
            )
            self.assertEqual(
                source_hashes,
                {path.name: bundle_common.sha256_file(path) for path in (config_path, large_json)},
            )
            staged = json.loads((staging / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(staged["_name_or_path"], hf_release.DEFAULT_REPOS["60"])
            self.assertEqual(staged["dtype"], "bfloat16")
            self.assertTrue((staging / "README.md").is_file())

    def test_preflight_rejects_unsafe_shard_path_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "preprocessor_config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x": "../escape.safetensors"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "不安全 shard"):
                preflight.validate_model(model)

    def test_preflight_rejects_wrong_key_to_shard_placement(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "preprocessor_config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "one.safetensors", "b": "two.safetensors"}}),
                encoding="utf-8",
            )
            (model / "one.safetensors").write_bytes(b"one")
            (model / "two.safetensors").write_bytes(b"two")

            class FakeSlice:
                @staticmethod
                def get_dtype():
                    return "BF16"

            class FakeHandle:
                def __init__(self, keys):
                    self._keys = keys

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return None

                def keys(self):
                    return self._keys

                @staticmethod
                def get_slice(_key):
                    return FakeSlice()

            def fake_safe_open(path, **_kwargs):
                keys = ["b"] if Path(path).name == "one.safetensors" else ["a"]
                return FakeHandle(keys)

            fake_safetensors = types.ModuleType("safetensors")
            fake_safetensors.safe_open = fake_safe_open
            with mock.patch.dict(sys.modules, {"safetensors": fake_safetensors}):
                with self.assertRaisesRegex(ValueError, "key-to-shard"):
                    preflight.validate_model(model)

    def test_preflight_rejects_unindexed_safetensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "preprocessor_config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "model.safetensors"}}),
                encoding="utf-8",
            )
            (model / "model.safetensors").write_bytes(b"indexed")
            (model / "extra.safetensors").write_bytes(b"unexpected")
            with self.assertRaisesRegex(ValueError, "文件集与 index 不一致"):
                preflight.validate_model(model)

    def test_preflight_allows_hf_cache_shard_symlink_but_release_rejects_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "snapshot"
            blobs = root / "blobs"
            model.mkdir()
            blobs.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "preprocessor_config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "model.safetensors"}}),
                encoding="utf-8",
            )
            blob = blobs / "sha256-model"
            blob.write_bytes(b"immutable-hf-cache-blob")
            (model / "model.safetensors").symlink_to(blob)

            class FakeSlice:
                @staticmethod
                def get_dtype():
                    return "BF16"

            class FakeHandle:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return None

                @staticmethod
                def keys():
                    return ["a"]

                @staticmethod
                def get_slice(_key):
                    return FakeSlice()

            fake_safetensors = types.ModuleType("safetensors")
            fake_safetensors.safe_open = lambda *_args, **_kwargs: FakeHandle()
            with mock.patch.dict(sys.modules, {"safetensors": fake_safetensors}):
                report = preflight.validate_model(model)
            self.assertEqual(report["tensors"], 1)
            with self.assertRaisesRegex(ValueError, "shard 不允许是 symlink"):
                hf_release.validate_bf16_model(model)

    def test_hf_release_rejects_symlink_cli_roots_before_resolve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            source_link = root / "source-link"
            source_link.symlink_to(source, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "source-dir 不允许是 symlink"):
                hf_release.resolve_release_roots(source_link, root / "staging")

            staging_target = root / "staging-target"
            staging_target.mkdir()
            staging_link = root / "staging-link"
            staging_link.symlink_to(staging_target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "staging-dir 不允许是 symlink"):
                hf_release.resolve_release_roots(source, staging_link)

    def test_preflight_checks_bf16_on_every_visible_gpu(self):
        class DeviceContext:
            def __init__(self, cuda, device):
                self.cuda = cuda
                self.device = device
                self.previous = cuda.current

            def __enter__(self):
                self.cuda.current = self.device

            def __exit__(self, *args):
                self.cuda.current = self.previous

        class FakeCuda:
            current = 0

            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 2

            def device(self, device):
                return DeviceContext(self, device)

            def is_bf16_supported(self):
                return self.current == 0

            @staticmethod
            def get_device_properties(device):
                return types.SimpleNamespace(total_memory=140 * 1024**3)

        fake_torch = types.SimpleNamespace(cuda=FakeCuda())
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(RuntimeError, r"\[1\]"):
                preflight.validate_gpus(
                    expected=2,
                    workers=1,
                    model_reports={"step60": {"weight_bytes": 16 * 1024**3}},
                )

    def test_analyzer_rejects_stale_code_or_mismatched_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code_hash = bundle_common.sha256_file(BUNDLE_DIR / "eval_stepcount.py")
            row = {
                "id": "x",
                "correct_answer": 2,
                "effective_max_rounds": 5,
                "adaptive_max_rounds_extra": 3,
                "eval_protocol_version": bundle_common.PROTOCOL_VERSION,
                "eval_protocol": "oracle_min_global_gt_plus_3",
                "model_dtype": "bfloat16",
                "dataset_id": "pixmo-test",
                "model_label": "step77",
                "history_mode": 0,
                "num_beams": 1,
                "num_beam_groups": 1,
                "num_return_sequences": 1,
                "do_sample": False,
                "require_explicit_answer": True,
                "allow_point_count_fallback": False,
                "used_point_count_fallback": False,
                "stop_after_first_complete_tag": True,
                "stop_on_no_progress": False,
                "keep_eval_images": False,
                "task_cap": 13,
                "dataset_sha256": "dataset-hash",
                "num_rounds": 1,
                "output": [{"role": "model", "content": "<answer>2</answer>"}],
                "image_paths": [],
                "predicted_answer": "2",
                "explicit_answer_detected": True,
                "is_correct": True,
                "termination_reason": "explicit_answer",
                "last_response_event": "answer",
                "model_identity": "model-id",
                "eval_code_sha256": code_hash,
                "config_fingerprint": "row-fingerprint",
            }
            manifest = {
                "protocol_version": bundle_common.PROTOCOL_VERSION,
                "protocol_tag": bundle_common.PROTOCOL_TAG,
                "suite": "pixmo-test",
                "dataset_id": "pixmo-test",
                "dataset_sha256": "dataset-hash",
                "sample_ids_sha256": "ids-hash",
                "expected_samples": 1,
                "actual_samples": 1,
                "model_label": "step77",
                "dtype": "bfloat16",
                "do_sample": False,
                "num_beams": 1,
                "num_beam_groups": 1,
                "num_return_sequences": 1,
                "history_mode": 0,
                "adaptive_max_rounds": True,
                "adaptive_max_rounds_extra": 3,
                "task_cap": 13,
                "require_explicit_answer": True,
                "allow_point_count_fallback": False,
                "stop_after_first_complete_tag": True,
                "stop_on_no_progress": False,
                "keep_eval_images": False,
                "model_identity": "model-id",
                "eval_code_sha256": code_hash,
                "config_fingerprint": "manifest-fingerprint",
            }
            dataset = {
                "ids": ["x"],
                "by_id": {"x": {"answer": 2}},
                "sha256": "dataset-hash",
                "ids_sha256": "ids-hash",
            }
            (root / "results.json").write_text(json.dumps([row]), encoding="utf-8")
            manifest_path = root / "run_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            suites = {"pixmo-test": bundle_common.SuiteSpec(1, 2, 2, 13, 20)}
            with mock.patch.object(analyze_results, "SUITES", suites):
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    analyze_results.validate_result_set(
                        step="77", suite="pixmo-test", result_dir=root, dataset=dataset
                    )
                manifest["config_fingerprint"] = "row-fingerprint"
                manifest["eval_code_sha256"] = "stale-code-hash"
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "code hash"):
                    analyze_results.validate_result_set(
                        step="77", suite="pixmo-test", result_dir=root, dataset=dataset
                    )
                manifest["eval_code_sha256"] = code_hash
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                row.update(
                    {
                        "output": [{"role": "model", "content": "<answer>invalid</answer>"}],
                        "predicted_answer": "",
                        "explicit_answer_detected": False,
                        "is_correct": False,
                        "termination_reason": "explicit_answer",
                        "last_response_event": "answer",
                    }
                )
                (root / "results.json").write_text(json.dumps([row]), encoding="utf-8")
                validated = analyze_results.validate_result_set(
                    step="77", suite="pixmo-test", result_dir=root, dataset=dataset
                )
                self.assertEqual(validated["correct"], 0)
                row.update(
                    {
                        "num_rounds": 2,
                        "output": [
                            {"role": "model", "content": "<answer>2</answer>"},
                            {"role": "model", "content": "no tag"},
                        ],
                        "termination_reason": "annotation_failed",
                        "last_response_event": None,
                    }
                )
                (root / "results.json").write_text(json.dumps([row]), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "较早轮"):
                    analyze_results.validate_result_set(
                        step="77", suite="pixmo-test", result_dir=root, dataset=dataset
                    )

    def test_exact_mcnemar(self):
        self.assertEqual(analyze_results.exact_mcnemar_p(0, 0), 1.0)
        self.assertAlmostEqual(analyze_results.exact_mcnemar_p(18, 11), 0.264930896)


if __name__ == "__main__":
    unittest.main()

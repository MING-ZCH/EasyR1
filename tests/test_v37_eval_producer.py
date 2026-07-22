import hashlib
import json
import os
from pathlib import Path

import pytest

from tools import v37_eval_producer as producer

NONCE = "7" * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, arm: str = "progress"):
    run = tmp_path / "run"
    run.mkdir()
    checkpoint = run / "global_step_12"
    model = checkpoint / "actor" / "huggingface"
    model.mkdir(parents=True)
    (model / "weights.bin").write_bytes(b"weights")
    manifest = run / "v37_run_manifest.json"
    manifest.write_text(json.dumps({
        "arm": arm,
        "seed": 11,
        "final_checkpoint_path": str(checkpoint),
        "final_checkpoint_sha256": producer._tree_sha256(checkpoint),
    }), encoding="utf-8")
    output = run / ".v37_postprocess.stage"
    output.mkdir()
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text(
        """#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--manifest', required=True)
p.add_argument('--checkpoint', required=True)
p.add_argument('--model', required=True)
p.add_argument('--output', required=True)
p.add_argument('--kind', required=True)
p.add_argument('--nonce', required=True)
a=p.parse_args()
Path(a.output).write_text(json.dumps({
    'kind': a.kind,
    'seed': os.environ['PYTHONHASHSEED'],
    'producer_invocation_nonce': a.nonce,
}))
""",
        encoding="utf-8",
    )
    os.chmod(evaluator, 0o750)
    steps = {}
    for name in producer.STEP_NAMES:
        steps[name] = {
            "arms": producer.STEP_ARMS[name],
            "program": {"path": str(evaluator), "sha256": _sha(evaluator)},
            "argv": [
                "--manifest", "{manifest}", "--checkpoint", "{checkpoint}",
                "--model", "{eval_model}", "--output", "{output}", "--kind", name,
                "--nonce", "{invocation_nonce}",
            ],
            "output": producer.STEP_OUTPUTS[name],
        }
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({
        "schema_version": 1,
        "recipe_type": producer.RECIPE_TYPE,
        "environment": {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            **producer.REQUIRED_ENVIRONMENT,
        },
        "steps": steps,
    }), encoding="utf-8")
    return run, checkpoint, manifest, output, evaluator, recipe


@pytest.mark.parametrize(
    ("arm", "expected"),
    [
        ("baseline", {"paired_eval.json", producer.RECEIPT_NAME}),
        ("progress", set(producer.STEP_OUTPUTS.values()) | {producer.RECEIPT_NAME}),
    ],
)
def test_producer_runs_only_preregistered_steps_in_sealed_environment(tmp_path, arm, expected):
    run, checkpoint, manifest, output, _, recipe = _fixture(tmp_path, arm)
    report = producer.produce(
        cell_id=f"seed11-{arm}", arm=arm, seed=11, manifest=manifest,
        checkpoint=checkpoint, output_dir=output, recipe=recipe, invocation_nonce=NONCE,
    )
    assert set(item.name for item in output.iterdir()) == expected
    assert set(report["steps"]) == {
        name for name in producer.STEP_NAMES if arm in producer.STEP_ARMS[name]
    }
    emitted = [item for item in output.iterdir() if item.name != producer.RECEIPT_NAME]
    assert all(json.loads(item.read_text())["seed"] == "11" for item in emitted)
    receipt = json.loads((output / producer.RECEIPT_NAME).read_text())
    assert receipt["invocation_nonce"] == NONCE and receipt["seed"] == 11
    expected_request = producer.invocation_request_sha256(
        cell_id=f"seed11-{arm}", arm=arm, seed=11, invocation_nonce=NONCE,
        manifest_path=str(manifest), manifest_sha256=_sha(manifest),
        checkpoint_path=str(checkpoint), checkpoint_sha256=producer._tree_sha256(checkpoint),
        eval_model_path=str(checkpoint / "actor" / "huggingface"),
        eval_model_sha256=producer._tree_sha256(checkpoint / "actor" / "huggingface"),
        recipe_path=str(recipe), recipe_sha256=_sha(recipe),
        producer_path=str(Path(producer.__file__).resolve()),
        producer_sha256=_sha(Path(producer.__file__).resolve()),
    )
    assert receipt["invocation_request_sha256"] == expected_request


def test_recipe_rejects_program_drift_and_unknown_placeholder(tmp_path):
    _, checkpoint, manifest, output, evaluator, recipe = _fixture(tmp_path)
    evaluator.write_text(evaluator.read_text() + "\n# drift\n", encoding="utf-8")
    with pytest.raises(producer.EvalProducerError, match="SHA256 mismatch"):
        producer.produce(
            cell_id="seed11-progress", arm="progress", seed=11, manifest=manifest,
            checkpoint=checkpoint, output_dir=output, recipe=recipe, invocation_nonce=NONCE,
        )
    value = json.loads(recipe.read_text())
    value["steps"]["paired_eval"]["program"]["sha256"] = _sha(evaluator)
    value["steps"]["paired_eval"]["argv"].append("{unknown}")
    recipe.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(producer.EvalProducerError, match="forbidden placeholder"):
        producer.produce(
            cell_id="seed11-progress", arm="progress", seed=11, manifest=manifest,
            checkpoint=checkpoint, output_dir=output, recipe=recipe, invocation_nonce=NONCE,
        )


def test_recipe_rejects_duplicate_required_placeholder_in_one_argument(tmp_path):
    _, checkpoint, manifest, output, _, recipe = _fixture(tmp_path)
    value = json.loads(recipe.read_text())
    value["steps"]["paired_eval"]["argv"][1] = "{manifest}:{manifest}"
    recipe.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(producer.EvalProducerError, match=r"\{manifest\} exactly once"):
        producer.produce(
            cell_id="seed11-progress", arm="progress", seed=11, manifest=manifest,
            checkpoint=checkpoint, output_dir=output, recipe=recipe, invocation_nonce=NONCE,
        )


def test_recipe_rejects_online_or_non_eight_gpu_environment(tmp_path):
    _, checkpoint, manifest, output, _, recipe = _fixture(tmp_path)
    value = json.loads(recipe.read_text())
    value["environment"]["WANDB_MODE"] = "online"
    recipe.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(producer.EvalProducerError, match="WANDB_MODE"):
        producer.produce(
            cell_id="seed11-progress", arm="progress", seed=11, manifest=manifest,
            checkpoint=checkpoint, output_dir=output, recipe=recipe, invocation_nonce=NONCE,
        )
    value["environment"]["WANDB_MODE"] = "offline"
    value["environment"]["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    recipe.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(producer.EvalProducerError, match="exactly eight"):
        producer.produce(
            cell_id="seed11-progress", arm="progress", seed=11, manifest=manifest,
            checkpoint=checkpoint, output_dir=output, recipe=recipe, invocation_nonce=NONCE,
        )


def test_evaluator_executes_verified_snapshot_after_original_path_is_replaced(
    tmp_path, monkeypatch,
):
    _, checkpoint, manifest, output, evaluator, recipe = _fixture(tmp_path, "baseline")
    write_snapshot = producer._write_execution_snapshot
    replaced = False

    def snapshot_then_replace(directory, name, raw):
        nonlocal replaced
        snapshot = write_snapshot(directory, name, raw)
        if name.endswith(".program.snapshot") and not replaced:
            evaluator.write_text(
                "#!/usr/bin/env python3\nraise SystemExit(91)\n", encoding="utf-8",
            )
            os.chmod(evaluator, 0o750)
            replaced = True
        return snapshot

    monkeypatch.setattr(producer, "_write_execution_snapshot", snapshot_then_replace)
    producer.produce(
        cell_id="seed11-baseline", arm="baseline", seed=11, manifest=manifest,
        checkpoint=checkpoint, output_dir=output, recipe=recipe, invocation_nonce=NONCE,
    )

    assert replaced is True
    paired = json.loads((output / "paired_eval.json").read_text())
    assert paired["kind"] == "paired_eval"

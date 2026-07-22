import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
WORKER_SOURCE = REPO / "verl" / "workers" / "fsdp_workers.py"
ACTOR_SOURCE = REPO / "verl" / "workers" / "actor" / "dp_actor.py"
TRAINER_SOURCE = REPO / "verl" / "trainer" / "ray_trainer.py"


def _load_scheduler_decision():
    tree = ast.parse(WORKER_SOURCE.read_text(encoding="utf-8"))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_adaptive_actor_scheduler_should_step"
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(WORKER_SOURCE), "exec"), namespace)
    return namespace[node.name]


def _load_function(path: Path, name: str, namespace=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
    values = dict(namespace or {})
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), values)
    return values[name]


def test_adaptive_scheduler_advances_only_after_executed_update():
    should_step = _load_scheduler_decision()
    assert should_step({"actor/optimizer_steps_executed": [1.0]}) is True
    assert should_step({"actor/optimizer_steps_executed": [0.0]}) is False


@pytest.mark.parametrize(
    "value", [None, [], [float("nan")], [-1.0], [0.5], [1.0, 1.0], [True], ["1"]]
)
def test_adaptive_scheduler_counter_fails_closed(value):
    should_step = _load_scheduler_decision()
    metrics = {} if value is None else {"actor/optimizer_steps_executed": value}
    with pytest.raises(RuntimeError):
        should_step(metrics)


def test_v36_nonadaptive_scheduler_path_remains_unconditional():
    source = WORKER_SOURCE.read_text(encoding="utf-8")
    assert '_adaptive_actor_scheduler_should_step(metrics) if adaptive_actor_kl else True' in source
    assert "if scheduler_advanced:\n                self.lr_scheduler.step()" in source


def test_skipped_update_does_not_apply_cooldown_factor_twice():
    reconcile = _load_function(WORKER_SOURCE, "_reconcile_spike_cooldown_lr")
    actor = SimpleNamespace(_spike_cooldown_remaining=3, _original_lrs=[1.0], _spike_lr_factor=0.1)
    optimizer = SimpleNamespace(param_groups=[{"lr": 0.1}])
    assert reconcile(actor, optimizer, False) == pytest.approx(0.1)
    assert actor._original_lrs == [1.0]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_scheduler_update_rebases_and_reapplies_one_cooldown_factor():
    reconcile = _load_function(WORKER_SOURCE, "_reconcile_spike_cooldown_lr")
    actor = SimpleNamespace(_spike_cooldown_remaining=2, _original_lrs=[1.0], _spike_lr_factor=0.1)
    optimizer = SimpleNamespace(param_groups=[{"lr": 0.8}])
    assert reconcile(actor, optimizer, True) == pytest.approx(0.08)
    assert actor._original_lrs == [0.8]


def test_zero_ema_never_marks_the_next_finite_gradient_as_spike():
    decision = _load_function(
        ACTOR_SOURCE,
        "_is_grad_spike_against_ema",
        {"Optional": __import__("typing").Optional},
    )
    assert decision(1.0, None, 3.0) is False
    assert decision(1.0, 0.0, 3.0) is True
    assert decision(1.0, 0.0, 3.0, ignore_zero_ema=True) is False
    assert decision(4.0, 1.0, 3.0) is True


def test_adaptive_actor_rpc_requires_exactly_one_optimizer_attempt():
    validate = _load_function(ACTOR_SOURCE, "_validate_adaptive_optimizer_rpc_counts")
    assert validate(1, 1) is None
    assert validate(1, 0) is None
    for attempted, executed in ((0, 0), (2, 2), (2, 1), (1, 2), (1, -1)):
        with pytest.raises(RuntimeError, match="exactly one optimizer attempt"):
            validate(attempted, executed)


def test_v36_does_not_attach_strict_actor_runtime_checkpoint_callbacks():
    source = ACTOR_SOURCE.read_text(encoding="utf-8")
    assert 'if actor_optimizer is not None and bool(getattr(config, "adaptive_actor_kl", False)):' in source
    assert 'else total_loss.detach().item()' in source


def test_trainer_counter_rejects_bool_and_string_coercion():
    read = _load_function(
        TRAINER_SOURCE,
        "strict_optimizer_counter_metrics",
        {"np": np, "Dict": dict, "Any": object},
    )
    good = {
        "actor/optimizer_counter_agreement": 1.0,
        "actor/optimizer_steps_attempted": 1.0,
        "actor/optimizer_steps_executed": 1.0,
        "actor/optimizer_steps_skipped": 0.0,
    }
    assert read(good) == (1, 1, 0)
    for field in ("actor/optimizer_steps_attempted", "actor/optimizer_steps_executed"):
        for invalid in (True, "1"):
            broken = dict(good)
            broken[field] = invalid
            with pytest.raises(RuntimeError):
                read(broken)


def test_checkpoint_skip_path_has_module_logger():
    source = TRAINER_SOURCE.read_text(encoding="utf-8")
    assert "logger = logging.getLogger(__name__)" in source

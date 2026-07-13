import tempfile
import unittest
from pathlib import Path

from tools.monitor_v36_training import estimate_eta, parse_log


class MonitorV36TrainingTest(unittest.TestCase):
    def test_estimate_eta_prefers_reported_step_time(self) -> None:
        state = {"completed_step_times": {"1": 100.0, "2": 100.0, "3": 100.0}}
        metrics = {
            "1": {"time_per_step": 2400.0},
            "2": {"time_per_step": 2500.0},
            "3": {"time_per_step": 2600.0},
        }

        eta = estimate_eta(state, total_steps=80, step_metrics=metrics)

        self.assertIn("median_step=41.7 min", eta)
        self.assertIn("done=3/80", eta)
        self.assertIn("source=log_time_per_step", eta)

    def test_estimate_eta_ignores_zero_observation_intervals(self) -> None:
        state = {"completed_step_times": {"1": 100.0, "2": 100.0}}

        eta = estimate_eta(state, total_steps=80)

        self.assertEqual(eta, "ETA pending: completed step intervals unavailable.")

    def test_parse_log_extracts_reported_step_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "train.log"
            log_path.write_text(
                "\x1b[36m(Runner pid=42)\x1b[0m Step 1\n"
                "\x1b[36m(Runner pid=42)\x1b[0m time_per_step: 2521.557\n",
                encoding="utf-8",
            )

            parsed, _ = parse_log(log_path, set())

        self.assertEqual(parsed["max_completed_step"], 1)
        self.assertEqual(parsed["step_metrics"]["1"]["time_per_step"], 2521.557)


if __name__ == "__main__":
    unittest.main()
